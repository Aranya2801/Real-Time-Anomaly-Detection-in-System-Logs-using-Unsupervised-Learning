"""
Neural detectors (PyTorch, CPU is fine for HDFS-sized data).

DeepLog   - Du et al., CCS'17. LSTM language model over template ids; a session
            is anomalous if some observed event is not in the model's top-g
            predictions given the previous h events.
AutoEncoder - plain MLP autoencoder on log-scaled event counts, scored by
            reconstruction error. A simple neural baseline for comparison with PCA.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from ..features import EventCounter
from .classical import BaseDetector


def _seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


class _NextKeyLSTM(nn.Module):
    def __init__(self, n_keys: int, emb: int = 32, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(n_keys, emb)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, n_keys)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.out(h[:, -1])


class DeepLog(BaseDetector):
    name = "DeepLog"

    def __init__(self, window: int = 10, top_g: int = 9, epochs: int = 15, batch_size: int = 512,
                 lr: float = 2e-3, hidden: int = 64, use_eos: bool = True, seed: int = 0,
                 verbose: bool = False):
        self.window, self.top_g, self.use_eos = window, top_g, use_eos
        self.epochs, self.batch_size, self.lr, self.hidden = epochs, batch_size, lr, hidden
        self.seed, self.verbose = seed, verbose

    # key 0 = start/padding, key 1 = template never seen in training, key 2 = end of session.
    #
    # The END key is my addition to DeepLog. Vanilla DeepLog only checks events that
    # do arrive, so a session that silently stops early (a very common HDFS failure:
    # the block is allocated and then nothing happens) can never be flagged. Asking
    # the model to also predict "session ends here" fixes that.
    START, UNK, END = 0, 1, 2

    def _encode(self, seq):
        return [self.key_map_.get(e, self.UNK) for e in seq]

    def _windows(self, sequences):
        xs, ys, owner = [], [], []
        for i, seq in enumerate(sequences):
            keys = [self.START] * self.window + self._encode(seq)
            if self.use_eos:
                keys.append(self.END)
            for t in range(self.window, len(keys)):
                xs.append(keys[t - self.window:t])
                ys.append(keys[t])
                owner.append(i)
        return (torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long),
                np.asarray(owner))

    def fit(self, sequences):
        _seed_everything(self.seed)
        events = sorted({e for s in sequences for e in s})
        self.key_map_ = {e: i + 3 for i, e in enumerate(events)}
        self.n_keys_ = len(events) + 3
        X, y, _ = self._windows(sequences)

        self.model_ = _NextKeyLSTM(self.n_keys_, hidden=self.hidden)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model_.train()
        for ep in range(self.epochs):
            perm = torch.randperm(len(X))
            total = 0.0
            for s in range(0, len(X), self.batch_size):
                idx = perm[s:s + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(X[idx]), y[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            if self.verbose:
                print(f"  deeplog epoch {ep + 1}/{self.epochs} loss={total / len(X):.4f}")
        self.model_.eval()
        self.threshold_ = 0.5  # predict() uses the top-g rule, see below
        return self

    @torch.no_grad()
    def _position_stats(self, sequences):
        X, y, owner = self._windows(sequences)
        nll = np.empty(len(X))
        miss = np.empty(len(X), dtype=bool)
        for s in range(0, len(X), 4096):
            logits = self.model_(X[s:s + 4096])
            logp = torch.log_softmax(logits, dim=1)
            yy = y[s:s + 4096]
            nll[s:s + 4096] = -logp[torch.arange(len(yy)), yy].numpy()
            topg = logits.topk(min(self.top_g, self.n_keys_), dim=1).indices
            miss[s:s + 4096] = ~(topg == yy[:, None]).any(dim=1).numpy()
        unk = (y == self.UNK).numpy()
        miss |= unk
        nll[unk] = max(nll.max(), 20.0)  # unseen template: treat as maximally surprising
        return nll, miss, owner

    def score(self, sequences):
        nll, _, owner = self._position_stats(sequences)
        out = np.zeros(len(sequences))
        np.maximum.at(out, owner, nll)
        return out

    def predict(self, sequences):
        _, miss, owner = self._position_stats(sequences)
        out = np.zeros(len(sequences), dtype=int)
        np.maximum.at(out, owner, miss.astype(int))
        return out

    @torch.no_grad()
    def check_next(self, history: list[int], event: int | None) -> tuple[bool, float]:
        """
        Streaming use: given the raw template ids seen so far in a session and the
        newly arrived one, return (is_anomalous, surprise in nats).
        Pass event=None when the session closes, to check the END prediction.
        """
        keys = [self.START] * self.window + self._encode(history)
        x = torch.tensor([keys[-self.window:]], dtype=torch.long)
        if event is None:
            if not self.use_eos:
                return False, 0.0
            k = self.END
        else:
            k = self.key_map_.get(event, self.UNK)
        if k == self.UNK:
            return True, 20.0
        logits = self.model_(x)[0]
        topg = logits.topk(min(self.top_g, self.n_keys_)).indices
        nll = float(-torch.log_softmax(logits, 0)[k])
        return bool(k not in topg), nll


class AutoEncoder(BaseDetector):
    name = "AutoEncoder"

    def __init__(self, hidden: int = 32, bottleneck: int = 8, epochs: int = 100,
                 lr: float = 1e-3, quantile: float = 0.99, seed: int = 0):
        self.hidden, self.bottleneck, self.epochs = hidden, bottleneck, epochs
        self.lr, self.quantile, self.seed = lr, quantile, seed

    def _X(self, sequences, fit=False):
        X = np.log1p(self.fe_.transform(sequences))
        X = self.scaler_.fit_transform(X) if fit else self.scaler_.transform(X)
        return torch.tensor(X, dtype=torch.float32)

    def fit(self, sequences):
        _seed_everything(self.seed)
        self.fe_ = EventCounter().fit(sequences)
        self.scaler_ = StandardScaler()
        X = self._X(sequences, fit=True)
        d = X.shape[1]
        self.net_ = nn.Sequential(
            nn.Linear(d, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.bottleneck), nn.ReLU(),
            nn.Linear(self.bottleneck, self.hidden), nn.ReLU(), nn.Linear(self.hidden, d),
        )
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-5)
        for _ in range(self.epochs):
            perm = torch.randperm(len(X))
            for s in range(0, len(X), 256):
                b = X[perm[s:s + 256]]
                opt.zero_grad()
                loss = ((self.net_(b) - b) ** 2).mean()
                loss.backward()
                opt.step()
        self.net_.eval()
        self.threshold_ = float(np.quantile(self._err(X), self.quantile))
        return self

    @torch.no_grad()
    def _err(self, X):
        return ((self.net_(X) - X) ** 2).mean(dim=1).numpy()

    def score(self, sequences):
        return self._err(self._X(sequences))
