"""
Count-vector based detectors.

Every detector follows the same small interface:
    fit(sequences)          -> self
    score(sequences)        -> np.ndarray, higher = more anomalous
    threshold_              -> decision threshold picked WITHOUT labels
    predict(sequences)      -> 0/1
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ..features import EventCounter


class BaseDetector:
    name = "base"

    def predict(self, sequences) -> np.ndarray:
        return (self.score(sequences) > self.threshold_).astype(int)


class PCADetector(BaseDetector):
    """
    Xu et al., SOSP'09. Project count vectors onto the residual subspace and use
    the squared prediction error (SPE). The threshold is the Q-statistic from
    Jackson & Mudholkar (1979), so it needs no labels at all.
    """

    name = "PCA"

    def __init__(self, var_ratio: float = 0.95, alpha: float = 0.001):
        self.var_ratio = var_ratio
        self.alpha = alpha

    def fit(self, sequences):
        self.fe_ = EventCounter(term_weighting="tfidf", normalize="zero-mean")
        X = self.fe_.fit_transform(sequences)
        cov = X.T @ X / max(len(X) - 1, 1)
        eigval, eigvec = np.linalg.eigh(cov)
        eigval, eigvec = eigval[::-1], eigvec[:, ::-1]
        eigval = np.clip(eigval, 0, None)
        k = int(np.searchsorted(np.cumsum(eigval) / eigval.sum(), self.var_ratio) + 1)
        P = eigvec[:, :k]
        self.k_ = k
        self.C_ = np.eye(X.shape[1]) - P @ P.T

        lam = eigval[k:]
        phi = [np.sum(lam ** i) for i in (1, 2, 3)]
        if phi[0] <= 0:  # degenerate: all variance explained
            self.threshold_ = 0.0
            return self
        h0 = 1.0 - 2 * phi[0] * phi[2] / (3.0 * phi[1] ** 2)
        c = norm.ppf(1 - self.alpha)
        self.threshold_ = phi[0] * (
            c * np.sqrt(2 * phi[1] * h0 ** 2) / phi[0] + 1.0 + phi[1] * h0 * (h0 - 1.0) / phi[0] ** 2
        ) ** (1.0 / h0)
        return self

    def score(self, sequences):
        X = self.fe_.transform(sequences)
        r = X @ self.C_
        return np.einsum("ij,ij->i", r, r)


class _SklearnDetector(BaseDetector):
    """Shared plumbing: log-scaled counts -> standardize -> sklearn model."""

    def __init__(self, quantile: float = 0.99):
        # label-free threshold: flag anything scoring above this quantile of the training scores
        self.quantile = quantile

    def _make_model(self):
        raise NotImplementedError

    def _features(self, sequences, fit=False):
        X = np.log1p(self.fe_.transform(sequences))
        return self.scaler_.fit_transform(X) if fit else self.scaler_.transform(X)

    def fit(self, sequences):
        self.fe_ = EventCounter().fit(sequences)
        self.scaler_ = StandardScaler()
        X = self._features(sequences, fit=True)
        self.model_ = self._make_model().fit(X)
        self.threshold_ = float(np.quantile(self._raw_score(X), self.quantile))
        return self

    def _raw_score(self, X):
        # sklearn convention is "higher = more normal", so flip it
        return -self.model_.score_samples(X)

    def score(self, sequences):
        return self._raw_score(self._features(sequences))


class IForestDetector(_SklearnDetector):
    name = "IsolationForest"

    def __init__(self, n_estimators=200, quantile=0.99, seed=0):
        super().__init__(quantile)
        self.n_estimators, self.seed = n_estimators, seed

    def _make_model(self):
        return IsolationForest(n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1)


class OCSVMDetector(_SklearnDetector):
    name = "OneClassSVM"

    def __init__(self, nu=0.05, gamma="scale", quantile=0.99):
        super().__init__(quantile)
        self.nu, self.gamma = nu, gamma

    def _make_model(self):
        return OneClassSVM(nu=self.nu, gamma=self.gamma)


class LOFDetector(_SklearnDetector):
    name = "LOF"

    def __init__(self, n_neighbors=20, quantile=0.99):
        super().__init__(quantile)
        self.n_neighbors = n_neighbors

    def _make_model(self):
        return LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)

    def _raw_score(self, X):
        return -self.model_.score_samples(X)


class RareEventRule(BaseDetector):
    """
    Deliberately dumb baseline: a session is anomalous if it contains a template
    that appears in fewer than `min_support` of the training sessions.

    I added this after noticing that on the HDFS sample almost every detectable
    anomaly contains an event type that normal sessions never produce. If a
    learned model can't beat this, it isn't learning much about the sequence.
    Using a support threshold (rather than "never seen") keeps it usable when
    the training data itself contains some anomalies.
    """

    name = "RareEvent"

    def __init__(self, min_support: float = 0.05):
        self.min_support = min_support

    def fit(self, sequences):
        n = len(sequences)
        counts: dict[int, int] = {}
        for s in sequences:
            for e in set(s):
                counts[e] = counts.get(e, 0) + 1
        self.support_ = {e: c / n for e, c in counts.items()}
        self.floor_ = 1.0 / (n + 1)
        self.threshold_ = -np.log(self.min_support)
        return self

    def score(self, sequences):
        # surprise of the rarest event in the session
        return np.array([
            max((-np.log(self.support_.get(e, self.floor_)) for e in set(s)), default=0.0)
            for s in sequences
        ])
