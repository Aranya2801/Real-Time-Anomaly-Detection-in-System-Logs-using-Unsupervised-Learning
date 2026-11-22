from __future__ import annotations

import numpy as np


class EventCounter:
    """
    Turns template-id sequences into an event count matrix.

    Columns are the templates seen during fit, plus one extra column that
    counts templates never seen in training (new templates showing up in
    production are themselves a decent anomaly signal).

    term_weighting='tfidf' and normalize='zero-mean' reproduce the
    preprocessing from Xu et al. (SOSP'09) that PCA expects.
    """

    def __init__(self, term_weighting: str | None = None, normalize: str | None = None):
        self.term_weighting = term_weighting
        self.normalize = normalize

    def _count(self, sequences) -> np.ndarray:
        X = np.zeros((len(sequences), len(self.vocab_) + 1), dtype=np.float64)
        oov = len(self.vocab_)
        for i, seq in enumerate(sequences):
            for e in seq:
                X[i, self.vocab_.get(e, oov)] += 1
        return X

    def fit(self, sequences):
        events = sorted({e for s in sequences for e in s})
        self.vocab_ = {e: i for i, e in enumerate(events)}
        X = self._count(sequences)
        if self.term_weighting == "tfidf":
            df = (X > 0).sum(axis=0)
            self.idf_ = np.log(len(X) / (df + 1e-8))
            self.idf_[df == 0] = np.log(len(X))  # unseen column gets the max weight
            X = X * self.idf_
        if self.normalize == "zero-mean":
            self.mean_ = X.mean(axis=0)
        return self

    def transform(self, sequences) -> np.ndarray:
        X = self._count(sequences)
        if self.term_weighting == "tfidf":
            X = X * self.idf_
        if self.normalize == "zero-mean":
            X = X - self.mean_
        return X

    def fit_transform(self, sequences) -> np.ndarray:
        return self.fit(sequences).transform(sequences)
