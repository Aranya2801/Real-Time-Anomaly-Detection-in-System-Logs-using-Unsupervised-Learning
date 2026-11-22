from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def grouping_accuracy(truth, pred) -> float:
    """
    Grouping accuracy (GA) from Zhu et al. 2019.

    A message counts as correctly parsed only if the set of messages that
    share its predicted group is exactly the set that shares its true group.
    """
    df = pd.DataFrame({"t": list(truth), "p": list(pred)})
    correct = 0
    for _, grp in df.groupby("p"):
        true_ids = grp["t"].unique()
        if len(true_ids) == 1 and (df["t"] == true_ids[0]).sum() == len(grp):
            correct += len(grp)
    return correct / len(df)


def detection_report(y_true, scores, threshold: float | None = None) -> dict:
    """
    Threshold-free metrics (ROC-AUC, PR-AUC), the best achievable F1 over all
    thresholds (an optimistic upper bound, reported separately so nobody mistakes
    it for a deployable number), and P/R/F1 at a given threshold if one is passed.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    out = {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "best_f1": best_f1(y_true, scores),
    }
    if threshold is not None:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, (scores > threshold).astype(int), average="binary", zero_division=0
        )
        out.update(precision=p, recall=r, f1=f)
    return out


def best_f1(y_true, scores) -> float:
    order = np.argsort(-scores)
    y = np.asarray(y_true)[order]
    s = np.asarray(scores)[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    # only evaluate at positions where the score actually changes
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp = tp[last], fp[last]
    fn = y.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(f1.max())
