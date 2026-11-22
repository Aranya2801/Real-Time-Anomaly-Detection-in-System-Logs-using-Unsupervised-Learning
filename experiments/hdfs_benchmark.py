"""
Main HDFS experiment.

Question: these methods are all called "unsupervised", but most papers train
them on normal-only data, which needs labels. What happens when the training
window contains some anomalies, like it would in a real deployment?

For each contamination rate we keep all normal training sessions, add a random
sample of training anomalies, fit every detector, and evaluate on the
chronologically later test half. Thresholds are picked without labels
(PCA: Q-statistic, DeepLog: top-g, others: 99th percentile of training scores).

    python experiments/hdfs_benchmark.py --seeds 3

Outputs go to results/.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from logsentinel.data import chronological_split, load_hdfs
from logsentinel.metrics import detection_report
from logsentinel.models import DETECTORS, DeepLog

STOCHASTIC = {"IsolationForest", "AutoEncoder", "DeepLog"}


def indistinguishable_mask(data, test_idx):
    """Test anomalies whose exact event sequence also occurs as a *normal* session."""
    normal = {tuple(s) for s, y in zip(data.sequences, data.labels) if y == 0}
    return np.array([data.labels[i] == 1 and tuple(data.sequences[i]) in normal for i in test_idx])


def make_train(train, rate, rng):
    normal = np.where(train.labels == 0)[0]
    anom = np.where(train.labels == 1)[0]
    if rate == "natural":
        return train
    n_anom = int(round(rate * len(normal) / (1 - rate)))
    n_anom = min(n_anom, len(anom))
    pick = rng.choice(anom, n_anom, replace=False) if n_anom else np.array([], dtype=int)
    idx = np.sort(np.r_[normal, pick])
    return train.subset(idx)


def evaluate(model, test, hidden):
    scores = model.score(test.sequences)
    pred = model.predict(test.sequences)
    y = test.labels
    rep = detection_report(y, scores)
    keep = ~hidden
    return dict(
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        roc_auc=rep["roc_auc"],
        pr_auc=rep["pr_auc"],
        best_f1=rep["best_f1"],
        f1_distinguishable=f1_score(y[keep], pred[keep], zero_division=0),
        recall_distinguishable=recall_score(y[keep], pred[keep], zero_division=0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="data/HDFS/HDFS_100k.log_structured.csv")
    ap.add_argument("--labels", default="data/HDFS/anomaly_label.csv")
    ap.add_argument("--max-lines", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rates", default="0,0.005,0.01,0.02,natural")
    ap.add_argument("--models", default=",".join(DETECTORS))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = load_hdfs(args.logs, args.labels, args.max_lines)
    tr, te = chronological_split(len(data))
    train, test = data.subset(tr), data.subset(te)
    hidden = indistinguishable_mask(data, te)

    info = dict(
        sessions=len(data), anomalies=int(data.labels.sum()), templates=len(data.parser.clusters),
        train_sessions=len(train), train_anomalies=int(train.labels.sum()),
        test_sessions=len(test), test_anomalies=int(test.labels.sum()),
        test_anomalies_indistinguishable=int(hidden.sum()),
        natural_contamination=round(float(train.labels.mean()), 4),
    )
    print(json.dumps(info, indent=2))
    (args.out / "dataset_info.json").write_text(json.dumps(info, indent=2))

    rates = [r if r == "natural" else float(r) for r in args.rates.split(",")]
    rows = []
    for rate in rates:
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            tr_set = make_train(train, rate, rng)
            actual = float(tr_set.labels.mean())
            for name in args.models.split(","):
                if seed > 0 and name not in STOCHASTIC and rate in (0, "natural"):
                    continue  # deterministic model on a deterministic training set
                kw = {"seed": seed} if name in STOCHASTIC else {}
                t = time.time()
                m = DETECTORS[name](**kw).fit(tr_set.sequences)
                res = evaluate(m, test, hidden)
                res.update(model=name, rate=str(rate), contamination=round(actual, 4), seed=seed,
                           fit_seconds=round(time.time() - t, 2))
                rows.append(res)
                print(f"rate={rate!s:8s} seed={seed} {name:16s} F1={res['f1']:.3f} "
                      f"PR-AUC={res['pr_auc']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "hdfs_contamination_runs.csv", index=False)

    # ---- DeepLog ablation: end-of-session token on/off, clean vs natural training
    abl = []
    for eos in (False, True):
        for rate in (0, "natural"):
            for seed in range(args.seeds):
                tr_set = make_train(train, rate, np.random.default_rng(seed))
                m = DeepLog(use_eos=eos, seed=seed).fit(tr_set.sequences)
                r = evaluate(m, test, hidden)
                r.update(use_eos=eos, rate=str(rate), seed=seed)
                abl.append(r)
                # top-g sensitivity, reusing the same trained model
                if rate == 0:
                    for g in (1, 3, 5, 7, 9, 12, 15):
                        m.top_g = g
                        rg = evaluate(m, test, hidden)
                        rg.update(use_eos=eos, rate="topg", top_g=g, seed=seed)
                        abl.append(rg)
                    m.top_g = 9
    pd.DataFrame(abl).to_csv(args.out / "deeplog_ablation_runs.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
