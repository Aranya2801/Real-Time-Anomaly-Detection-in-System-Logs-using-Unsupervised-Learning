"""
Replay the HDFS sample as a live stream and measure how early anomalies get caught.

The first half of the blocks (by first appearance) is the training set, the rest
is streamed one line at a time from the cut point, exactly as it would arrive.
Train blocks that keep logging after the cut are ignored for scoring.

Two warm-up modes:
  offline  DeepLog is trained on the *complete* training sessions, i.e. you
           trained on historical logs and then deployed. This is the normal setup.
  prefix   DeepLog only sees what was logged before the cut. HDFS blocks live a
           long time (median ~5.5k lines, p90 ~75k lines in this sample), so most
           warm-up sessions are cut off mid-way and the model learns that sessions
           can end anywhere. Reported to show why the offline mode matters.

    python experiments/streaming_replay.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from logsentinel.data import BLOCK_RE, read_hdfs_contents
from logsentinel.models import DeepLog
from logsentinel.parsing import PARSER_SETTINGS, Drain
from logsentinel.stream import StreamingDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="data/HDFS/HDFS_100k.log_structured.csv")
    ap.add_argument("--labels", default="data/HDFS/anomaly_label.csv")
    ap.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True,
                    help="drop labelled anomalies from the warm-up data")
    ap.add_argument("--warmup", choices=["offline", "prefix"], default="offline")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    contents = read_hdfs_contents(args.logs).tolist()
    labels = pd.read_csv(args.labels).set_index("BlockId")["Label"].eq("Anomaly")

    # find the cut: the line where the first test block appears
    first_seen = {}
    for i, msg in enumerate(contents):
        for b in BLOCK_RE.findall(msg):
            first_seen.setdefault(b, i)
    blocks = [b for b in first_seen if b in labels.index]
    train_blocks = set(blocks[: len(blocks) // 2])
    cut = first_seen[blocks[len(blocks) // 2]]

    parser = Drain(**PARSER_SETTINGS["HDFS"])
    sessions = {}
    for msg in contents[:cut]:
        cid = parser.add(msg).cluster_id
        for b in set(BLOCK_RE.findall(msg)):
            sessions.setdefault(b, []).append(cid)
    if args.warmup == "offline":
        # complete the training sessions with a throwaway copy of the parser so the
        # live parser's state at the cut is untouched
        import copy
        p2 = copy.deepcopy(parser)
        for msg in contents[cut:]:
            bl = [b for b in set(BLOCK_RE.findall(msg)) if b in train_blocks]
            if bl:
                cid = p2.add(msg).cluster_id
                for b in bl:
                    sessions.setdefault(b, []).append(cid)
        # template ids created after the cut by p2 don't exist in the live parser
        known = len(parser.clusters)
        sessions = {b: [c if c < known else -1 for c in s] for b, s in sessions.items()}
    train = [s for b, s in sessions.items()
             if b in train_blocks and not (args.clean and labels.get(b, False))]
    model = DeepLog(seed=0).fit(train)

    det = StreamingDetector(parser, model, ignore_sessions=train_blocks)
    alerts = []
    t0 = time.perf_counter()
    for msg in contents[cut:]:
        alerts += det.process(msg)
    stream_secs = time.perf_counter() - t0
    alerts += det.flush()

    test_blocks = [b for b in blocks if b not in train_blocks]
    final_len = {b: len(det.closed[b].history) for b in test_blocks if b in det.closed}
    first_alert = {}
    for a in alerts:
        first_alert.setdefault(a.session, a)

    y = np.array([labels[b] for b in test_blocks])
    flagged = np.array([b in first_alert for b in test_blocks])
    tp = flagged & y
    early = [first_alert[b].position < final_len[b] for b in np.array(test_blocks)[tp]]
    frac_seen = [first_alert[b].position / final_len[b] for b in np.array(test_blocks)[tp]]
    reasons = pd.Series([first_alert[b].reason for b in np.array(test_blocks)[tp]]).value_counts()

    summary = dict(
        warmup=args.warmup,
        warmup_sessions=len(train),
        test_blocks=len(test_blocks),
        test_anomalies=int(y.sum()),
        lines_streamed=len(contents) - cut,
        lines_per_second=int((len(contents) - cut) / stream_secs),
        precision=round(float(tp.sum() / max(flagged.sum(), 1)), 3),
        recall=round(float(tp.sum() / y.sum()), 3),
        caught_before_session_end=round(float(np.mean(early)), 3),
        median_fraction_of_session_seen=round(float(np.median(frac_seen)), 3),
        alert_reasons={k: int(v) for k, v in reasons.items()},
    )
    print(json.dumps(summary, indent=2))
    args.out.mkdir(exist_ok=True)
    (args.out / f"streaming_replay_{args.warmup}.json").write_text(json.dumps(summary, indent=2))
    with open(args.out / f"streaming_alerts_{args.warmup}.txt", "w") as fh:
        for a in alerts[:40]:
            fh.write(f"{a}  [label={'anomaly' if labels.get(a.session) else 'normal'}]\n")


if __name__ == "__main__":
    main()
