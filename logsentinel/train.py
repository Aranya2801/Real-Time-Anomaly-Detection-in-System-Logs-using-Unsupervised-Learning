"""
Train a parser + DeepLog model on HDFS logs and save them for streaming.

    python -m logsentinel.train data/HDFS/HDFS_100k.log_structured.csv \
        --labels data/HDFS/anomaly_label.csv --out models/hdfs_deeplog.pkl

If --labels is given, sessions labelled as anomalies are dropped from training
(the "clean" setting). Without labels it trains on everything, which is the
honest unsupervised setting but noticeably worse, see results/.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from .data import BLOCK_RE, load_hdfs, read_hdfs_contents
from .models import DeepLog
from .parsing import PARSER_SETTINGS, Drain


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("--labels")
    ap.add_argument("--max-lines", type=int)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="models/hdfs_deeplog.pkl")
    args = ap.parse_args(argv)

    if args.labels:
        data = load_hdfs(args.logs, args.labels, args.max_lines)
        keep = np.where(data.labels == 0)[0]
        seqs, parser = [data.sequences[i] for i in keep], data.parser
    else:
        parser = Drain(**PARSER_SETTINGS["HDFS"])
        sessions = {}
        contents = read_hdfs_contents(args.logs)
        if args.max_lines:
            contents = contents.iloc[:args.max_lines]
        for msg in contents:
            cid = parser.add(msg).cluster_id
            for b in set(BLOCK_RE.findall(msg)):
                sessions.setdefault(b, []).append(cid)
        seqs = list(sessions.values())

    print(f"training DeepLog on {len(seqs)} sessions, {len(parser.clusters)} templates")
    model = DeepLog(epochs=args.epochs, verbose=True).fit(seqs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"parser": parser, "model": model}, fh)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
