"""
Evaluate my Drain implementation on the 16 LogHub 2k samples.

    python experiments/parser_benchmark.py --loghub data/loghub

Writes results/parser_accuracy.csv and prints a markdown table.
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from logsentinel.metrics import grouping_accuracy
from logsentinel.parsing import PARSER_SETTINGS, Drain


def run(loghub_dir: Path) -> pd.DataFrame:
    rows = []
    for name, cfg in PARSER_SETTINGS.items():
        f = loghub_dir / name / f"{name}_2k.log_structured.csv"
        if not f.exists():
            print(f"skip {name}: {f} not found")
            continue
        df = pd.read_csv(f)
        parser = Drain(**cfg)
        t0 = time.perf_counter()
        pred = parser.parse_all(df["Content"].astype(str))
        elapsed = time.perf_counter() - t0
        rows.append(
            dict(
                dataset=name,
                true_templates=df["EventId"].nunique(),
                found_templates=len(parser.clusters),
                grouping_accuracy=round(grouping_accuracy(df["EventId"], pred), 3),
                lines_per_sec=int(len(df) / elapsed),
            )
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loghub", type=Path, default=Path("data/loghub"))
    ap.add_argument("--out", type=Path, default=Path("results/parser_accuracy.csv"))
    args = ap.parse_args()

    res = run(args.loghub)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(res.to_markdown(index=False))
    print(f"\nmean grouping accuracy: {res.grouping_accuracy.mean():.3f}")
