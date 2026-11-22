"""
Loading and sessionizing HDFS logs.

HDFS logs are grouped into sessions by block id: every line that mentions
blk_xxx belongs to that block's session, and labels are given per block.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .parsing import PARSER_SETTINGS, Drain

BLOCK_RE = re.compile(r"(blk_-?\d+)")
# raw HDFS_v1 line: "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block ..."
HDFS_LINE_RE = re.compile(r"^(\d{6}) (\d{6}) (\d+) (\w+) ([^:]+): (.*)$")


@dataclass
class SessionData:
    session_ids: list[str]
    sequences: list[list[int]]  # template ids in arrival order
    labels: np.ndarray  # 1 = anomaly
    parser: Drain

    def __len__(self):
        return len(self.session_ids)

    def subset(self, idx) -> SessionData:
        idx = np.asarray(idx)
        return SessionData(
            [self.session_ids[i] for i in idx],
            [self.sequences[i] for i in idx],
            self.labels[idx],
            self.parser,
        )


def read_hdfs_contents(path: Path) -> pd.Series:
    """Return the message part of each line, from either a raw .log or a LogPAI structured csv."""
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, usecols=["Content"])["Content"].astype(str)
    contents = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = HDFS_LINE_RE.match(line.rstrip("\n"))
            contents.append(m.group(6) if m else line.strip())
    return pd.Series(contents)


def load_hdfs(log_path: Path, label_path: Path, max_lines: int | None = None) -> SessionData:
    contents = read_hdfs_contents(log_path)
    if max_lines:
        contents = contents.iloc[:max_lines]

    parser = Drain(**PARSER_SETTINGS["HDFS"])
    sessions: OrderedDict[str, list[int]] = OrderedDict()
    for msg in contents:
        # parse first: the parser masks block ids, so we pull them from the raw text
        cid = parser.add(msg).cluster_id
        for blk in set(BLOCK_RE.findall(msg)):
            sessions.setdefault(blk, []).append(cid)

    labels = pd.read_csv(label_path).set_index("BlockId")["Label"]
    ids = [b for b in sessions if b in labels.index]
    y = (labels.reindex(ids) == "Anomaly").astype(int).to_numpy()
    return SessionData(ids, [sessions[b] for b in ids], y, parser)


def chronological_split(n: int, train_frac: float = 0.5):
    """
    Sessions are already ordered by first appearance, so a prefix split keeps
    the test set strictly 'in the future' relative to training. Random splits on
    HDFS leak near-duplicate sessions across the boundary and inflate scores.
    """
    cut = int(n * train_frac)
    return np.arange(cut), np.arange(cut, n)
