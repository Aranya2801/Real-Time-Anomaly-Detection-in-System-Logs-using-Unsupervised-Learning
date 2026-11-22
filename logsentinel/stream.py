"""
Line-at-a-time anomaly detection.

The batch experiments score a session after it is complete. In practice you
want to know as soon as something looks off, so this keeps per-session state
and checks every new event against DeepLog's next-event prediction as it
arrives. A session is also checked when it goes idle (did the model expect it
to end here?).
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field

from .models.deep import DeepLog
from .parsing import Drain


@dataclass
class Alert:
    session: str
    position: int  # how many events into the session the alert fired
    reason: str  # "unexpected event" | "unseen template" | "ended early"
    template: str
    surprise: float
    line_no: int

    def __str__(self):
        return (f"[line {self.line_no}] {self.session} event #{self.position}: {self.reason} "
                f"(surprise={self.surprise:.1f}) :: {self.template}")


@dataclass
class _Session:
    history: list[int] = field(default_factory=list)
    last_seen: int = 0
    alerted: bool = False


class StreamingDetector:
    def __init__(self, parser: Drain, model: DeepLog, session_regex: str = r"(blk_-?\d+)",
                 idle_lines: int | None = None, ignore_sessions=None):
        self.parser = parser
        self.model = model
        self.session_re = re.compile(session_regex)
        self.idle_lines = idle_lines
        self.ignore = set(ignore_sessions or ())
        self.sessions: dict[str, _Session] = {}
        self.line_no = 0
        self.closed: dict[str, _Session] = {}

    def process(self, message: str) -> list[Alert]:
        self.line_no += 1
        n_before = len(self.parser.clusters)
        cluster = self.parser.add(message)
        is_new_template = cluster.cluster_id >= n_before
        alerts = []
        for sid in set(self.session_re.findall(message)):
            if sid in self.ignore:
                continue
            s = self.sessions.setdefault(sid, _Session())
            if not s.alerted:
                bad, nll = self.model.check_next(s.history, cluster.cluster_id)
                if bad:
                    s.alerted = True
                    reason = ("unseen template" if is_new_template or cluster.cluster_id not in
                              self.model.key_map_ else "unexpected event")
                    alerts.append(Alert(sid, len(s.history) + 1, reason, cluster.template_str,
                                        nll, self.line_no))
            s.history.append(cluster.cluster_id)
            s.last_seen = self.line_no
        if self.idle_lines and self.line_no % 1000 == 0:
            alerts += self._close_idle()
        return alerts

    def _close(self, sid) -> Alert | None:
        s = self.sessions.pop(sid)
        self.closed[sid] = s
        if s.alerted:
            return None
        bad, nll = self.model.check_next(s.history, None)
        if bad:
            s.alerted = True
            return Alert(sid, len(s.history), "ended early", "<end of session>", nll, self.line_no)
        return None

    def _close_idle(self) -> list[Alert]:
        stale = [k for k, s in self.sessions.items() if self.line_no - s.last_seen > self.idle_lines]
        return [a for a in map(self._close, stale) if a]

    def flush(self) -> list[Alert]:
        """Close every open session (end of input)."""
        return [a for a in map(self._close, list(self.sessions)) if a]


def main(argv=None):
    """Tail a file / stdin:  python -m logsentinel.stream model.pt < app.log"""
    import argparse
    import pickle

    from .data import HDFS_LINE_RE

    ap = argparse.ArgumentParser(description="Stream logs through a trained detector.")
    ap.add_argument("model", help="pickle produced by `python -m logsentinel.train`")
    ap.add_argument("logfile", nargs="?", default="-")
    ap.add_argument("--idle-lines", type=int, default=50_000)
    args = ap.parse_args(argv)

    with open(args.model, "rb") as fh:
        bundle = pickle.load(fh)
    det = StreamingDetector(bundle["parser"], bundle["model"], idle_lines=args.idle_lines)
    src = sys.stdin if args.logfile == "-" else open(args.logfile, errors="replace")
    t0, n = time.time(), 0
    for line in src:
        m = HDFS_LINE_RE.match(line.rstrip("\n"))
        for a in det.process(m.group(6) if m else line.strip()):
            print(a, flush=True)
        n += 1
    for a in det.flush():
        print(a)
    print(f"# {n} lines in {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
