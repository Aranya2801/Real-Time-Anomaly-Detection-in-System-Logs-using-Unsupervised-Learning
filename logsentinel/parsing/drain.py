"""
Drain: an online log parser with a fixed-depth parse tree.

Reimplementation of He et al., "Drain: An Online Log Parsing Approach with
Fixed Depth Tree" (ICWS 2017). I wrote this from the paper instead of pulling
in drain3 because I wanted to (a) understand the algorithm properly and
(b) have a parser that works one line at a time, which the streaming detector
needs.

Tree layout:
    root -> [token count] -> [first token] -> ... -> [(depth-2)th token] -> clusters

Tokens that contain digits are routed to a wildcard child, since they are
almost always parameters. Inside a leaf, a message joins the most similar
cluster if similarity >= sim_threshold, otherwise it starts a new one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

WILDCARD = "<*>"


@dataclass
class LogCluster:
    template: list[str]
    cluster_id: int
    size: int = 1

    @property
    def template_str(self) -> str:
        return " ".join(self.template)


@dataclass
class _Node:
    children: dict[str, _Node] = field(default_factory=dict)
    clusters: list[LogCluster] = field(default_factory=list)


def _has_digit(s: str) -> bool:
    return any(c.isdigit() for c in s)


class Drain:
    def __init__(
        self,
        depth: int = 4,
        sim_threshold: float = 0.5,
        max_children: int = 100,
        masks: list[str] | None = None,
    ):
        if depth < 3:
            raise ValueError("depth must be >= 3 (root + length layer + at least one token layer)")
        self.depth = depth
        self.sim_threshold = sim_threshold
        self.max_children = max_children
        self.masks = [re.compile(m) for m in (masks or [])]
        self.root = _Node()
        self.clusters: list[LogCluster] = []

    # ---- preprocessing --------------------------------------------------

    def tokenize(self, line: str) -> list[str]:
        for m in self.masks:
            line = m.sub(WILDCARD, line)
        return line.strip().split()

    # ---- tree search ----------------------------------------------------

    def _leaf(self, tokens: list[str], create: bool) -> _Node | None:
        node = self.root.children.get(str(len(tokens)))
        if node is None:
            if not create:
                return None
            node = self.root.children[str(len(tokens))] = _Node()

        for i, tok in enumerate(tokens):
            if i >= self.depth - 2:
                break
            key = WILDCARD if _has_digit(tok) else tok
            if key in node.children:
                node = node.children[key]
            elif WILDCARD in node.children and not create:
                node = node.children[WILDCARD]
            elif not create:
                return None
            else:
                # new branch: respect max_children by falling back to the wildcard
                if key != WILDCARD and len(node.children) >= self.max_children - 1:
                    key = WILDCARD
                if key not in node.children:
                    node.children[key] = _Node()
                node = node.children[key]
        return node

    @staticmethod
    def _similarity(template: list[str], tokens: list[str]) -> tuple[float, int]:
        same = 0
        params = 0
        for t, s in zip(template, tokens):
            if t == WILDCARD:
                params += 1
            elif t == s:
                same += 1
        return same / len(template), params

    def _best_match(self, clusters: list[LogCluster], tokens: list[str]) -> LogCluster | None:
        best, best_sim, best_params = None, -1.0, -1
        for c in clusters:
            sim, params = self._similarity(c.template, tokens)
            if sim > best_sim or (sim == best_sim and params > best_params):
                best, best_sim, best_params = c, sim, params
        if best is not None and best_sim >= self.sim_threshold:
            return best
        return None

    # ---- public API -----------------------------------------------------

    def add(self, line: str) -> LogCluster:
        """Parse one log message, updating the tree. Returns its cluster."""
        tokens = self.tokenize(line)
        if not tokens:
            tokens = [""]
        leaf = self._leaf(tokens, create=True)
        cluster = self._best_match(leaf.clusters, tokens)
        if cluster is None:
            cluster = LogCluster(template=list(tokens), cluster_id=len(self.clusters))
            leaf.clusters.append(cluster)
            self.clusters.append(cluster)
        else:
            cluster.template = [
                t if t == s else WILDCARD for t, s in zip(cluster.template, tokens)
            ]
            cluster.size += 1
        return cluster

    def match(self, line: str) -> LogCluster | None:
        """Look up a message without changing the tree (used at inference time)."""
        tokens = self.tokenize(line) or [""]
        leaf = self._leaf(tokens, create=False)
        if leaf is None:
            return None
        return self._best_match(leaf.clusters, tokens)

    def parse_all(self, lines) -> list[int]:
        return [self.add(line).cluster_id for line in lines]

    def templates(self) -> dict[int, str]:
        return {c.cluster_id: c.template_str for c in self.clusters}
