from logsentinel.metrics import grouping_accuracy
from logsentinel.parsing import WILDCARD, Drain


def test_same_template_merges_and_generalizes():
    d = Drain(depth=4, sim_threshold=0.5)
    a = d.add("Receiving block blk_1 src: /10.0.0.1:5000")
    b = d.add("Receiving block blk_2 src: /10.0.0.7:5000")
    assert a.cluster_id == b.cluster_id
    assert d.clusters[0].template[2] == WILDCARD
    assert d.clusters[0].size == 2


def test_different_lengths_never_merge():
    d = Drain()
    a = d.add("connection closed")
    b = d.add("connection closed by peer")
    assert a.cluster_id != b.cluster_id


def test_masks_are_applied():
    d = Drain(masks=[r"blk_-?\d+"])
    assert d.tokenize("deleting blk_-123 now") == ["deleting", WILDCARD, "now"]


def test_match_does_not_mutate():
    d = Drain()
    d.add("user alice logged in")
    n = len(d.clusters)
    assert d.match("totally different message here ok") is None
    assert len(d.clusters) == n
    assert d.match("user alice logged out") is not None


def test_grouping_accuracy():
    assert grouping_accuracy([1, 1, 2, 2], [9, 9, 8, 8]) == 1.0
    # predicted group 9 mixes two true groups -> those 3 messages are wrong
    assert grouping_accuracy([1, 1, 2, 3], [9, 9, 9, 8]) == 0.25
