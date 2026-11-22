"""
Small synthetic sanity checks. Normal sessions follow one of two workflows;
anomalies either stop early or contain an event normal sessions never have.
"""

import numpy as np
import pytest

from logsentinel.metrics import best_f1
from logsentinel.models import DETECTORS, DeepLog
from logsentinel.parsing import Drain
from logsentinel.stream import StreamingDetector

WORKFLOWS = [[0, 1, 2, 2, 2, 3, 4], [0, 1, 2, 2, 2, 3, 5, 4]]


def make_data(n_normal=300, seed=0):
    rng = np.random.default_rng(seed)
    normal = [list(WORKFLOWS[rng.integers(2)]) for _ in range(n_normal)]
    anomalies = [[0, 1], [0, 1, 2], [0, 1, 2, 6, 2, 3, 4], [0, 1, 7, 7, 7]] * 5
    return normal, anomalies


@pytest.mark.parametrize("name", list(DETECTORS))
def test_detector_ranks_anomalies_higher(name):
    normal, anomalies = make_data()
    kw = {"epochs": 5} if name == "DeepLog" else {}
    if name == "AutoEncoder":
        kw = {"epochs": 30}
    m = DETECTORS[name](**kw).fit(normal)
    test = normal[:50] + anomalies
    y = np.r_[np.zeros(50), np.ones(len(anomalies))]
    s = m.score(test)
    assert s.shape == (len(test),)
    assert np.mean(s[y == 1]) > np.mean(s[y == 0])
    assert set(np.unique(m.predict(test))) <= {0, 1}


def test_deeplog_eos_catches_truncated_sessions():
    normal, _ = make_data()
    truncated = [[0, 1], [0, 1, 2]]
    with_eos = DeepLog(epochs=8, top_g=2, use_eos=True).fit(normal)
    without = DeepLog(epochs=8, top_g=2, use_eos=False).fit(normal)
    assert with_eos.predict(truncated).all()
    assert not without.predict(truncated).any()


def test_streaming_flags_unseen_template_immediately():
    parser = Drain()
    msgs = ["start job s1", "load data s1", "write output s1", "finish job s1"]
    seqs = []
    for i in range(50):
        seqs.append([parser.add(m.replace("s1", f"s{i}")).cluster_id for m in msgs])
    model = DeepLog(epochs=10).fit(seqs)
    det = StreamingDetector(parser, model, session_regex=r"\b(s\d+)\b")
    assert det.process("start job s999") == []
    alerts = det.process("kernel panic while handling s999 disk failure")
    assert len(alerts) == 1 and alerts[0].reason == "unseen template"


def test_best_f1():
    assert best_f1(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
