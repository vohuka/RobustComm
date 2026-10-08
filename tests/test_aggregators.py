"""Correctness of robust aggregators (Def 3.1-3.2) and Prop 9.1 demonstration."""
import numpy as np
import pytest

from robustcomm.aggregators import (
    KrumAggregator,
    MeanAggregator,
    MedianAggregator,
    TrimmedMeanAggregator,
    build_aggregator,
)


def test_mean_is_coordinatewise_mean():
    M = np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]])
    assert np.allclose(MeanAggregator().aggregate(M), [2.0, 4.0])


def test_median_is_coordinatewise_median():
    M = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    assert np.allclose(MedianAggregator().aggregate(M), [1.0, 20.0])


def test_krum_returns_an_input_message():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(7, 3))
    out = KrumAggregator(f=2).aggregate(M)
    # Krum selects an actual inbound message.
    assert any(np.allclose(out, M[i]) for i in range(M.shape[0]))


def test_krum_selects_central_message():
    # 5 tight points + 2 far outliers; Krum must pick one of the tight cluster.
    M = np.array([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1], [0.05, 0.05],
                  [50.0, 50.0], [-50.0, -50.0]], dtype=float)
    out = KrumAggregator(f=2).aggregate(M)
    assert np.linalg.norm(out) < 1.0  # from the tight cluster, not an outlier


def test_krum_regime_violation_raises():
    M = np.zeros((5, 2))
    with pytest.raises(ValueError):
        KrumAggregator(f=2).aggregate(M)  # 2*2+2=6 !< 5


def test_trimmed_mean_removes_extremes():
    M = np.array([[-100.0], [1.0], [2.0], [3.0], [100.0]])
    out = TrimmedMeanAggregator(beta=0.2).aggregate(M)  # drop 1 each side
    assert np.allclose(out, [2.0])


def test_trimmed_mean_too_much_raises():
    M = np.zeros((3, 2))
    with pytest.raises(ValueError):
        TrimmedMeanAggregator(beta=0.49, f=0).aggregate(M)  # trims everything


def test_mean_unbounded_vs_robagg():
    """Prop 9.1: one Byzantine sender moves the mean without bound; RobAgg stays bounded."""
    rng = np.random.default_rng(1)
    honest = rng.normal(0, 0.1, size=(6, 3))
    ref = honest.mean(axis=0)
    krum = KrumAggregator(f=1)
    median = MedianAggregator(f=1)
    prev_mean_dev = 0.0
    for scale in [1e1, 1e3, 1e5]:
        M = np.vstack([honest, np.full((1, 3), scale)])  # 1 Byzantine at 'scale'
        mean_dev = np.linalg.norm(MeanAggregator().aggregate(M) - ref)
        krum_dev = np.linalg.norm(krum.aggregate(M) - ref)
        med_dev = np.linalg.norm(median.aggregate(M) - ref)
        assert mean_dev > prev_mean_dev  # grows without bound
        assert krum_dev < 1.0 and med_dev < 1.0  # stays bounded regardless of scale
        prev_mean_dev = mean_dev


def test_build_aggregator_dispatch():
    for op in ["krum", "median", "trmean", "mean"]:
        agg = build_aggregator(op, f=1, beta=0.2)
        assert agg.name == op
    with pytest.raises(ValueError):
        build_aggregator("nope")
