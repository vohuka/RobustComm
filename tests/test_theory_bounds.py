"""Section-9 bound functions and the empirical inequality they assert."""
import math

import numpy as np
import pytest

from robustcomm.aggregators import build_aggregator
from robustcomm.theory import bounds


def test_krum_factor_matches_paper_worked_example():
    # Paper App. B worked example: n=7, f=2 -> beta_K = 2*sqrt(3)+1.
    assert math.isclose(bounds.krum_deviation_factor(7, 2), 2 * math.sqrt(3) + 1)


def test_krum_factor_regime_error():
    with pytest.raises(ValueError):
        bounds.krum_deviation_factor(5, 2)  # 2*2+2=6 !< 5


def test_median_bounds():
    assert bounds.median_deviation_bound(4, 0.5, "linf") == 0.5
    assert math.isclose(bounds.median_deviation_bound(4, 0.5, "l2"), 2 * 0.5)


def test_episodic_factor_gamma_one_is_horizon():
    assert bounds.episodic_factor(1.0, 50) == 50.0
    assert math.isclose(bounds.episodic_factor(0.99, 50), (1 - 0.99**50) / (1 - 0.99))


def test_tolerance_counts_match_worked_example():
    assert bounds.max_tolerated_byzantine("krum", 7) == 2
    assert bounds.max_tolerated_byzantine("median", 7) == 3
    # boundary: n=6, krum needs f<2 -> f<=1 (strict inequality edge case)
    assert bounds.max_tolerated_byzantine("krum", 6) == 1


def test_deviation_bound_holds_empirically():
    """Realised deviation of RobAgg never exceeds the Lem 9.2/9.3 bound; mean is inf."""
    rng = np.random.default_rng(3)
    n, d, B, f = 9, 4, 0.5, 2
    for _ in range(50):
        # honest messages within l2 ball radius B around origin
        dirs = rng.normal(size=(n, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        radii = B * rng.uniform(0, 1, size=(n, 1)) ** (1 / d)
        M = dirs * radii
        ref = M.mean(axis=0)
        M[:f] = rng.uniform(-20, 20, size=(f, d))  # Byzantine outliers
        for op in ["krum", "median", "trmean"]:
            agg = build_aggregator(op, f=f, beta=0.3)
            dev = np.linalg.norm(agg.aggregate(M) - ref)
            bnd = bounds.aggregate_deviation_bound(op, n, f, d, B)
            assert dev <= bnd + 1e-9
    assert bounds.aggregate_deviation_bound("mean", n, f, d, B) == math.inf


def test_return_degradation_budget_independent():
    b1 = bounds.return_degradation_bound("krum", 9, 2, 4, 0.5, 1.0, 1.0, 0.99, 20)
    # bound does not depend on epsilon (no epsilon argument); positive and finite.
    assert b1["per_step"] > 0 and math.isfinite(b1["episodic"])
