"""Metric formulas and edge cases (Sec. 10.6)."""
import pytest
import math

import numpy as np

from robustcomm.aggregators import build_aggregator
from robustcomm.metrics import (
    aggregate_deviation,
    aggregation_overhead,
    attack_success_rate,
    bootstrap_ci,
    empirical_tolerance_fraction,
    reward_under_attack,
)


def test_reward_under_attack_mean_and_empty():
    assert reward_under_attack([0.2, 0.4, 0.6]) == pytest.approx(0.4)
    assert math.isnan(reward_under_attack([]))


def test_empirical_tolerance_fraction():
    fractions = [0.0, 0.1, 0.2, 0.3]
    rewards = [[0.9], [0.8], [0.6], [0.4]]  # threshold 0.7 met up to phi=0.1
    assert empirical_tolerance_fraction(fractions, rewards, tau=0.7) == 0.1
    # nothing meets a very high tau -> 0.0
    assert empirical_tolerance_fraction(fractions, rewards, tau=0.99) == 0.0


def test_attack_success_rate():
    clean = [1.0, 1.0, 1.0, 1.0]
    adv = [1.0, 0.5, 0.85, 0.2]  # drops of 0.0, 0.5, 0.15, 0.8 -> 3 of 4 exceed 0.1
    assert attack_success_rate(clean, adv, drop_threshold=0.1) == 0.75
    assert math.isnan(attack_success_rate([1.0], [], 0.1))  # length mismatch


def test_aggregate_deviation():
    assert aggregate_deviation(np.array([3.0, 4.0]), np.array([0.0, 0.0])) == 5.0


def test_bootstrap_ci_bounds_contain_point():
    rng = np.random.default_rng(0)
    data = rng.normal(0.5, 0.1, size=200)
    ci = bootstrap_ci(data, n_boot=300, rng=rng)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["n"] == 200


def test_bootstrap_ci_degenerate():
    assert bootstrap_ci([]).get("n") == 0
    one = bootstrap_ci([0.7])
    assert one["lo"] == one["hi"] == one["point"] == 0.7


def test_aggregation_overhead_reports_complexity():
    M = np.zeros((7, 4))
    info = aggregation_overhead(build_aggregator("krum", f=2), M, repeats=10)
    assert info["operator"] == "krum"
    assert "n_i^2" in info["complexity"]
    assert info["seconds_per_call"] >= 0.0
