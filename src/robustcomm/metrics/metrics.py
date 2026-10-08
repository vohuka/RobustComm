"""Evaluation metrics (Sec. 10.6 / Problem formulation).

All metrics are pure functions over arrays/records with numerically safe edge-case
handling (empty inputs return ``nan`` rather than raising). None of them fabricate data;
they summarise whatever the experiment runner measured.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from ..aggregators.base import RobustAggregator

_COMPLEXITY = {
    "krum": "O(n_i^2 * d)",
    "median": "O(n_i * d)",
    "trmean": "O(n_i * d)",
    "mean": "O(n_i * d)",
}


def reward_under_attack(adv_returns: Sequence[float]) -> float:
    """J_adv: mean team return under the (strongest evaluated) attacker."""
    a = np.asarray(adv_returns, dtype=np.float64)
    return float(a.mean()) if a.size else float("nan")


def empirical_tolerance_fraction(
    fractions: Sequence[float], adv_returns_by_fraction: Sequence[Sequence[float]], tau: float
) -> float:
    """Largest malicious fraction ``phi=f/n`` whose mean reward-under-attack >= ``tau``.

    Returns 0.0 if no fraction meets the target (edge case), scanning all fractions
    rather than assuming monotonicity.
    """
    ok = [
        float(f)
        for f, r in zip(fractions, adv_returns_by_fraction)
        if np.asarray(r, dtype=np.float64).size and float(np.mean(r)) >= tau
    ]
    return max(ok) if ok else 0.0


def attack_success_rate(
    clean_returns: Sequence[float], adv_returns: Sequence[float], drop_threshold: float = 0.1
) -> float:
    """Fraction of paired episodes whose return dropped by at least ``drop_threshold``.

    Pairing requires equal-length inputs (same seeds/episodes under clean vs attack).
    """
    c = np.asarray(clean_returns, dtype=np.float64)
    a = np.asarray(adv_returns, dtype=np.float64)
    if c.size == 0 or a.size == 0 or c.size != a.size:
        return float("nan")
    return float(np.mean((c - a) >= drop_threshold))


def aggregate_deviation(aggregate: np.ndarray, reference: np.ndarray, order: int | float = 2) -> float:
    """``||m_hat - reference||`` (diagnostic validating Lem 9.2/9.3)."""
    return float(np.linalg.norm(np.asarray(aggregate) - np.asarray(reference), ord=order))


def tradeoff_points(
    clean_returns_by_op: dict[str, Sequence[float]], adv_returns_by_op: dict[str, Sequence[float]]
) -> list[dict[str, float]]:
    """Robustness--clean-reward tradeoff points, one per operator/method."""
    pts = []
    for op in clean_returns_by_op:
        c = np.asarray(clean_returns_by_op[op], dtype=np.float64)
        a = np.asarray(adv_returns_by_op.get(op, []), dtype=np.float64)
        pts.append(
            {
                "method": op,
                "clean_reward": float(c.mean()) if c.size else float("nan"),
                "adv_reward": float(a.mean()) if a.size else float("nan"),
            }
        )
    return pts


def aggregation_overhead(
    aggregator: RobustAggregator, messages: np.ndarray, repeats: int = 200
) -> dict[str, object]:
    """Measure per-call wall-clock and report the asymptotic complexity (RQ6)."""
    repeats = max(1, int(repeats))
    t0 = time.perf_counter()
    for _ in range(repeats):
        aggregator.aggregate(messages)
    seconds = (time.perf_counter() - t0) / repeats
    return {
        "operator": aggregator.name,
        "seconds_per_call": float(seconds),
        "n_senders": int(messages.shape[0]),
        "msg_dim": int(messages.shape[1]),
        "complexity": _COMPLEXITY.get(aggregator.name, "unknown"),
    }
