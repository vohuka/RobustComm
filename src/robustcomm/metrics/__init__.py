"""Evaluation metrics (Sec 10.6) and bootstrap confidence intervals."""
from .metrics import (
    reward_under_attack,
    empirical_tolerance_fraction,
    attack_success_rate,
    aggregate_deviation,
    tradeoff_points,
    aggregation_overhead,
)
from .confidence_intervals import bootstrap_ci

__all__ = [
    "reward_under_attack", "empirical_tolerance_fraction", "attack_success_rate",
    "aggregate_deviation", "tradeoff_points", "aggregation_overhead", "bootstrap_ci",
]
