"""Operator dispatch for RobAgg (Alg 1)."""
from __future__ import annotations

from .base import RobustAggregator
from .krum import KrumAggregator
from .mean import MeanAggregator
from .median import MedianAggregator
from .trimmed_mean import TrimmedMeanAggregator

AGGREGATORS = {
    "krum": KrumAggregator,
    "median": MedianAggregator,
    "trmean": TrimmedMeanAggregator,
    "mean": MeanAggregator,
}


def build_aggregator(operator: str, f: int = 0, beta: float = 0.2) -> RobustAggregator:
    """Construct a RobAgg operator by name.

    ``krum`` and ``median`` take the assumed Byzantine count ``f``; ``trmean`` takes the
    trim level ``beta`` (and ``f`` for its guarantee check); ``mean`` takes neither.
    """
    operator = operator.lower()
    if operator == "krum":
        return KrumAggregator(f=f)
    if operator == "median":
        return MedianAggregator(f=f)
    if operator == "trmean":
        return TrimmedMeanAggregator(beta=beta, f=f)
    if operator == "mean":
        return MeanAggregator()
    raise ValueError(f"unknown operator '{operator}'; expected one of {sorted(AGGREGATORS)}.")
