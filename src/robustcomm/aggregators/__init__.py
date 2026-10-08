"""Robust message-aggregation operators (RobAgg, Alg 1 / Def 3.1-3.2)."""
from .base import RobustAggregator
from .krum import KrumAggregator
from .mean import MeanAggregator
from .median import MedianAggregator
from .trimmed_mean import TrimmedMeanAggregator
from .registry import AGGREGATORS, build_aggregator

__all__ = [
    "RobustAggregator",
    "KrumAggregator",
    "MeanAggregator",
    "MedianAggregator",
    "TrimmedMeanAggregator",
    "AGGREGATORS",
    "build_aggregator",
]
