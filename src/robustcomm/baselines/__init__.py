"""Baselines B1-B5 (Sec 10.5) and external-defense adapters."""
from .base import BaselineSpec, build_defense, standard_baselines
from .no_defense import NoCommAggregator
from .message_signing import MessageSigningComparator
from .external_wrappers import ExternalDefenseAdapter, DummyExternalDefense

__all__ = [
    "BaselineSpec", "build_defense", "standard_baselines", "NoCommAggregator",
    "MessageSigningComparator", "ExternalDefenseAdapter", "DummyExternalDefense",
]
