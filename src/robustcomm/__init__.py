"""RobustComm: Byzantine-resilient learned communication for cooperative MARL.

Implements the paper's robust message-space aggregation (RobAgg: Krum / coordinate-wise
median / trimmed mean), the surrogate-based inter-agent message attack, MADDPG-style
ensembles, minimax adversarial communication training on a synthetic environment, the
Section-9 theoretical bounds, and the Section-10 evaluation metrics.

The core (aggregators, attacks, theory, metrics) is NumPy-only, deterministic under a
fixed seed, and requires no GPU or internet. Real environments (DIAL / MPE / SMAC) are
provided as adapters and are intended to run on EC2 after manual data provisioning.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Public API (kept import-light; submodules import lazily to avoid cycles).
from .aggregators.registry import build_aggregator, AGGREGATORS  # noqa: E402
from .theory import bounds  # noqa: E402

__all__ = ["__version__", "build_aggregator", "AGGREGATORS", "bounds"]
