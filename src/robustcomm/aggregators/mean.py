"""Mean aggregation — the DIAL / undefended baseline (Prop 9.1).

A single Byzantine sender can drive this to any point (demonstrated in
``tests/test_aggregators.py::test_mean_unbounded_vs_robagg``). It is included as the
baseline the paper argues against, and as a component of the trimmed mean at ``beta=0``.
"""
from __future__ import annotations

import numpy as np

from ..utils.validation import validate_messages
from .base import RobustAggregator


class MeanAggregator(RobustAggregator):
    name = "mean"

    def aggregate(self, messages: np.ndarray) -> np.ndarray:
        M = validate_messages(messages)
        return M.mean(axis=0)

    def guarantee_holds(self, n: int) -> bool:  # noqa: ARG002
        # The mean has breakdown point 0: one Byzantine sender suffices (Prop 9.1).
        return False
