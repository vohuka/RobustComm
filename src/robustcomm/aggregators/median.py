"""Coordinate-wise median aggregation (Def 3.2, Alg 1).

Computable for any inputs, but the deviation guarantee (Lem 9.3) holds only under the
majority regime ``f/n_i < 1/2``. When the regime is violated we log a warning and report
``guarantee_holds == False`` in metadata rather than failing, since the median is still a
well-defined estimator.
"""
from __future__ import annotations

import numpy as np

from ..utils.logging import get_logger
from ..utils.validation import validate_messages
from .base import RobustAggregator

_log = get_logger(__name__)


class MedianAggregator(RobustAggregator):
    name = "median"

    def __init__(self, f: int = 0) -> None:
        self.f = int(f)

    def guarantee_holds(self, n: int) -> bool:
        return self.f < n / 2.0

    def aggregate(self, messages: np.ndarray) -> np.ndarray:
        """Per-coordinate median. Even ``n`` averages the two central order statistics
        (NumPy default), which still lies within the honest range, so Lem 9.3 holds."""
        M = validate_messages(messages)
        if not self.guarantee_holds(M.shape[0]):
            _log.warning(
                "median: f/n >= 1/2 (f=%d, n=%d); deviation guarantee (Lem 9.3) is void.",
                self.f,
                M.shape[0],
            )
        return np.median(M, axis=0)
