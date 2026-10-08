"""Abstract robust aggregator (RobAgg, Alg 1)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..utils.validation import validate_messages


class RobustAggregator(ABC):
    """Maps an ``(n_i, d)`` array of inbound messages to a single ``(d,)`` aggregate."""

    name: str = "base"

    @abstractmethod
    def aggregate(self, messages: np.ndarray) -> np.ndarray:
        """Return the robust aggregate of the inbound messages."""

    def guarantee_holds(self, n: int) -> bool:
        """Whether the operator's Byzantine-regime guarantee holds for ``n`` senders."""
        return True

    def __call__(self, messages: np.ndarray) -> np.ndarray:
        return self.aggregate(validate_messages(messages))
