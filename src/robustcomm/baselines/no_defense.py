"""Undefended references (B1, B2)."""
from __future__ import annotations

import numpy as np

from ..aggregators.base import RobustAggregator
from ..utils.validation import validate_messages


class NoCommAggregator(RobustAggregator):
    """No-communication lower reference (B2): the agent receives no usable signal.

    Returns the zero vector regardless of inbound messages, modelling an agent that
    ignores the (untrusted) channel entirely.
    """

    name = "nocomm"

    def __init__(self, msg_dim: int) -> None:
        self.d = int(msg_dim)

    def guarantee_holds(self, n: int) -> bool:  # noqa: ARG002
        return True  # trivially unaffected by Byzantine messages (but uninformative)

    def aggregate(self, messages: np.ndarray) -> np.ndarray:
        validate_messages(messages)
        return np.zeros(self.d, dtype=np.float64)
