"""Admissible perturbation set and message-attack interface (Def 5.1, Eq pertset).

The admissible set is
``Delta_f = { (delta_j) : |B| <= f, ||delta_j|| <= eps }`` (Eq. pertset). A Byzantine
sender may add any perturbation of bounded norm to its own outgoing message; honest
messages are untouched.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..aggregators.base import RobustAggregator


class PerturbationSet:
    """Norm-ball projection for the admissible perturbation set (Eq. pertset)."""

    def __init__(self, epsilon: float, norm: str = "linf") -> None:
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0.")
        if norm not in ("linf", "l2"):
            raise ValueError("norm must be 'linf' or 'l2'.")
        self.epsilon = float(epsilon)
        self.norm = norm

    def project(self, delta: np.ndarray) -> np.ndarray:
        """Project a perturbation onto the ``epsilon`` ball in the chosen norm."""
        delta = np.asarray(delta, dtype=np.float64)
        if self.norm == "linf":
            return np.clip(delta, -self.epsilon, self.epsilon)
        norm = float(np.linalg.norm(delta))
        if norm <= self.epsilon or norm == 0.0:
            return delta
        return delta * (self.epsilon / norm)


class MessageAttack(ABC):
    """Crafts perturbed messages for the Byzantine senders in a neighbourhood."""

    name: str = "base"

    @abstractmethod
    def craft(
        self,
        messages: np.ndarray,
        byzantine_mask: np.ndarray,
        aggregator: RobustAggregator,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return a copy of ``messages`` with Byzantine rows perturbed within ``Delta_f``."""
