"""Coordinate-wise beta-trimmed mean (Def 3.2, Alg 1).

Drops the largest and smallest ``ceil(beta*n_i)`` values per coordinate and averages the
rest. The deviation guarantee (Lem 9.3) requires ``beta >= f/n_i``; the code additionally
requires a non-empty remainder (``n_i - 2*ceil(beta*n_i) >= 1``) and raises otherwise.
At ``beta=0`` it reduces to the (non-robust) mean.
"""
from __future__ import annotations

import math

import numpy as np

from ..utils.validation import validate_messages
from .base import RobustAggregator


class TrimmedMeanAggregator(RobustAggregator):
    name = "trmean"

    def __init__(self, beta: float, f: int = 0) -> None:
        if not (0.0 <= beta < 0.5):
            raise ValueError("trimmed-mean beta must be in [0, 0.5).")
        self.beta = float(beta)
        self.f = int(f)

    def guarantee_holds(self, n: int) -> bool:
        return self.beta >= (self.f / n if n else 1.0) and (n - 2 * self._k(n)) >= 1

    def _k(self, n: int) -> int:
        return math.ceil(self.beta * n)

    def aggregate(self, messages: np.ndarray) -> np.ndarray:
        M = validate_messages(messages)
        n = M.shape[0]
        k = self._k(n)
        if n - 2 * k < 1:
            raise ValueError(
                f"trimmed mean removes too much: n={n}, k={k} per side leaves "
                f"{n - 2 * k} values. Reduce beta."
            )
        S = np.sort(M, axis=0)
        kept = S[k : n - k, :] if k > 0 else S
        return kept.mean(axis=0)
