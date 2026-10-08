"""Trainable linear policy and MADDPG-style ensemble (Sec. 7.3).

Policies map a received aggregate ``m_hat`` to an action ``a = W m_hat``; on the synthetic
task the optimal ``W`` is the identity (recover the signal). Adversarial training adjusts
``W`` under attack (``robustcomm.training.adversarial``). The ensemble holds ``K`` members;
one is sampled per training episode (MADDPG-style), and deployment averages them.
"""
from __future__ import annotations

import numpy as np


class LinearPolicy:
    """Action ``a = W m_hat``; ``W`` initialised near the identity."""

    def __init__(self, msg_dim: int, rng: np.random.Generator | None = None) -> None:
        rng = rng if rng is not None else np.random.default_rng(0)
        self.W = np.eye(msg_dim) + rng.normal(0.0, 0.01, size=(msg_dim, msg_dim))

    def act(self, aggregate: np.ndarray) -> np.ndarray:
        return self.W @ np.asarray(aggregate, dtype=np.float64)

    def mse_grad(self, aggregate: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Gradient of ``||W m - s||^2`` w.r.t. ``W`` = ``2 (W m - s) m^T``."""
        m = np.asarray(aggregate, dtype=np.float64)
        a = self.W @ m
        return 2.0 * np.outer(a - np.asarray(target, dtype=np.float64), m)

    def apply_grad(self, grad_W: np.ndarray, lr: float) -> None:
        self.W = self.W - lr * grad_W


class PolicyEnsemble:
    """``K`` linear policies; sample one per episode, average for deployment."""

    def __init__(self, msg_dim: int, ensemble_size: int = 3, rng: np.random.Generator | None = None) -> None:
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1.")
        rng = rng if rng is not None else np.random.default_rng(0)
        self.members = [LinearPolicy(msg_dim, rng) for _ in range(ensemble_size)]

    def sample_member(self, rng: np.random.Generator) -> LinearPolicy:
        return self.members[int(rng.integers(len(self.members)))]

    def act(self, aggregate: np.ndarray) -> np.ndarray:
        return np.mean([m.act(aggregate) for m in self.members], axis=0)
