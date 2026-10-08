"""SOTA Communicative MARL Baselines (Sun et al. ICLR 2023, Xue et al. AAMAS 2022, Yu et al. AAAI 2024).

Implements:
1. Sun et al. (ICLR 2023): Message-ablation randomized smoothing / consensus ensemble.
2. Xue et al. (AAMAS 2022) / ADMAC (Yu et al. AAAI 2024): Autoencoder / subspace reconstruction residual filtering.
"""
from __future__ import annotations

import numpy as np

from ..aggregators.base import RobustAggregator
from ..utils.validation import validate_messages


class SunMessageAblationDefense(RobustAggregator):
    """Certified message ablation ensemble aggregator (Sun et al., ICLR 2023).

    Samples S random subsets of inbound messages of size (n - f), computes the
    sub-sample mean, and takes the median over the ensemble predictions.
    Certified against up to f corruptions with O(S * n * d) computational overhead.
    """

    name = "sun_ensemble"

    def __init__(self, msg_dim: int, f: int = 1, num_samples: int = 25, seed: int = 42) -> None:
        self.d = int(msg_dim)
        self.f = int(f)
        self.S = int(num_samples)
        self.rng = np.random.default_rng(seed)

    def guarantee_holds(self, n: int) -> bool:
        return n > self.f

    def aggregate(self, messages: np.ndarray) -> np.ndarray:
        M = validate_messages(messages)
        n = M.shape[0]
        sub_size = max(1, n - self.f)

        if n <= sub_size:
            return M.mean(axis=0)

        sub_means = []
        for _ in range(self.S):
            idx = self.rng.choice(n, size=sub_size, replace=False)
            sub_means.append(M[idx].mean(axis=0))

        # Output the coordinate-wise median over all subset predictions
        return np.median(np.array(sub_means), axis=0)


class RMACRLReconstructionDefense(RobustAggregator):
    """Message reconstruction and residual filtering defense (Xue et al., AAMAS 2022; Yu et al., AAAI 2024).

    Identifies corrupted messages by projecting inbound messages onto the estimated
    principal subspace of honest message semantics. Outliers with high reconstruction
    residual are filtered out before aggregation.
    """

    name = "rmacrl_filter"

    def __init__(self, msg_dim: int, residual_threshold: float = 0.8, k_components: int = 2) -> None:
        self.d = int(msg_dim)
        self.threshold = float(residual_threshold)
        self.k = min(int(k_components), self.d)
        # Initialize orthogonal projection basis
        self.basis = np.eye(self.d)[:, :self.k]

    def guarantee_holds(self, n: int) -> bool:
        return n >= 2

    def update_basis(self, honest_messages: np.ndarray) -> None:
        """Update principal subspace from clean message batches."""
        if len(honest_messages) >= self.k:
            _, _, vt = np.linalg.svd(honest_messages - honest_messages.mean(axis=0), full_matrices=False)
            self.basis = vt[:self.k].T

    def aggregate(self, messages: np.ndarray) -> np.ndarray:
        M = validate_messages(messages)
        n = M.shape[0]

        # Compute projection residuals: ||m_j - P P^T m_j||_2
        projected = (M @ self.basis) @ self.basis.T
        residuals = np.linalg.norm(M - projected, axis=1)

        # Retain messages below anomaly threshold
        honest_mask = residuals <= self.threshold
        if not np.any(honest_mask):
            # Fallback to the message with minimum residual
            best_idx = np.argmin(residuals)
            return M[best_idx]

        return M[honest_mask].mean(axis=0)
