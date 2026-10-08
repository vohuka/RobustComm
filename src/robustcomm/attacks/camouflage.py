"""Statistics-Mimicking / Camouflage Byzantine Attack (Reviewer 4 stress test).

Evaluates the threat where Byzantine senders craft adversarial messages that mimic the
empirical statistics (mean, covariance, dispersion radius B_i) of honest messages, staying
strictly inside the honest dispersion ball B_i while injecting subtle, coordinated misleading
guidance (e.g. misleading landmark vectors or inverted guidance).

Bypasses anomaly-detection, reconstruction autoencoders (R-MACRL), and outlier filters,
serving as the definitive test of Theorem 5.4's dispersion-bounded guarantee.
"""
from __future__ import annotations

import numpy as np

from ..aggregators.base import RobustAggregator
from ..policy.surrogate_policy import SurrogateObjective
from .base import MessageAttack, PerturbationSet


class CamouflageAttack(MessageAttack):
    """Statistics-mimicking camouflage attack bounded strictly within honest dispersion."""

    name = "camouflage"

    def __init__(
        self,
        epsilon: float = 1.0,
        norm: str = "l2",
        dispersion_bound: float = 0.45,
        invert_target: bool = True,
    ) -> None:
        self.pset = PerturbationSet(epsilon, norm)
        self.B = float(dispersion_bound)
        self.invert = bool(invert_target)

    def craft(
        self,
        messages: np.ndarray,
        byzantine_mask: np.ndarray,
        aggregator: RobustAggregator,
        rng: np.random.Generator,
    ) -> np.ndarray:
        M = np.array(messages, dtype=np.float64, copy=True)
        mask = np.asarray(byzantine_mask, dtype=bool)
        byz_idx = np.where(mask)[0]
        if len(byz_idx) == 0 or self.pset.epsilon == 0.0:
            return M

        honest = M[~mask]
        if len(honest) == 0:
            return M

        # Compute empirical statistics of honest messages
        h_mean = np.mean(honest, axis=0)  # (d,)
        h_std = np.std(honest, axis=0) + 1e-6

        # Target direction: inverted or orthogonal to honest consensus
        if self.invert:
            # Adverse direction: inverted from true honest guidance
            s_hat = SurrogateObjective.estimate_target(honest)
            target_direction = -(s_hat - h_mean)
            norm_td = np.linalg.norm(target_direction)
            if norm_td > 1e-6:
                target_direction = target_direction / norm_td
            else:
                target_direction = -rng.normal(size=M.shape[1])
                target_direction /= np.linalg.norm(target_direction) + 1e-6
        else:
            # Orthogonal misleading direction
            target_direction = rng.normal(size=M.shape[1])
            target_direction /= np.linalg.norm(target_direction) + 1e-6

        # Synthesize camouflage messages that match covariance but point along adversarial direction
        # strictly bounded within radius B around h_mean
        eff_radius = min(self.B, self.pset.epsilon)

        for idx_pos, b in enumerate(byz_idx):
            # Subtle perturbation along target direction plus small honest-like jitter
            jitter = rng.normal(0, 0.1 * h_std, size=M.shape[1])
            crafted = h_mean + eff_radius * 0.9 * target_direction + jitter
            # Enforce strict dispersion ball membership: ||crafted - h_mean||_2 <= B
            dist_to_mean = np.linalg.norm(crafted - h_mean)
            if dist_to_mean > self.B:
                crafted = h_mean + (crafted - h_mean) * (self.B / (dist_to_mean + 1e-12))

            M[b] = crafted

        return M
