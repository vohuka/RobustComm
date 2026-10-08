"""Boundary-Pushing Collusion Attack against Coordinate-wise Aggregators (Reviewer 2 stress test).

Unlike clustered collusion attacks that group all Byzantine perturbations toward a single
shared vector (designed to deceive distance-based metrics like Krum), the Boundary-Pushing
Attack specifically targets Coordinate-wise Median and Trimmed Mean.

An adversary aware of coordinate-wise sorting optimizes perturbations dimension-by-dimension
to bias the median coordinate-by-coordinate toward the outermost extreme boundary of the honest
bounding hypercube on *every* dimension simultaneously.
"""
from __future__ import annotations

import numpy as np

from ..aggregators.base import RobustAggregator
from ..policy.surrogate_policy import SurrogateObjective
from .base import MessageAttack, PerturbationSet


class BoundaryPushingAttack(MessageAttack):
    """Colluding boundary-pushing attack explicitly stressing coordinate-wise median."""

    name = "boundary_pushing"

    def __init__(
        self,
        epsilon: float = 1.0,
        norm: str = "linf",
        steps: int = 10,
        step_size: float = 0.2,
        target_bias: str = "worst_case",  # 'worst_case', 'positive', 'negative'
    ) -> None:
        self.pset = PerturbationSet(epsilon, norm)
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.target_bias = target_bias

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

        # Determine honest hypercube boundaries per dimension: [min_k, max_k]
        h_min = np.min(honest, axis=0)  # shape (d,)
        h_max = np.max(honest, axis=0)  # shape (d,)
        h_mean = np.mean(honest, axis=0)  # shape (d,)

        # Surrogate guidance direction that maximizes task disruption
        s_hat = SurrogateObjective.estimate_target(honest)
        # Gradient direction of task loss w.r.t. message
        grad_dir = SurrogateObjective.proxy_grad(h_mean, s_hat)

        # Coordinate-by-coordinate bias direction:
        # If grad_dir[k] > 0, pushing positive increases loss; else pushing negative increases loss
        if self.target_bias == "worst_case":
            push_sign = np.sign(grad_dir)
            push_sign[push_sign == 0] = 1.0
        elif self.target_bias == "positive":
            push_sign = np.ones(M.shape[1])
        else:
            push_sign = -np.ones(M.shape[1])

        # For each coordinate k, target extreme boundary past the honest edge
        # Target = boundary + push_sign * epsilon
        extreme_targets = np.where(push_sign > 0, h_max + self.pset.epsilon, h_min - self.pset.epsilon)

        # Byzantine senders coordinate to transmit messages at or beyond the bounding edge
        pert = np.zeros((len(byz_idx), M.shape[1]))
        for idx_pos, b in enumerate(byz_idx):
            desired_perturbation = extreme_targets - M[b]
            # Project onto allowed epsilon ball
            pert[idx_pos] = self.pset.project(desired_perturbation)

        M[byz_idx] = M[byz_idx] + pert
        return M
