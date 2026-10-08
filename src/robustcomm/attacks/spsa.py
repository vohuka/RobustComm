"""Simultaneous Perturbation Stochastic Approximation (SPSA) Zeroth-Order Attack.

Gradient-free black-box adversarial attack that optimizes perturbations directly
from zero-order function evaluations. Completely immune to gradient masking or
non-differentiable defense surfaces.
"""
from __future__ import annotations

import numpy as np

from ..aggregators.base import RobustAggregator
from ..policy.surrogate_policy import SurrogateObjective
from .base import MessageAttack, PerturbationSet


class SPSAMessageAttack(MessageAttack):
    name = "spsa"

    def __init__(
        self,
        epsilon: float,
        steps: int = 15,
        step_size: float = 0.1,
        norm: str = "linf",
        delta_eps: float = 1e-2,
        restarts: int = 2,
        colluding: bool = False,
    ) -> None:
        self.pset = PerturbationSet(epsilon, norm)
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.delta_eps = float(delta_eps)
        self.restarts = int(restarts)
        self.colluding = bool(colluding)

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
        s_hat = SurrogateObjective.estimate_target(honest if len(honest) else M)

        best_pert = np.zeros((len(byz_idx), M.shape[1]))
        best_loss = -np.inf

        for _ in range(self.restarts):
            # Initialize perturbation randomly within bounds
            if self.pset.norm == "linf":
                pert = rng.uniform(-self.pset.epsilon * 0.5, self.pset.epsilon * 0.5, size=(len(byz_idx), M.shape[1]))
            else:
                pert = rng.normal(0, self.pset.epsilon * 0.5, size=(len(byz_idx), M.shape[1]))
                pert = np.array([self.pset.project(p) for p in pert])

            for _ in range(self.steps):
                # Sample random Rademacher perturbation vector (+1 or -1)
                Delta = rng.choice([-1.0, 1.0], size=pert.shape)

                # Positive and negative query points
                pert_plus = np.array([self.pset.project(p) for p in pert + self.delta_eps * Delta])
                pert_minus = np.array([self.pset.project(p) for p in pert - self.delta_eps * Delta])

                M_plus = M.copy()
                M_plus[byz_idx] += pert_plus
                loss_plus = SurrogateObjective.proxy_loss(aggregator.aggregate(M_plus), s_hat)

                M_minus = M.copy()
                M_minus[byz_idx] += pert_minus
                loss_minus = SurrogateObjective.proxy_loss(aggregator.aggregate(M_minus), s_hat)

                # SPSA gradient estimate: g_k = (loss+ - loss-) / (2 * delta * Delta_k)
                g_hat = (loss_plus - loss_minus) / (2.0 * self.delta_eps * Delta)

                if self.colluding:
                    shared_g = g_hat.mean(axis=0, keepdims=True)
                    pert = pert + self.step_size * np.tile(np.sign(shared_g), (len(byz_idx), 1))
                else:
                    pert = pert + self.step_size * np.sign(g_hat)

                pert = np.array([self.pset.project(p) for p in pert])

            # Check final loss of this restart
            M_cand = M.copy()
            M_cand[byz_idx] += pert
            cand_loss = SurrogateObjective.proxy_loss(aggregator.aggregate(M_cand), s_hat)
            if cand_loss > best_loss:
                best_loss = cand_loss
                best_pert = pert.copy()

        M[byz_idx] += best_pert
        return M
