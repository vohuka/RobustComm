"""Backward Pass Differentiable Approximation (BPDA) Attack (Athalye et al., ICML 2018).

Bypasses gradient masking / obfuscated gradients in non-smooth robust aggregators
(such as Coordinate-wise Median and Krum) by evaluating the true operator on the
forward pass while computing gradients via a smooth soft-median relaxation on the
backward pass.
"""
from __future__ import annotations

import numpy as np

from ..aggregators.base import RobustAggregator
from ..policy.surrogate_policy import SurrogateObjective
from .base import MessageAttack, PerturbationSet


class BPDAMessageAttack(MessageAttack):
    name = "bpda"

    def __init__(
        self,
        epsilon: float,
        steps: int = 10,
        step_size: float = 0.2,
        norm: str = "linf",
        tau: float = 0.2,
        colluding: bool = False,
    ) -> None:
        self.pset = PerturbationSet(epsilon, norm)
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.tau = float(tau)
        self.colluding = bool(colluding)

    def _soft_weights(self, M: np.ndarray, agg: np.ndarray) -> np.ndarray:
        """Compute soft weights w_{j, k} proportional to exp(-|M_{j,k} - agg_k| / tau)."""
        diff = np.abs(M - agg[None, :])
        # Numerically stable softmax weights per coordinate
        scaled = -diff / max(1e-4, self.tau)
        scaled -= np.max(scaled, axis=0, keepdims=True)
        exp_w = np.exp(scaled)
        weights = exp_w / (np.sum(exp_w, axis=0, keepdims=True) + 1e-12)
        return weights  # shape (n, d)

    def _bpda_gradients(
        self,
        M: np.ndarray,
        byz_idx: np.ndarray,
        aggregator: RobustAggregator,
        s_hat: np.ndarray,
    ) -> np.ndarray:
        # Forward pass through true aggregator
        agg = aggregator.aggregate(M)
        # Gradient of proxy loss w.r.t. aggregate vector
        g_agg = SurrogateObjective.proxy_grad(agg, s_hat)  # shape (d,)

        # Backward pass through soft relaxation: d(agg_k) / d(M_{j,k}) ~ soft_weight_{j,k}
        weights = self._soft_weights(M, agg)  # shape (n, d)
        byz_weights = weights[byz_idx]  # shape (f, d)

        # Chain rule: grad_{j, k} = g_agg[k] * byz_weights[j, k]
        grads = byz_weights * g_agg[None, :]
        return grads

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

        pert = np.zeros((len(byz_idx), M.shape[1]))

        # Colluding target direction
        if self.colluding:
            # Pick a coordinated malicious direction
            target_dir = rng.normal(size=(1, M.shape[1]))
            target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-12)

        for _ in range(self.steps):
            M_curr = M.copy()
            M_curr[byz_idx] = M[byz_idx] + pert
            grads = self._bpda_gradients(M_curr, byz_idx, aggregator, s_hat)

            if self.colluding:
                # Coordinate into a tight adversarial cluster toward worst-case direction
                shared_step = np.sign(grads.mean(axis=0, keepdims=True))
                step = self.step_size * np.tile(shared_step, (len(byz_idx), 1))
                # Add tiny jitter to prevent exact collinear collapse if needed
                pert = pert + step
            else:
                step = self.step_size * np.sign(grads)
                pert = pert + step

            pert = np.array([self.pset.project(p) for p in pert])

        M[byz_idx] = M[byz_idx] + pert
        return M
