"""Surrogate-based inter-agent message attack (Alg 2 / Def 5.1).

Implements the FGSM-style projected attack the paper describes: a black-box attacker
crafts perturbations for an ``f``-fraction of senders on a *surrogate* estimate of the
team objective, projected into the admissible set ``Delta_f`` (Eq. pertset).

Modes (matching the paper):
* ``static``  — perturbation crafted against the mean aggregator (the undefended target).
* ``adaptive`` — perturbation crafted against the *deployed* ``RobAgg`` so it survives
  aggregation; gradients through the non-differentiable robust operator are taken by
  finite differences.
* ``colluding`` — Byzantine senders coordinate onto a shared perturbation direction.

The FGSM step is ``delta <- Proj_{Delta_f}( delta + alpha * sign(grad) )`` where ``grad``
is the gradient of the attacker's proxy loss (distance of the aggregate from the surrogate
target ``s_hat``) with respect to a Byzantine message (Goodfellow et al., FGSM).
"""
from __future__ import annotations

import numpy as np

from ..aggregators.base import RobustAggregator
from ..policy.surrogate_policy import SurrogateObjective
from .base import MessageAttack, PerturbationSet


class FGSMMessageAttack(MessageAttack):
    name = "fgsm"

    def __init__(
        self,
        epsilon: float,
        steps: int = 5,
        step_size: float = 0.5,
        norm: str = "linf",
        mode: str = "adaptive",
        colluding: bool = False,
        fd_eps: float = 1e-3,
    ) -> None:
        if mode not in ("static", "adaptive"):
            raise ValueError("mode must be 'static' or 'adaptive'.")
        self.pset = PerturbationSet(epsilon, norm)
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.mode = mode
        self.colluding = bool(colluding)
        self.fd_eps = float(fd_eps)

    # -- gradient of the proxy loss w.r.t. each Byzantine message -----------------
    def _gradients(
        self,
        messages: np.ndarray,
        byz_idx: np.ndarray,
        pert: np.ndarray,
        aggregator: RobustAggregator,
        s_hat: np.ndarray,
    ) -> np.ndarray:
        M = messages.copy()
        M[byz_idx] = messages[byz_idx] + pert
        if self.mode == "static":
            # Analytic: mean aggregate has d(agg)/d(m_b) = 1/n * I; chain with proxy grad.
            agg = M.mean(axis=0)
            g_agg = SurrogateObjective.proxy_grad(agg, s_hat) / M.shape[0]
            return np.tile(g_agg, (len(byz_idx), 1))
        # Adaptive: finite differences through the deployed robust aggregator.
        base = SurrogateObjective.proxy_loss(aggregator.aggregate(M), s_hat)
        grads = np.zeros((len(byz_idx), M.shape[1]))
        for a, b in enumerate(byz_idx):
            for k in range(M.shape[1]):
                Mp = M.copy()
                Mp[b, k] += self.fd_eps
                lp = SurrogateObjective.proxy_loss(aggregator.aggregate(Mp), s_hat)
                grads[a, k] = (lp - base) / self.fd_eps
        return grads

    def craft(
        self,
        messages: np.ndarray,
        byzantine_mask: np.ndarray,
        aggregator: RobustAggregator,
        rng: np.random.Generator,  # noqa: ARG002 (kept for interface symmetry / stochastic variants)
    ) -> np.ndarray:
        M = np.array(messages, dtype=np.float64, copy=True)
        mask = np.asarray(byzantine_mask, dtype=bool)
        byz_idx = np.where(mask)[0]
        if len(byz_idx) == 0 or self.pset.epsilon == 0.0:
            return M

        honest = M[~mask]
        s_hat = SurrogateObjective.estimate_target(honest if len(honest) else M)

        pert = np.zeros((len(byz_idx), M.shape[1]))
        for _ in range(self.steps):
            grads = self._gradients(M, byz_idx, pert, aggregator, s_hat)
            if self.colluding:
                # coordinate onto a shared direction (sign of the mean gradient)
                shared = np.sign(grads.mean(axis=0))
                step = self.step_size * np.tile(shared, (len(byz_idx), 1))
            else:
                step = self.step_size * np.sign(grads)
            pert = pert + step
            pert = np.array([self.pset.project(p) for p in pert])

        M[byz_idx] = messages[byz_idx] + pert
        return M
