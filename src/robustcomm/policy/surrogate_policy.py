"""Surrogate model for the black-box attacker (Sec. 5, 7).

The paper's attacker is black-box to its victims and crafts perturbations on a
*surrogate* to exploit transferability. Here the surrogate is a simple, differentiable
estimate of the team objective: it estimates the target signal ``s_hat`` that honest
messages cluster around, and scores an aggregate by its distance from ``s_hat`` (a proxy
for negative team reward). Moving the aggregate away from ``s_hat`` lowers the proxy
reward, which is what the attacker optimises.

``LinearSurrogatePolicy`` is provided as the extensible general interface (a learned
linear map with an analytic gradient); the synthetic attack uses ``SurrogateObjective``.
"""
from __future__ import annotations

import numpy as np


class SurrogateObjective:
    """Attacker's black-box proxy for the team objective.

    ``estimate_target`` uses the robust centre (coordinate-wise median) of the observed
    messages as ``s_hat`` — a transfer-style estimate that does not require the victim's
    parameters. ``proxy_loss`` is the attacker's objective to *increase* (distance of the
    aggregate from ``s_hat``); its gradient w.r.t. an aggregate is a unit vector away from
    ``s_hat``.
    """

    @staticmethod
    def estimate_target(messages: np.ndarray) -> np.ndarray:
        return np.median(np.asarray(messages, dtype=np.float64), axis=0)

    @staticmethod
    def proxy_loss(aggregate: np.ndarray, s_hat: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(aggregate) - np.asarray(s_hat)))

    @staticmethod
    def proxy_grad(aggregate: np.ndarray, s_hat: np.ndarray) -> np.ndarray:
        diff = np.asarray(aggregate, dtype=np.float64) - np.asarray(s_hat, dtype=np.float64)
        norm = float(np.linalg.norm(diff))
        if norm == 0.0:
            # degenerate: pick an arbitrary unit direction (deterministic)
            g = np.zeros_like(diff)
            g[0] = 1.0
            return g
        return diff / norm


class LinearSurrogatePolicy:
    """A learned linear surrogate ``a = W m + b`` with an analytic gradient.

    Extensible interface for a general differentiable surrogate; ``value`` returns a
    scalar proxy and ``grad_wrt_message`` its analytic gradient (= ``W^T`` applied to the
    upstream gradient). Not required by the default synthetic attack but available for
    experiments that train a surrogate policy.
    """

    def __init__(self, msg_dim: int, act_dim: int | None = None, rng: np.random.Generator | None = None):
        act_dim = act_dim or msg_dim
        rng = rng if rng is not None else np.random.default_rng(0)
        self.W = rng.normal(0, 0.1, size=(act_dim, msg_dim))
        self.b = np.zeros(act_dim)

    def act(self, message: np.ndarray) -> np.ndarray:
        return self.W @ np.asarray(message, dtype=np.float64) + self.b

    def grad_wrt_message(self, upstream_grad: np.ndarray) -> np.ndarray:
        """Backprop an upstream gradient (d_loss/d_action) to the message input."""
        return self.W.T @ np.asarray(upstream_grad, dtype=np.float64)
