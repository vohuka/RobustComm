"""Synthetic cooperative communication environment (smoke tests / demonstration).

This is a light, fully-specified stand-in for the paper's DIAL/MPE learned-communication
tasks, used ONLY to exercise the pipeline end-to-end on controlled data. It is NOT a paper
benchmark; any output produced from it is a "synthetic smoke-test output, not a paper
result." Real environments are provided as adapters in ``robustcomm.data.adapters``.

Task. Each step a target signal ``s in R^d`` is drawn. Honest sender ``j`` emits a message
within an ``l2`` ball of radius ``B`` (=honest_dispersion, Assumption A1) around ``s``.
Byzantine senders emit adversarial messages. Each honest agent ``i`` aggregates its inbound
neighbourhood (including its own honest message) into ``m_hat_i`` and acts; the team is
rewarded for recovering ``s``. Robust aggregation keeps ``m_hat_i`` near ``s`` (reward
holds); the mean is pulled away by Byzantine senders (reward drops) — the central claim of
the paper, demonstrated on synthetic data.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..aggregators.base import RobustAggregator
from ..attacks.base import MessageAttack
from ..data.schemas import Topology
from ..utils.validation import validate_krum_regime


class IdentityPolicy:
    """Default policy: the action is the received aggregate (optimal when m_hat == s)."""

    def act(self, aggregate: np.ndarray) -> np.ndarray:
        return np.asarray(aggregate, dtype=np.float64)


class SyntheticCommEnv:
    def __init__(
        self,
        n_agents: int,
        msg_dim: int,
        topology: Topology,
        honest_dispersion: float = 0.5,
        horizon: int = 20,
        discrete: bool = False,
        codebook_levels: int = 5,
        signal_scale: float = 1.0,
    ) -> None:
        self.n = int(n_agents)
        self.d = int(msg_dim)
        self.topology = topology
        self.B = float(honest_dispersion)
        self.horizon = int(horizon)
        self.discrete = bool(discrete)
        self.levels = int(codebook_levels)
        self.scale = float(signal_scale)
        # Aggregation neighbourhood includes the agent's own (honest) message.
        self.agg_neighbors = {
            i: sorted(set(topology.inbound[i]) | {i}) for i in range(self.n)
        }

    def min_agg_size(self) -> int:
        return min(len(v) for v in self.agg_neighbors.values())

    def check_operator_compatible(self, aggregator: RobustAggregator) -> None:
        """Raise a clear error if Krum's regime fails for the smallest neighbourhood."""
        if getattr(aggregator, "name", "") == "krum":
            validate_krum_regime(self.min_agg_size(), aggregator.f)

    # -- signal / messages --------------------------------------------------------
    def sample_target(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-self.scale, self.scale, size=self.d)

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        grid = np.linspace(-self.scale - self.B, self.scale + self.B, self.levels)
        idx = np.abs(x[..., None] - grid).argmin(axis=-1)
        return grid[idx]

    def honest_messages(self, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Honest messages uniform in the ``l2`` ball of radius ``B`` around ``s`` (A1)."""
        dirs = rng.normal(size=(self.n, self.d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        radii = self.B * rng.uniform(0.0, 1.0, size=(self.n, 1)) ** (1.0 / self.d)
        M = s[None, :] + dirs * radii
        return self._quantize(M) if self.discrete else M

    def _reward(self, action: np.ndarray, s: np.ndarray) -> float:
        # bounded in (0, 1]; 1 when the action recovers the signal exactly.
        dist2 = float(np.sum((np.asarray(action) - s) ** 2))
        return 1.0 / (1.0 + dist2)

    # -- rollout ------------------------------------------------------------------
    def rollout(
        self,
        aggregator: RobustAggregator,
        attack: MessageAttack | None,
        policy: Any,
        byzantine_ids: Iterable[int],
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Run one episode; return team return and aggregate-deviation diagnostics."""
        self.check_operator_compatible(aggregator)
        byz = {int(b) for b in byzantine_ids}
        byz_mask = np.array([i in byz for i in range(self.n)], dtype=bool)
        honest_ids = [i for i in range(self.n) if i not in byz]

        step_returns: list[float] = []
        deviations: list[float] = []
        for _ in range(self.horizon):
            s = self.sample_target(rng)
            M = self.honest_messages(s, rng)
            if attack is not None and byz_mask.any():
                M = attack.craft(M, byz_mask, aggregator, rng)
            rewards = []
            for i in honest_ids:
                nbr = self.agg_neighbors[i]
                m_hat = aggregator.aggregate(M[nbr])
                rewards.append(self._reward(policy.act(m_hat), s))
                deviations.append(float(np.linalg.norm(m_hat - s)))
            step_returns.append(float(np.mean(rewards)) if rewards else 0.0)

        return {
            "return": float(np.mean(step_returns)),
            "mean_deviation": float(np.mean(deviations)) if deviations else 0.0,
            "max_deviation": float(np.max(deviations)) if deviations else 0.0,
        }
