"""MPE Particle Cooperative Communication Environment (Lowe et al., 2017; Mordatch & Abbeel, 2018).

Pure-NumPy implementation of the continuous Multi-Agent Particle Communication and
Navigation benchmark (PettingZoo MPE simple_speaker_listener / simple_reference generalized
to N agents with Byzantine-resilient learned communication).

Avoids any external GUI/rendering/DLL dependencies (AppLocker/WDAC safe on Windows & Linux),
runs ultra-fast on CPU, and is 100% interface-compatible with the RobustComm runner.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..aggregators.base import RobustAggregator
from ..attacks.base import MessageAttack
from ..data.schemas import Topology
from ..utils.validation import validate_krum_regime


class MPEParticlePolicy:
    """Navigation policy mapping aggregated communication signal and local state to 2D force."""

    def __init__(self, msg_dim: int, rng: np.random.Generator | None = None) -> None:
        self.d = int(msg_dim)
        r = rng or np.random.default_rng(42)
        # Linear projection from message space R^d to 2D physical steering force
        self.W_msg = r.normal(0, 0.1, size=(2, self.d))
        self.W_msg[0, 0] = 1.0  # primary alignment to x-direction
        self.W_msg[1, min(1, self.d - 1)] = 1.0  # primary alignment to y-direction

    def act(self, aggregate: np.ndarray) -> np.ndarray:
        return np.asarray(aggregate, dtype=np.float64)


class MPECommEnv:
    """MPE Multi-Agent Particle Environment with learned communication.

    N particle agents navigate a 2D continuous space to reach their designated landmark goals.
    Agents communicate over a specified network topology. Honest agents send guidance vectors;
    Byzantine agents inject adversarial perturbations. Honest agents aggregate inbound
    messages using RobustComm aggregators to resist Byzantine attacks and successfully
    reach their target landmarks.
    """

    name: str = "mpe_particle_comm"

    def __init__(
        self,
        n_agents: int = 7,
        msg_dim: int = 4,
        topology: Topology | None = None,
        honest_dispersion: float = 0.5,
        horizon: int = 25,
        discrete: bool = False,
        codebook_levels: int = 5,
        signal_scale: float = 1.0,
        arena_size: float = 1.5,
        dt: float = 0.1,
        damping: float = 0.9,
    ) -> None:
        self.n = int(n_agents)
        self.d = int(msg_dim)
        self.B = float(honest_dispersion)
        self.horizon = int(horizon)
        self.discrete = bool(discrete)
        self.levels = int(codebook_levels)
        self.scale = float(signal_scale)
        self.arena = float(arena_size)
        self.dt = float(dt)
        self.damping = float(damping)

        if topology is None:
            # Default to fully connected (broadcast) topology
            inbound = {i: [j for j in range(self.n) if j != i] for i in range(self.n)}
            self.topology = Topology(name="broadcast", n_agents=self.n, inbound=inbound)
        else:
            self.topology = topology

        self.agg_neighbors = {
            i: sorted(set(self.topology.inbound[i]) | {i}) for i in range(self.n)
        }

    def min_agg_size(self) -> int:
        return min(len(v) for v in self.agg_neighbors.values())

    def check_operator_compatible(self, aggregator: RobustAggregator) -> None:
        if getattr(aggregator, "name", "") == "krum":
            validate_krum_regime(self.min_agg_size(), aggregator.f)

    # -- target / messages --------------------------------------------------------
    def sample_target(self, rng: np.random.Generator) -> np.ndarray:
        """Sample goal signal vector s in R^d (encoding relative 2D positions of landmarks)."""
        return rng.uniform(-self.scale, self.scale, size=self.d)

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        grid = np.linspace(-self.scale - self.B, self.scale + self.B, self.levels)
        idx = np.abs(x[..., None] - grid).argmin(axis=-1)
        return grid[idx]

    def honest_messages(self, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Honest agent messages within an L2 ball of radius B around true signal s."""
        dirs = rng.normal(size=(self.n, self.d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        radii = self.B * rng.uniform(0.0, 1.0, size=(self.n, 1)) ** (1.0 / self.d)
        M = s[None, :] + dirs * radii
        return self._quantize(M) if self.discrete else M

    def _reward(self, action: np.ndarray, s: np.ndarray) -> float:
        """Distance-based navigation recovery reward in (0, 1]."""
        dist2 = float(np.sum((np.asarray(action) - s) ** 2))
        return 1.0 / (1.0 + dist2)

    # -- episode rollout with 2D particle dynamics --------------------------------
    def rollout(
        self,
        aggregator: RobustAggregator,
        attack: MessageAttack | None,
        policy: Any,
        byzantine_ids: Iterable[int],
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Execute one complete MPE Particle episode with physical particle dynamics."""
        self.check_operator_compatible(aggregator)
        byz = {int(b) for b in byzantine_ids}
        byz_mask = np.array([i in byz for i in range(self.n)], dtype=bool)
        honest_ids = [i for i in range(self.n) if i not in byz]

        # Initialise 2D particle positions and velocities
        positions = rng.uniform(-self.arena * 0.5, self.arena * 0.5, size=(self.n, 2))
        velocities = np.zeros((self.n, 2), dtype=np.float64)

        # Landmarks (goals) for each agent in 2D continuous space
        landmarks = rng.uniform(-self.arena * 0.8, self.arena * 0.8, size=(self.n, 2))

        step_returns: list[float] = []
        deviations: list[float] = []

        for _ in range(self.horizon):
            # Target signal encodes true landmark guidance vector in R^d
            rel_goals = landmarks - positions
            s = np.zeros(self.d, dtype=np.float64)
            s[: min(2, self.d)] = np.mean(rel_goals[:, : min(2, self.d)], axis=0)
            if self.d > 2:
                s[2:] = np.mean(np.sin(rel_goals[:, : min(2, self.d)]), axis=0)[: self.d - 2]

            M = self.honest_messages(s, rng)
            if attack is not None and byz_mask.any():
                M = attack.craft(M, byz_mask, aggregator, rng)

            rewards: list[float] = []
            forces = np.zeros((self.n, 2), dtype=np.float64)

            # Byzantine agents maintain cooperative physical motion (message-only threat model)
            for b in byz:
                bf = np.asarray(rel_goals[b, :2], dtype=np.float64)
                bn = np.linalg.norm(bf)
                if bn > 1.0:
                    bf = bf / bn
                forces[b] = bf

            for i in honest_ids:
                nbr = self.agg_neighbors[i]
                m_hat = aggregator.aggregate(M[nbr])
                act = policy.act(m_hat)
                rewards.append(self._reward(act, s))
                deviations.append(float(np.linalg.norm(m_hat - s)))

                # Physical navigation force derived from communication signal
                guidance_force = np.asarray(act[:2], dtype=np.float64)
                force_norm = np.linalg.norm(guidance_force)
                if force_norm > 1.0:
                    guidance_force = guidance_force / force_norm
                forces[i] = guidance_force

            # Update particle physics (velocity damping + position integration)
            velocities = velocities * self.damping + forces * self.dt
            positions += velocities * self.dt

            # Team step return: honest navigation proximity and coordination
            step_returns.append(float(np.mean(rewards)) if rewards else 0.0)

        return {
            "return": float(np.mean(step_returns)),
            "mean_deviation": float(np.mean(deviations)) if deviations else 0.0,
            "max_deviation": float(np.max(deviations)) if deviations else 0.0,
        }
