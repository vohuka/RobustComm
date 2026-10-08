"""Predator-Prey Continuous Multi-Agent Pursuit-Evasion Environment.

Cooperative pursuit-evasion benchmark where N communicating predator agents cooperate
to track, encircle, and capture M agile evading prey agents in a continuous 2D plane.
Predators transmit continuous coordination messages m_j in R^d (encoding observed relative
prey kinematics and encirclement headings). Byzantine predators inject adversarial
perturbations into the communication channel. Honest predators apply RobustComm aggregators
to maintain coherent coordination, encircle the prey, and secure high capture returns.

Pure-NumPy implementation safe under Windows AppLocker / WDAC, interface-compatible
with the RobustComm benchmark runner.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ..aggregators.base import RobustAggregator
from ..attacks.base import MessageAttack
from ..data.schemas import Topology
from ..utils.validation import validate_krum_regime


class PredatorPreyPolicy:
    """Predator pursuit policy mapping aggregated guidance signal to 2D pursuit propulsion."""

    def __init__(self, msg_dim: int, rng: np.random.Generator | None = None) -> None:
        self.d = int(msg_dim)
        r = rng or np.random.default_rng(42)
        # Projection from message space R^d to 2D physical steering force
        self.W = r.normal(0, 0.1, size=(2, self.d))
        self.W[0, 0] = 1.0  # x-pursuit
        self.W[1, min(1, self.d - 1)] = 1.0  # y-pursuit

    def act(self, aggregate: np.ndarray) -> np.ndarray:
        return np.asarray(aggregate, dtype=np.float64)


class PredatorPreyEnv:
    """Cooperative Predator-Prey Pursuit Environment with learned communication.

    N predators navigate 2D continuous space to capture M faster preys.
    Predators communicate over a designated network topology.
    Capture requires multi-agent encirclement: a single predator cannot catch prey alone;
    at least 2 predators must close in simultaneously within capture radius.
    """

    name: str = "predator_prey"

    def __init__(
        self,
        n_agents: int = 6,
        n_preys: int = 2,
        msg_dim: int = 4,
        topology: Topology | None = None,
        honest_dispersion: float = 0.45,
        horizon: int = 30,
        discrete: bool = False,
        codebook_levels: int = 5,
        arena_size: float = 2.0,
        dt: float = 0.1,
        damping: float = 0.85,
        capture_radius: float = 0.35,
    ) -> None:
        self.n = int(n_agents)
        self.n_preys = int(n_preys)
        self.d = int(msg_dim)
        self.B = float(honest_dispersion)
        self.horizon = int(horizon)
        self.discrete = bool(discrete)
        self.levels = int(codebook_levels)
        self.arena = float(arena_size)
        self.dt = float(dt)
        self.damping = float(damping)
        self.capture_radius = float(capture_radius)

        if topology is None:
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

    def honest_messages(self, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Emits honest predator guidance messages concentrated in an L2 ball around target signal s."""
        dirs = rng.normal(size=(self.n, self.d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        radii = self.B * (rng.uniform(0.0, 1.0, size=(self.n, 1)) ** (1.0 / self.d))
        M = s[None, :] + dirs * radii
        if self.discrete:
            grid = np.linspace(-self.arena - self.B, self.arena + self.B, self.levels)
            idx = np.abs(M[..., None] - grid).argmin(axis=-1)
            return grid[idx]
        return M

    def rollout(
        self,
        aggregator: RobustAggregator,
        attack: MessageAttack | None,
        policy: Any,
        byzantine_ids: Iterable[int],
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Execute one complete pursuit-evasion episode."""
        self.check_operator_compatible(aggregator)
        byz = {int(b) for b in byzantine_ids}
        byz_mask = np.array([i in byz for i in range(self.n)], dtype=bool)
        honest_ids = [i for i in range(self.n) if i not in byz]

        # Initialize predator positions (distributed in arena)
        pred_pos = rng.uniform(-self.arena * 0.4, self.arena * 0.4, size=(self.n, 2))
        pred_vel = np.zeros((self.n, 2), dtype=np.float64)

        # Initialize prey positions (evaders)
        prey_pos = rng.uniform(-self.arena * 0.7, self.arena * 0.7, size=(self.n_preys, 2))
        prey_vel = np.zeros((self.n_preys, 2), dtype=np.float64)

        step_returns: list[float] = []
        deviations: list[float] = []
        captured_counts = 0

        for _ in range(self.horizon):
            # Target signal s in R^d encodes relative prey centroid and motion
            active_prey = prey_pos[0]  # primary target prey
            rel_vecs = active_prey[None, :] - pred_pos  # (n, 2)
            
            s = np.zeros(self.d, dtype=np.float64)
            s[: min(2, self.d)] = np.mean(rel_vecs[:, : min(2, self.d)], axis=0)
            if self.d > 2:
                s[2:] = np.mean(np.sin(rel_vecs[:, : min(2, self.d)]), axis=0)[: self.d - 2]

            # Generate messages
            M = self.honest_messages(s, rng)
            if attack is not None and byz_mask.any():
                M = attack.craft(M, byz_mask, aggregator, rng)

            forces = np.zeros((self.n, 2), dtype=np.float64)
            step_honest_rewards: list[float] = []

            # Byzantine predators physically cooperate (message-only Byzantine model)
            for b in byz:
                bf = np.asarray(rel_vecs[b, :2], dtype=np.float64)
                bn = np.linalg.norm(bf)
                if bn > 1.0:
                    bf = bf / bn
                forces[b] = bf

            # Honest predators process aggregated message
            for i in honest_ids:
                nbr = self.agg_neighbors[i]
                m_hat = aggregator.aggregate(M[nbr])
                act = policy.act(m_hat)

                # Steering force
                f_steer = np.asarray(act[:2], dtype=np.float64)
                fn = np.linalg.norm(f_steer)
                if fn > 1.0:
                    f_steer = f_steer / fn
                forces[i] = f_steer

                # Tracking accuracy reward
                dist2 = float(np.sum((act[: min(2, self.d)] - s[: min(2, self.d)]) ** 2))
                step_honest_rewards.append(1.0 / (1.0 + dist2))
                deviations.append(float(np.linalg.norm(m_hat - s)))

            # Update predator physics
            pred_vel = pred_vel * self.damping + forces * self.dt
            pred_pos += pred_vel * self.dt

            # Prey reactive evasion physics: evade the nearest predator
            for p_idx in range(self.n_preys):
                diffs = prey_pos[p_idx][None, :] - pred_pos
                dists = np.linalg.norm(diffs, axis=1)
                nearest_idx = np.argmin(dists)
                evade_dir = diffs[nearest_idx] / (dists[nearest_idx] + 1e-6)
                prey_vel[p_idx] = prey_vel[p_idx] * 0.8 + evade_dir * 1.2 * self.dt
                prey_pos[p_idx] += prey_vel[p_idx] * self.dt

                # Check multi-agent encirclement capture: at least 2 predators within capture radius
                close_predators = np.sum(dists < self.capture_radius)
                if close_predators >= 2:
                    captured_counts += 1

            step_returns.append(float(np.mean(step_honest_rewards)) if step_honest_rewards else 0.0)

        mean_step_return = float(np.mean(step_returns)) if step_returns else 0.0
        # Capture bonus normalized into [0, 1] return metric
        capture_ratio = min(1.0, captured_counts / max(1, self.horizon))
        combined_return = 0.6 * mean_step_return + 0.4 * capture_ratio

        return {
            "return": float(combined_return),
            "mean_deviation": float(np.mean(deviations)) if deviations else 0.0,
            "max_deviation": float(np.max(deviations)) if deviations else 0.0,
            "capture_ratio": float(capture_ratio),
        }
