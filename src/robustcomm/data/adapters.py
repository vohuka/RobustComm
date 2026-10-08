"""Real cooperative learned-communication environment adapters (Sec. 10.2).

These are interfaces + stubs for the paper's real environments. They NEVER download data.
On EC2, provision the environment package and data, set ``env.data_root``, and wire the
adapter to the underlying simulator (see ``docs/ec2_experiment_guide.md`` and
``docs/data_format.md``). Until then they raise a clear, actionable error.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class EnvAdapter(ABC):
    """Interface a real environment adapter must implement to plug into the runner."""

    name: str = "base"

    @property
    @abstractmethod
    def n_agents(self) -> int: ...

    @property
    @abstractmethod
    def msg_dim(self) -> int: ...

    @abstractmethod
    def rollout(self, aggregator: Any, attack: Any, policy: Any, byzantine_ids: list[int],
                rng: np.random.Generator) -> dict[str, float]:
        """Run one episode and return at least {'return', 'mean_deviation'}."""


class _UnavailableEnv(EnvAdapter):
    """Base for stubs: fail loudly with an install/provision hint (never silently)."""

    def __init__(self, name: str, data_root: str | None, hint: str) -> None:
        self._name = name
        self._hint = hint
        if data_root is None:
            raise NotImplementedError(
                f"[{name}] real environment not provisioned. Set env.data_root and install "
                f"the environment. {hint}"
            )
        if not Path(data_root).exists():
            raise FileNotFoundError(
                f"[{name}] env.data_root '{data_root}' does not exist on this machine. {hint}"
            )
        raise NotImplementedError(
            f"[{name}] adapter is a stub: wire it to the simulator on EC2. {hint}"
        )

    @property
    def n_agents(self) -> int:  # pragma: no cover - stub
        raise NotImplementedError

    @property
    def msg_dim(self) -> int:  # pragma: no cover - stub
        raise NotImplementedError

    def rollout(self, aggregator, attack, policy, byzantine_ids, rng):  # pragma: no cover
        raise NotImplementedError


class DIALEnvAdapter(_UnavailableEnv):
    """DIAL switch-riddle referential/coordination tasks (Foerster et al., 2016)."""

    def __init__(self, data_root: str | None = None, **_: Any) -> None:
        super().__init__(
            "dial",
            data_root,
            "Install the DIAL/switch-riddle environment and expose per-step messages; "
            "expected layout in docs/data_format.md.",
        )


class MPEEnvAdapter(EnvAdapter):
    """MPE particle communication adapter (Lowe et al., 2017)."""

    name: str = "mpe"

    def __init__(self, n_agents: int = 7, msg_dim: int = 4, **kwargs: Any) -> None:
        from ..envs.mpe_comm import MPECommEnv
        self._env = MPECommEnv(n_agents=n_agents, msg_dim=msg_dim, **kwargs)

    @property
    def n(self) -> int:
        return self._env.n

    @property
    def n_agents(self) -> int:
        return self._env.n

    @property
    def d(self) -> int:
        return self._env.d

    @property
    def msg_dim(self) -> int:
        return self._env.d

    @property
    def horizon(self) -> int:
        return self._env.horizon

    @property
    def agg_neighbors(self) -> dict[int, list[int]]:
        return self._env.agg_neighbors

    def min_agg_size(self) -> int:
        return self._env.min_agg_size()

    def check_operator_compatible(self, aggregator: Any) -> None:
        self._env.check_operator_compatible(aggregator)

    def sample_target(self, rng: np.random.Generator) -> np.ndarray:
        return self._env.sample_target(rng)

    def honest_messages(self, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self._env.honest_messages(s, rng)

    def _reward(self, action: np.ndarray, s: np.ndarray) -> float:
        return self._env._reward(action, s)

    def rollout(self, aggregator: Any, attack: Any, policy: Any, byzantine_ids: list[int],
                rng: np.random.Generator) -> dict[str, float]:
        return self._env.rollout(aggregator, attack, policy, byzantine_ids, rng)


class SMACEnvAdapter(_UnavailableEnv):
    """SMAC micromanagement (StarCraft II; Samvelyan et al., 2019)."""

    def __init__(self, data_root: str | None = None, **_: Any) -> None:
        super().__init__(
            "smac",
            data_root,
            "Install StarCraft II + SMAC maps (license required); expected layout in "
            "docs/data_format.md.",
        )


_REAL_ADAPTERS = {"dial": DIALEnvAdapter, "mpe": MPEEnvAdapter, "smac": SMACEnvAdapter}


def make_environment(cfg: dict[str, Any], rng: np.random.Generator) -> Any:
    """Factory: synthetic env for ``synthetic_comm``; a real adapter otherwise.

    The synthetic env is imported lazily to avoid a package import cycle.
    """
    env_cfg = cfg["env"]
    name = env_cfg["name"]
    if name == "synthetic_comm":
        from ..envs.synthetic_comm import SyntheticCommEnv  # lazy import
        from .topologies import build_topology

        topo = build_topology(
            env_cfg["topology"], env_cfg["n_agents"], env_cfg.get("neighbors", 4), rng
        )
        return SyntheticCommEnv(
            n_agents=env_cfg["n_agents"],
            msg_dim=env_cfg["msg_dim"],
            topology=topo,
            honest_dispersion=env_cfg["honest_dispersion"],
            horizon=env_cfg["horizon"],
            discrete=env_cfg.get("discrete", False),
            codebook_levels=env_cfg.get("codebook_levels", 5),
        )
    if name in ("mpe", "mpe_particle_comm"):
        from ..envs.mpe_comm import MPECommEnv
        from .topologies import build_topology

        topo = build_topology(
            env_cfg.get("topology", "broadcast"),
            env_cfg["n_agents"],
            env_cfg.get("neighbors", 4),
            rng,
        )
        return MPECommEnv(
            n_agents=env_cfg["n_agents"],
            msg_dim=env_cfg["msg_dim"],
            topology=topo,
            honest_dispersion=env_cfg.get("honest_dispersion", 0.5),
            horizon=env_cfg.get("horizon", 25),
            discrete=env_cfg.get("discrete", False),
            codebook_levels=env_cfg.get("codebook_levels", 5),
            arena_size=env_cfg.get("arena_size", 1.5),
        )
    if name in _REAL_ADAPTERS:
        return _REAL_ADAPTERS[name](data_root=env_cfg.get("data_root"))
    raise ValueError(f"unknown env.name '{name}'.")

