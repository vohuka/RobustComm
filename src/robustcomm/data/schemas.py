"""Structured representations of the paper's objects (System model, Sec. 4-5).

Messages live in ``R^d`` (not ``[0,1]``); a communication *round* bundles all senders'
messages, which of them are Byzantine, and (for synthetic evaluation) the honest
reference ``m_bar^H`` and the task target signal ``s``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Topology:
    """Inbound-neighbourhood structure (Sec. 4).

    ``inbound[i]`` is the list of sender ids whose messages agent ``i`` aggregates.
    """
    name: str
    n_agents: int
    inbound: dict[int, list[int]]

    def neighbourhood_size(self, agent: int) -> int:
        return len(self.inbound[agent])

    def min_neighbourhood_size(self) -> int:
        return min(len(v) for v in self.inbound.values())


@dataclass
class CommRound:
    """One communication step.

    Attributes
    ----------
    messages : (n, d) honest messages emitted by all senders.
    byzantine : (n,) bool mask of Byzantine senders.
    honest_reference : optional (d,) honest consensus ``m_bar^H`` (synthetic eval).
    target : optional (d,) task signal ``s`` the team must recover (synthetic eval).
    """
    messages: np.ndarray
    byzantine: np.ndarray
    honest_reference: np.ndarray | None = None
    target: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.messages = np.asarray(self.messages, dtype=np.float64)
        self.byzantine = np.asarray(self.byzantine, dtype=bool)
        if self.messages.ndim != 2:
            raise ValueError("messages must be (n, d).")
        if self.byzantine.shape[0] != self.messages.shape[0]:
            raise ValueError("byzantine mask length must match number of senders.")

    @property
    def n_senders(self) -> int:
        return self.messages.shape[0]

    @property
    def msg_dim(self) -> int:
        return self.messages.shape[1]

    @property
    def n_byzantine(self) -> int:
        return int(self.byzantine.sum())

    def honest_messages(self) -> np.ndarray:
        return self.messages[~self.byzantine]


@dataclass
class AggregationResult:
    """Output of a robust aggregator with decision metadata (Alg 1)."""
    aggregate: np.ndarray
    operator: str
    n_senders: int
    f_assumed: int
    guarantee_valid: bool
    group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalRecord:
    """One row of an evaluation sweep (Sec. 10.6)."""
    operator: str
    fraction: float
    clean_reward: float
    adv_reward: float
    attack_success: float
    aggregate_deviation: float
    n_agents: int
    msg_dim: int
    topology: str
    mode: str
    colluding: bool
    seed: int
    note: str = "synthetic smoke-test output, not a paper result"
