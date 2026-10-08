"""Communication topologies (Sec. 4): broadcast, ring, star, neighbourhood.

Each builder returns a :class:`~robustcomm.data.schemas.Topology` whose ``inbound[i]``
lists the senders whose messages agent ``i`` aggregates. Aggregation is always
per-agent over its own inbound neighbourhood (no global collector).
"""
from __future__ import annotations

import numpy as np

from .schemas import Topology


def build_topology(
    name: str,
    n_agents: int,
    neighbors: int = 4,
    rng: np.random.Generator | None = None,
) -> Topology:
    """Construct an inbound-neighbourhood topology.

    Parameters
    ----------
    name : ``broadcast`` | ``ring`` | ``star`` | ``neighborhood``.
    n_agents : number of agents ``n``.
    neighbors : inbound size for ``ring`` and ``neighborhood`` (clamped to ``n-1``).
    rng : generator for the random ``neighborhood`` graph (reproducibility).
    """
    if n_agents < 2:
        raise ValueError("n_agents must be >= 2.")
    k = max(1, min(neighbors, n_agents - 1))

    if name == "broadcast":
        inbound = {i: [j for j in range(n_agents) if j != i] for i in range(n_agents)}
    elif name == "ring":
        # each agent reads the next k agents around the cycle
        inbound = {
            i: [(i + off) % n_agents for off in range(1, k + 1)] for i in range(n_agents)
        }
    elif name == "star":
        # hub = agent 0; peripherals read the hub, hub reads all peripherals
        inbound = {0: list(range(1, n_agents))}
        for i in range(1, n_agents):
            inbound[i] = [0]
    elif name == "neighborhood":
        rng = rng if rng is not None else np.random.default_rng(0)
        inbound = {}
        for i in range(n_agents):
            others = [j for j in range(n_agents) if j != i]
            chosen = rng.choice(others, size=k, replace=False)
            inbound[i] = sorted(int(x) for x in chosen)
    else:
        raise ValueError(
            f"unknown topology '{name}'; expected broadcast|ring|star|neighborhood."
        )
    return Topology(name=name, n_agents=n_agents, inbound=inbound)
