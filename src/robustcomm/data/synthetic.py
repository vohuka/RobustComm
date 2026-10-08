"""Standalone synthetic-data generator (for examples, tests, and adapter schema demos).

Produces a small table of communication rounds in the documented message schema
(``docs/data_format.md``). This is synthetic data for pipeline testing only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_message_rounds(
    n_rounds: int = 20,
    n_senders: int = 7,
    msg_dim: int = 4,
    honest_dispersion: float = 0.5,
    n_byzantine: int = 2,
    signal_scale: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a long-format DataFrame of synthetic messages.

    Columns: ``round, sender, byzantine, target_k, message_k`` for k in ``0..d-1``.
    Honest messages lie within an ``l2`` ball of radius ``honest_dispersion`` around the
    per-round target; Byzantine messages are large-magnitude outliers.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_rounds):
        s = rng.uniform(-signal_scale, signal_scale, size=msg_dim)
        dirs = rng.normal(size=(n_senders, msg_dim))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        radii = honest_dispersion * rng.uniform(0, 1, size=(n_senders, 1)) ** (1.0 / msg_dim)
        M = s[None, :] + dirs * radii
        byz = np.zeros(n_senders, dtype=bool)
        byz[:n_byzantine] = True
        M[byz] = rng.uniform(-10, 10, size=(int(byz.sum()), msg_dim))  # outliers
        for j in range(n_senders):
            row = {"round": r, "sender": j, "byzantine": bool(byz[j])}
            for k in range(msg_dim):
                row[f"target_{k}"] = float(s[k])
                row[f"message_{k}"] = float(M[j, k])
            rows.append(row)
    return pd.DataFrame(rows)
