"""Input and config validation with clear error messages.

Kept dependency-free (numpy only) so it can be used everywhere without import cycles.
"""
from __future__ import annotations

from typing import Any

import numpy as np

VALID_OPERATORS = {"krum", "median", "trmean", "mean"}
VALID_TOPOLOGIES = {"broadcast", "ring", "star", "neighborhood"}
VALID_NORMS = {"linf", "l2"}
VALID_ENVS = {"synthetic_comm", "dial", "mpe", "smac"}


def validate_messages(messages: np.ndarray) -> np.ndarray:
    """Validate an ``(n, d)`` array of inbound messages; return it as float64."""
    arr = np.asarray(messages, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"messages must be 2D (n_senders, msg_dim); got shape {arr.shape}.")
    if arr.shape[0] < 1:
        raise ValueError("messages must contain at least one sender.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("messages contain non-finite values (NaN/Inf).")
    return arr


def check_unit_interval(values: np.ndarray, name: str = "values") -> np.ndarray:
    """Validate that values lie in [0, 1] (used for risk-like signals, not messages)."""
    arr = np.asarray(values, dtype=np.float64)
    if not np.all((arr >= 0.0) & (arr <= 1.0)):
        raise ValueError(f"{name} must lie in [0, 1].")
    return arr


def validate_krum_regime(n: int, f: int) -> None:
    """Krum requires 2f + 2 < n (Assumption A2 / Def 3.1)."""
    if f < 0:
        raise ValueError("f must be non-negative.")
    if not (2 * f + 2 < n):
        raise ValueError(
            f"Krum regime violated: need 2f+2 < n_i, got 2*{f}+2={2 * f + 2} !< {n}. "
            "Krum is undefined outside its Byzantine regime; use a smaller f or a larger "
            "neighbourhood (see audits/math_to_code_audit.md)."
        )


def validate_majority_regime(n: int, f: int) -> bool:
    """Median/trimmed mean need f/n < 1/2. Returns True if the guarantee holds."""
    return f < n / 2.0


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Light structural validation of a resolved config dict with helpful messages."""
    for key in ("env", "aggregator", "attack", "eval", "theory", "output"):
        if key not in cfg:
            raise ValueError(f"config missing required section '{key}'.")
    env, agg, atk = cfg["env"], cfg["aggregator"], cfg["attack"]
    if env["name"] not in VALID_ENVS:
        raise ValueError(f"env.name must be one of {sorted(VALID_ENVS)}.")
    if env["topology"] not in VALID_TOPOLOGIES:
        raise ValueError(f"env.topology must be one of {sorted(VALID_TOPOLOGIES)}.")
    if agg["operator"] not in VALID_OPERATORS:
        raise ValueError(f"aggregator.operator must be one of {sorted(VALID_OPERATORS)}.")
    if atk["norm"] not in VALID_NORMS:
        raise ValueError(f"attack.norm must be one of {sorted(VALID_NORMS)}.")
    if atk["mode"] not in {"static", "adaptive"}:
        raise ValueError("attack.mode must be 'static' or 'adaptive'.")
    if env["n_agents"] < 2:
        raise ValueError("env.n_agents must be >= 2 (communication needs >= 2 agents).")
    if env["msg_dim"] < 1:
        raise ValueError("env.msg_dim must be >= 1.")
    return cfg
