"""Deterministic seeding utilities.

All stochastic behaviour in RobustComm flows through a :class:`numpy.random.Generator`
created here, so that fixing ``seed`` makes runs reproducible (see
``docs/reproducibility.md``). We deliberately avoid global mutable RNG state in library
code; scripts may call :func:`set_global_seed` for convenience.
"""
from __future__ import annotations

import os
import random

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return a fresh, independent NumPy Generator seeded by ``seed``."""
    return np.random.default_rng(int(seed))


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python ``random``, NumPy's legacy global RNG, and ``PYTHONHASHSEED``.

    Returns a dedicated :class:`numpy.random.Generator` that library code should use
    explicitly rather than relying on global state.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return make_rng(seed)
