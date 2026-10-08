"""Bootstrap confidence intervals (numpy only; no scipy dependency)."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(
    data: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    statistic: Callable[[np.ndarray], float] = np.mean,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Percentile bootstrap CI for ``statistic`` at level ``1 - alpha``.

    Returns ``{point, lo, hi, n}``; degenerate inputs yield ``nan`` bounds rather than
    raising, so callers can report "insufficient samples" gracefully.
    """
    arr = np.asarray(data, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    if n == 1:
        v = float(statistic(arr))
        return {"point": v, "lo": v, "hi": v, "n": 1}
    rng = rng if rng is not None else np.random.default_rng(0)
    boots = np.array([statistic(rng.choice(arr, size=n, replace=True)) for _ in range(int(n_boot))])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(statistic(arr)), "lo": float(lo), "hi": float(hi), "n": n}
