"""Theoretical bounds from Section 9 of the paper, as explicit functions.

These are *worst-case upper bounds* (the paper states they are not tight). The empirical
tests in ``tests/test_theory_bounds.py`` check the inequality direction: the realised
deviation of a robust aggregate never exceeds the bound these functions return, while the
mean has no such bound (Prop 9.1).

Symbols: ``n`` = neighbourhood size, ``f`` = Byzantine count, ``d`` = message dim,
``B`` = honest dispersion ``B_i`` (A1), ``L_pi``/``L_r`` = Lipschitz constants (A3/A4),
``gamma`` = discount, ``T`` = horizon.
"""
from __future__ import annotations

import math


# ----------------------------------------------------------------------------------
# Lemma 9.2 (Krum) and Lemma 9.3 (median) deviation bounds
# ----------------------------------------------------------------------------------
def krum_deviation_factor(n: int, f: int) -> float:
    r"""Krum deviation factor :math:`\beta_K = 2\sqrt{n-f-2}+1` (Lem 9.2).

    Requires the regime ``2f+2 < n`` (equivalently ``n-f-2 >= f >= 0``); raises otherwise.
    """
    if not (2 * f + 2 < n):
        raise ValueError(f"Krum regime 2f+2<n violated (f={f}, n={n}); beta_K undefined.")
    return 2.0 * math.sqrt(n - f - 2) + 1.0


def krum_deviation_bound(n: int, f: int, B: float) -> float:
    """Upper bound ``beta_K * B`` on ``||m_hat - m_bar^H||_2`` for Krum (Lem 9.2)."""
    return krum_deviation_factor(n, f) * float(B)


def median_deviation_factor(d: int) -> float:
    r"""Median deviation factor :math:`\sqrt{d}` for the :math:`\ell_2` bound (Lem 9.3)."""
    if d < 1:
        raise ValueError("d must be >= 1.")
    return math.sqrt(d)


def median_deviation_bound(d: int, B: float, norm: str = "l2") -> float:
    """Median deviation bound: ``B`` in ``l_inf``; ``sqrt(d)*B`` in ``l_2`` (Lem 9.3)."""
    if norm == "linf":
        return float(B)
    if norm == "l2":
        return median_deviation_factor(d) * float(B)
    raise ValueError("norm must be 'l2' or 'linf'.")


def aggregate_deviation_bound(operator: str, n: int, f: int, d: int, B: float) -> float:
    """Dispatch the l2 deviation bound ``beta_i * B`` by operator (Lem 9.2/9.3).

    The mean has no finite bound (Prop 9.1); we return ``inf`` for it.
    """
    operator = operator.lower()
    if operator == "krum":
        return krum_deviation_bound(n, f, B)
    if operator in ("median", "trmean"):
        return median_deviation_bound(d, B, norm="l2")
    if operator == "mean":
        return math.inf
    raise ValueError(f"unknown operator '{operator}'.")


# ----------------------------------------------------------------------------------
# Theorem 9.4 — bounded team-return degradation
# ----------------------------------------------------------------------------------
def episodic_factor(gamma: float, horizon: int) -> float:
    r"""Discount sum :math:`\frac{1-\gamma^T}{1-\gamma}`; equals ``T`` when ``gamma==1``."""
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1].")
    if math.isclose(gamma, 1.0):
        return float(horizon)
    return (1.0 - gamma**horizon) / (1.0 - gamma)


def per_step_degradation_bound(
    operator: str, n: int, f: int, d: int, B: float, L_pi: float, L_r: float, n_affected: int = 1
) -> float:
    r"""Per-step team-reward degradation bound :math:`L_r \sum_i L_\pi \beta_i B_i`
    (Thm 9.4). Assumes ``n_affected`` identical affected agents for simplicity."""
    beta_B = aggregate_deviation_bound(operator, n, f, d, B)
    return float(L_r) * n_affected * float(L_pi) * beta_B


def return_degradation_bound(
    operator: str,
    n: int,
    f: int,
    d: int,
    B: float,
    L_pi: float,
    L_r: float,
    gamma: float,
    horizon: int,
    n_affected: int = 1,
) -> dict[str, float]:
    """Return per-step and episodic degradation bounds (Thm 9.4).

    Budget-independent: no ``epsilon`` appears. Returns ``inf`` bounds for the mean.
    """
    per_step = per_step_degradation_bound(operator, n, f, d, B, L_pi, L_r, n_affected)
    episodic = per_step * episodic_factor(gamma, horizon)
    return {"per_step": per_step, "episodic": episodic}


# ----------------------------------------------------------------------------------
# Corollary 9.5 — Byzantine tolerance fraction
# ----------------------------------------------------------------------------------
def tolerance_fraction(operator: str, n: int) -> float:
    r"""Per-neighbourhood tolerated malicious *fraction* (Cor 9.5).

    Krum: :math:`(n-2)/(2n)`. Median / trimmed mean: approaches :math:`1/2` (strict).
    """
    operator = operator.lower()
    if operator == "krum":
        return (n - 2) / (2.0 * n)
    if operator in ("median", "trmean"):
        return 0.5  # strict upper limit; max tolerated integer is ceil(n/2)-1
    if operator == "mean":
        return 0.0
    raise ValueError(f"unknown operator '{operator}'.")


def max_tolerated_byzantine(operator: str, n: int) -> int:
    """Largest tolerated Byzantine *count* ``f`` in a neighbourhood of size ``n`` (Cor 9.5)."""
    operator = operator.lower()
    if operator == "krum":
        # 2f+2 < n  ->  f < (n-2)/2 (strict); largest integer is floor((n-3)/2).
        return max(0, math.floor((n - 3) / 2.0))
    if operator in ("median", "trmean"):
        return max(0, math.ceil(n / 2.0) - 1)  # f/n < 1/2  ->  f <= ceil(n/2)-1
    if operator == "mean":
        return 0
    raise ValueError(f"unknown operator '{operator}'.")


def global_tolerance_fraction(operator: str, neighbourhood_sizes: list[int]) -> float:
    """Global tolerated fraction = min over neighbourhoods (Cor 9.5)."""
    if not neighbourhood_sizes:
        raise ValueError("neighbourhood_sizes must be non-empty.")
    return min(tolerance_fraction(operator, n) for n in neighbourhood_sizes)


# ----------------------------------------------------------------------------------
# Proposition 9.6 — robustness/utility loss bound
# ----------------------------------------------------------------------------------
def robustness_utility_loss_bound(
    L_pi: float, L_r: float, delta_sig: float, gamma: float | None = None, horizon: int | None = None
) -> dict[str, float]:
    r"""Clean-reward loss from discarding a correct minority signal (Prop 9.6).

    Per-step bound :math:`L_r L_\pi \Delta_{sig}`; episodic multiplies by the discount sum
    if ``gamma`` and ``horizon`` are given.
    """
    per_step = float(L_r) * float(L_pi) * float(delta_sig)
    out = {"per_step": per_step}
    if gamma is not None and horizon is not None:
        out["episodic"] = per_step * episodic_factor(gamma, horizon)
    return out
