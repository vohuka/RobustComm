"""Krum aggregation (Def 3.1, Alg 1).

Krum selects the inbound message minimising the sum of squared Euclidean distances to
its ``n_i - f - 2`` nearest neighbours. It requires the regime ``2f + 2 < n_i``
(Assumption A2); outside the regime it is undefined and we raise ``ValueError`` rather
than return a non-robust result (see ``audits/math_to_code_audit.md``).
"""
from __future__ import annotations

import numpy as np

from ..utils.validation import validate_krum_regime, validate_messages
from .base import RobustAggregator


class KrumAggregator(RobustAggregator):
    name = "krum"

    def __init__(self, f: int) -> None:
        if f < 0:
            raise ValueError("f (assumed Byzantine count) must be non-negative.")
        self.f = int(f)

    def guarantee_holds(self, n: int) -> bool:
        return 2 * self.f + 2 < n

    def aggregate(self, messages: np.ndarray) -> np.ndarray:
        r"""Return :math:`m_{i^\star}` with :math:`i^\star=\arg\min_j s(j)` where
        :math:`s(j)=\sum_{j\to k}\lVert m_j-m_k\rVert_2^2` over the ``n_i - f - 2``
        nearest neighbours of ``j`` (Def 3.1)."""
        M = validate_messages(messages)
        n = M.shape[0]
        validate_krum_regime(n, self.f)
        m = n - self.f - 2  # number of nearest neighbours to sum over (>= 1 by regime)

        # Pairwise squared Euclidean distances; exclude self via +inf on the diagonal.
        diff = M[:, None, :] - M[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        np.fill_diagonal(d2, np.inf)

        # Sum of the m smallest distances for each candidate.
        nearest = np.sort(d2, axis=1)[:, :m]
        scores = nearest.sum(axis=1)
        i_star = int(np.argmin(scores))  # deterministic tie-break (stable argmin)
        return M[i_star].copy()
