"""Message signing / authentication comparator (B5) — ADAPTER STUB with a dummy.

The paper compares message signing as an *orthogonal availability-cost* baseline, not as
the defense. A faithful implementation requires a real signature scheme and a model of the
availability cost (dropped/late messages). This module provides:

* ``MessageSigningComparator`` — a DUMMY oracle signer for pipeline testing only. It is
  given the Byzantine mask (an oracle it would not have in reality), drops forged messages,
  and additionally drops an ``availability_cost`` fraction of honest messages to represent
  verification latency/failure. It is clearly NOT a real cryptographic defense.
* A documented interface (``verify``/``aggregate_signed``) that a real signer must satisfy.

TODO (see TODO_IMPLEMENTATION.md): replace the oracle with a real signature scheme and a
measured availability-cost model before using this comparator for any claim.
"""
from __future__ import annotations

import numpy as np

from ..utils.logging import get_logger
from ..utils.validation import validate_messages

_log = get_logger(__name__)


class MessageSigningComparator:
    def __init__(self, availability_cost: float = 0.0, rng: np.random.Generator | None = None) -> None:
        if not (0.0 <= availability_cost < 1.0):
            raise ValueError("availability_cost must be in [0, 1).")
        self.availability_cost = float(availability_cost)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        _log.warning(
            "MessageSigningComparator is a DUMMY oracle signer (adapter stub), not a real "
            "cryptographic defense. See TODO_IMPLEMENTATION.md."
        )

    def verify(self, messages: np.ndarray, byzantine_mask: np.ndarray) -> np.ndarray:
        """Return a boolean 'accepted' mask. Dummy: reject forged; randomly drop honest."""
        M = validate_messages(messages)
        mask = np.asarray(byzantine_mask, dtype=bool)
        accepted = ~mask  # perfect (oracle) rejection of forged messages
        honest_idx = np.where(accepted)[0]
        n_drop = int(round(self.availability_cost * len(honest_idx)))
        if n_drop > 0:
            drop = self.rng.choice(honest_idx, size=n_drop, replace=False)
            accepted[drop] = False
        if not accepted.any():
            accepted[honest_idx[0] if len(honest_idx) else 0] = True  # keep >=1
        return accepted

    def aggregate_signed(self, messages: np.ndarray, byzantine_mask: np.ndarray) -> np.ndarray:
        M = validate_messages(messages)
        accepted = self.verify(M, byzantine_mask)
        return M[accepted].mean(axis=0)
