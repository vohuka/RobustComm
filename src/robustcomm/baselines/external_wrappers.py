"""Adapters for external comm-robustness defenses (Sec. 10.5 comparators).

The paper compares against external defenses such as Sun et al. (certified message-ensemble,
ICLR 2023) and Yu et al. (active/reliability-weighting defense, AAAI 2024). Those require
their own repositories/models and are NOT bundled here. This module defines a clean
interface plus a dummy so the pipeline and tests run; a real backend must be supplied on
EC2 (see docs/baseline_adapters.md, TODO_IMPLEMENTATION.md).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..utils.validation import validate_messages


class DefenseBackend(Protocol):
    """Expected interface a real external defense backend must implement."""

    def aggregate(self, messages: np.ndarray) -> np.ndarray:  # (n, d) -> (d,)
        ...


class ExternalDefenseAdapter:
    """Wrap a real external defense backend; raise clearly if none is provided."""

    def __init__(self, name: str, backend: DefenseBackend | None = None) -> None:
        self.name = name
        self.backend = backend

    def aggregate(self, messages: np.ndarray) -> np.ndarray:
        M = validate_messages(messages)
        if self.backend is None:
            raise NotImplementedError(
                f"External defense '{self.name}' has no backend. Provide an object with "
                "`.aggregate(messages)` (e.g. a wrapper around the authors' released code). "
                "See docs/baseline_adapters.md and TODO_IMPLEMENTATION.md."
            )
        return self.backend.aggregate(M)


class DummyExternalDefense:
    """Dummy backend for tests ONLY (coordinate-wise median). Not the real method."""

    def aggregate(self, messages: np.ndarray) -> np.ndarray:
        return np.median(validate_messages(messages), axis=0)
