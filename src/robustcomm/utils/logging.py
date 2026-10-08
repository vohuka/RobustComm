"""Logging helper. Library code uses ``logging`` (never ``print``)."""
from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str = "robustcomm", level: int = logging.INFO) -> logging.Logger:
    """Return a module logger, configuring a single stream handler once."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        root = logging.getLogger("robustcomm")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logger
