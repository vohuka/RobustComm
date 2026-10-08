"""I/O helpers: config loading, result writing, and run-provenance metadata.

Provenance is part of reproducibility: every experiment writes a ``run_metadata.json``
recording the resolved config, seed, timestamp, and library versions, plus a hash of the
config so results can be traced back to the exact settings that produced them.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    return data


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(obj: Any, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(obj), fh, indent=2, sort_keys=True)
    return p


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)
    return p


def config_hash(config: dict[str, Any]) -> str:
    """Stable short hash of a config dict (for result provenance)."""
    blob = json.dumps(_jsonable(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def write_run_metadata(config: dict[str, Any], out_dir: str | Path, seed: int) -> Path:
    """Write ``run_metadata.json`` capturing config, seed, versions and timestamp."""
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "config_hash": config_hash(config),
        "config": _jsonable(config),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "note": "Synthetic smoke-test unless env.name is a real adapter with data_root set.",
    }
    return save_json(meta, Path(out_dir) / "run_metadata.json")
