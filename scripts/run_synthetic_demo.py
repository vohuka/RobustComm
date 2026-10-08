#!/usr/bin/env python3
"""Run the synthetic smoke-test demo.

    python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml

Outputs are written to the config's ``output.dir`` and are labelled
"synthetic smoke-test output, not a paper result".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robustcomm.experiments.demo import run_demo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/synthetic_demo.yaml")
    args = ap.parse_args()
    summary = run_demo(args.config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
