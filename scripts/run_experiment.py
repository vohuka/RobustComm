#!/usr/bin/env python3
"""Run a RobustComm experiment stage from a config.

    python scripts/run_experiment.py --config configs/experiment_template.yaml --stage evaluate
    python scripts/run_experiment.py --config configs/experiment_template.yaml --stage ablate
    python scripts/run_experiment.py --config configs/experiment_template.yaml --stage all

With ``env.name: synthetic_comm`` this runs offline on synthetic data. Set a real
``env.name`` + ``env.data_root`` on EC2 (see docs/ec2_experiment_guide.md) to run on a real
benchmark; this script never downloads data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robustcomm.experiments.run_ablation import run_ablation  # noqa: E402
from robustcomm.experiments.run_evaluation import run_evaluation  # noqa: E402
from robustcomm.utils.io import ensure_dir, load_yaml, write_run_metadata  # noqa: E402
from robustcomm.utils.seeding import set_global_seed  # noqa: E402
from robustcomm.utils.validation import validate_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=["evaluate", "ablate", "all"], default="all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = validate_config(load_yaml(args.config))
    rng = set_global_seed(int(cfg["seed"]))
    out = args.out or cfg["output"]["dir"]
    ensure_dir(out)
    write_run_metadata(cfg, out, int(cfg["seed"]))

    if args.stage in ("evaluate", "all"):
        df = run_evaluation(cfg, rng, out)
        print(f"[evaluate] wrote {len(df)} rows to {out}/evaluation.csv")
    if args.stage in ("ablate", "all"):
        df = run_ablation(cfg, rng, out)
        print(f"[ablate] wrote {len(df)} rows to {out}/ablation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
