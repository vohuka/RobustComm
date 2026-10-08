"""Command-line interface for RobustComm.

Examples
--------
    python -m robustcomm.cli --help
    python -m robustcomm.cli info
    python -m robustcomm.cli demo --config configs/synthetic_demo.yaml
    python -m robustcomm.cli evaluate --config configs/experiment_template.yaml --out outputs/exp
    python -m robustcomm.cli ablate  --config configs/experiment_template.yaml --out outputs/exp
    python -m robustcomm.cli plot    --input outputs/exp --output outputs/exp/plots
    python -m robustcomm.cli theory  --operator krum --n 7 --f 2 --d 4 --B 0.5
"""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from . import __version__
from .aggregators.registry import AGGREGATORS
from .theory import bounds
from .utils.io import ensure_dir, load_yaml, write_run_metadata
from .utils.logging import get_logger
from .utils.seeding import set_global_seed
from .utils.validation import validate_config

_log = get_logger("robustcomm.cli")


def _cmd_info(_: argparse.Namespace) -> int:
    print(f"RobustComm v{__version__}")
    print("Byzantine-resilient learned communication for cooperative MARL.")
    print(f"Aggregation operators (RobAgg): {sorted(AGGREGATORS)}")
    print("Real experiments run on EC2 after data provisioning; local runs are synthetic.")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .experiments.demo import run_demo

    summary = run_demo(args.config)
    print(json.dumps(summary, indent=2))
    return 0


def _load(args: argparse.Namespace) -> dict:
    cfg = validate_config(load_yaml(args.config))
    return cfg


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .experiments.run_evaluation import run_evaluation

    cfg = _load(args)
    rng = set_global_seed(int(cfg["seed"]))
    out = args.out or cfg["output"]["dir"]
    ensure_dir(out)
    write_run_metadata(cfg, out, int(cfg["seed"]))
    df = run_evaluation(cfg, rng, out)
    print(f"wrote {len(df)} rows to {out}/evaluation.csv")
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    from .experiments.run_ablation import run_ablation

    cfg = _load(args)
    rng = set_global_seed(int(cfg["seed"]))
    out = args.out or cfg["output"]["dir"]
    ensure_dir(out)
    df = run_ablation(cfg, rng, out)
    print(f"wrote {len(df)} rows to {out}/ablation.csv")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    from .experiments.plot_results import make_all_plots

    made = make_all_plots(args.input, args.output)
    print("wrote plots:", [str(p) for p in made] or "(no input CSVs found)")
    return 0


def _cmd_theory(args: argparse.Namespace) -> int:
    dev = bounds.aggregate_deviation_bound(args.operator, args.n, args.f, args.d, args.B)
    out = {
        "operator": args.operator,
        "n": args.n,
        "f": args.f,
        "d": args.d,
        "B": args.B,
        "deviation_bound_l2": dev,
        "tolerance_fraction": bounds.tolerance_fraction(args.operator, args.n),
        "max_tolerated_byzantine": bounds.max_tolerated_byzantine(args.operator, args.n),
    }
    if args.operator == "krum":
        out["beta_K"] = bounds.krum_deviation_factor(args.n, args.f)
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="robustcomm", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="version", version=f"robustcomm {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="package info").set_defaults(func=_cmd_info)

    d = sub.add_parser("demo", help="run the synthetic smoke-test demo")
    d.add_argument("--config", default="configs/synthetic_demo.yaml")
    d.set_defaults(func=_cmd_demo)

    e = sub.add_parser("evaluate", help="run the evaluation sweep")
    e.add_argument("--config", required=True)
    e.add_argument("--out", default=None)
    e.set_defaults(func=_cmd_evaluate)

    a = sub.add_parser("ablate", help="run the ablation")
    a.add_argument("--config", required=True)
    a.add_argument("--out", default=None)
    a.set_defaults(func=_cmd_ablate)

    pl = sub.add_parser("plot", help="make plots from result CSVs")
    pl.add_argument("--input", required=True)
    pl.add_argument("--output", required=True)
    pl.set_defaults(func=_cmd_plot)

    th = sub.add_parser("theory", help="compute Section-9 bounds for given parameters")
    th.add_argument("--operator", choices=sorted(AGGREGATORS), default="krum")
    th.add_argument("--n", type=int, required=True)
    th.add_argument("--f", type=int, default=0)
    th.add_argument("--d", type=int, default=1)
    th.add_argument("--B", type=float, default=1.0)
    th.set_defaults(func=_cmd_theory)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
