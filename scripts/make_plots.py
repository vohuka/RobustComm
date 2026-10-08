#!/usr/bin/env python3
"""Generate plots from result CSVs produced by run_experiment.py / the demo.

    python scripts/make_plots.py --input outputs/synthetic_demo --output outputs/synthetic_demo/plots

Plots are produced ONLY from the CSV files; nothing is hard-coded. Synthetic inputs yield
plots labelled as synthetic smoke-test figures.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robustcomm.experiments.plot_results import make_all_plots  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="directory containing evaluation.csv / ablation.csv")
    ap.add_argument("--output", required=True, help="directory for the .png plots")
    args = ap.parse_args()
    made = make_all_plots(args.input, args.output)
    print("wrote:", [str(p) for p in made] or "(no input CSVs found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
