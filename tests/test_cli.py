"""CLI smoke tests and a tiny evaluation run on synthetic data."""
import numpy as np

from robustcomm.cli import main
from robustcomm.experiments.run_evaluation import run_evaluation
from robustcomm.experiments.plot_results import make_all_plots
from tests.test_defense_pipeline import CFG


def test_cli_info():
    assert main(["info"]) == 0


def test_cli_theory():
    assert main(["theory", "--operator", "krum", "--n", "7", "--f", "2", "--d", "4", "--B", "0.5"]) == 0


def test_run_evaluation_produces_rows(tmp_path):
    df = run_evaluation(CFG, np.random.default_rng(0), out_dir=str(tmp_path))
    assert not df.empty
    assert (tmp_path / "evaluation.csv").exists()
    assert (tmp_path / "evaluation_summary.json").exists()
    # every row is explicitly labelled synthetic (no fake paper results)
    assert all("synthetic" in n for n in df["note"].unique())


def test_cli_plot(tmp_path):
    run_evaluation(CFG, np.random.default_rng(0), out_dir=str(tmp_path))
    made = make_all_plots(str(tmp_path), str(tmp_path / "plots"))
    assert any(p.name == "reward_vs_fraction.png" for p in made)
    assert (tmp_path / "plots" / "reward_vs_fraction.png").exists()
