"""Plotting (Sec. 10 figures). Reads result CSVs ONLY; never hard-codes numbers.

Every figure title carries the label "SYNTHETIC smoke-test - not a paper result" when the
inputs come from the synthetic environment, so plots cannot be mistaken for paper results.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; no display required
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_SYN = "SYNTHETIC smoke-test - not a paper result"


def plot_reward_vs_fraction(eval_csv: str | Path, out_png: str | Path) -> Path:
    df = pd.read_csv(eval_csv)
    fig, ax = plt.subplots(figsize=(6, 4))
    for op, sub in df.groupby("operator"):
        sub = sub.sort_values("fraction")
        ax.plot(sub["fraction"], sub["adv_reward"], marker="o", label=op)
    ax.set_xlabel(r"malicious fraction $\phi = f/n$")
    ax.set_ylabel("reward-under-attack")
    ax.set_title(f"Reward-under-attack vs. fraction\n[{_SYN}]", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_tradeoff(eval_csv: str | Path, out_png: str | Path) -> Path:
    df = pd.read_csv(eval_csv)
    agg = df.groupby("operator")[["clean_reward", "adv_reward"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(agg["clean_reward"], agg["adv_reward"])
    for _, r in agg.iterrows():
        ax.annotate(r["operator"], (r["clean_reward"], r["adv_reward"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("clean reward (mean over fractions)")
    ax.set_ylabel("reward-under-attack (mean over fractions)")
    ax.set_title(f"Robustness-clean-reward tradeoff\n[{_SYN}]", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_ablation(ablation_csv: str | Path, out_png: str | Path) -> Path:
    df = pd.read_csv(ablation_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    y = range(len(df))
    ax.barh(list(y), df["adv_reward"], alpha=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["variant"], fontsize=7)
    ax.set_xlabel("reward-under-attack")
    ax.set_title(f"Component ablation\n[{_SYN}]", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def make_all_plots(input_dir: str | Path, output_dir: str | Path) -> list[Path]:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    made: list[Path] = []
    if (input_dir / "evaluation.csv").exists():
        made.append(plot_reward_vs_fraction(input_dir / "evaluation.csv", output_dir / "reward_vs_fraction.png"))
        made.append(plot_tradeoff(input_dir / "evaluation.csv", output_dir / "tradeoff.png"))
    if (input_dir / "ablation.csv").exists():
        made.append(plot_ablation(input_dir / "ablation.csv", output_dir / "ablation.png"))
    return made
