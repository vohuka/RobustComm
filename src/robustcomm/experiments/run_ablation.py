"""Ablation runner (Sec. 10.10; RQ3/RQ5).

Isolates each component of RobustComm at a fixed malicious fraction: robust operator
choice (Krum / median / trimmed mean / mean), adversarial training on/off, ensembles
on/off, and continuous vs. discrete messages. Writes ``ablation.csv``.
Synthetic smoke-test outputs only.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from ..baselines.base import BaselineSpec, build_defense
from ..data.adapters import make_environment
from ..utils.io import ensure_dir, save_dataframe
from ..utils.logging import get_logger
from .run_evaluation import _episode_returns, build_attack

_log = get_logger(__name__)


def _eval_spec(cfg: dict[str, Any], spec: BaselineSpec, discrete: bool, rng: np.random.Generator) -> dict[str, float]:
    cfg = copy.deepcopy(cfg)
    cfg["env"]["discrete"] = discrete
    env = make_environment(cfg, rng)
    f = int(cfg["attack"]["f"])
    byz = list(range(f))
    attack = build_attack(cfg) if cfg["attack"]["enabled"] else None
    agg, policy = build_defense(spec, env, cfg, attack, byz, rng)
    episodes = int(cfg["eval"]["episodes"])
    seed = int(cfg.get("seed", 42))
    clean = _episode_returns(env, agg, None, policy, byz, seed, episodes)
    adv = _episode_returns(env, agg, attack, policy, byz, seed, episodes)
    clean_mean = float(clean.mean())
    adv_mean = float(adv.mean())
    return {
        "clean_reward": clean_mean,
        "adv_reward": adv_mean,
        "drop": clean_mean - adv_mean,
    }


def run_ablation(cfg: dict[str, Any], rng: np.random.Generator, out_dir: str | None = None) -> pd.DataFrame:
    f = int(cfg["attack"]["f"])
    beta = float(cfg["aggregator"]["beta"])

    variants: list[tuple[str, BaselineSpec, bool]] = [
        ("Full RobustComm (median)", BaselineSpec("median_full", "median", f, beta, True, True, "Full (Median+Adv+Ens)"), False),
        ("Full RobustComm (trimmed mean)", BaselineSpec("trmean_full", "trmean", f, beta, True, True, "Full (TrimmedMean+Adv+Ens)"), False),
        ("Full RobustComm (Krum)", BaselineSpec("krum_full", "krum", f, beta, True, True, "Full (Krum+Adv+Ens)"), False),
        ("Median - ensembles", BaselineSpec("median_noens", "median", f, beta, True, False, "Median + Adv (no ens)"), False),
        ("Median - adv training", BaselineSpec("median_noadv", "median", f, beta, False, True, "Median + Ens (no adv)"), False),
        ("Median alone (plain control)", BaselineSpec("median_plain", "median", f, beta, False, False, "Plain Median only"), False),
        ("Mean agg (no robust operator)", BaselineSpec("mean_adv", "mean", 0, beta, True, True, "Mean + Adv + Ens"), False),
        ("Messages = discrete", BaselineSpec("discrete_full", "median", f, beta, True, True, "Discrete messages"), True),
    ]

    rows = []
    for label, spec, discrete in variants:
        try:
            res = _eval_spec(cfg, spec, discrete, rng)
        except ValueError as exc:
            _log.warning("ablation '%s' skipped: %s", label, exc)
            continue
        rows.append(
            {
                "variant": label,
                "operator": spec.operator,
                "adversarial_training": spec.adversarial_training,
                "ensembles": spec.ensembles,
                "discrete": discrete,
                "clean_reward": res["clean_reward"],
                "adv_reward": res["adv_reward"],
                "drop": res["drop"],
            }
        )
        _log.info("%-34s clean=%.3f adv=%.3f drop=%.3f", label, res["clean_reward"], res["adv_reward"], res["drop"])

    df = pd.DataFrame(rows)
    if out_dir is not None and not df.empty:
        ensure_dir(out_dir)
        save_dataframe(df, f"{out_dir}/ablation.csv")
    return df
