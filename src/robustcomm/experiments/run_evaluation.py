"""Evaluation runner (Sec. 10 protocol; RQ1/RQ2/RQ4).

For each baseline/method (Sec. 10.5) and each malicious fraction in the sweep, measure the
clean team return and the reward-under-attack, plus the aggregate-deviation diagnostic and
the message-attack success rate. Writes ``evaluation.csv`` and a summary JSON. All outputs
are synthetic smoke-test outputs unless a real environment adapter is provisioned.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from ..attacks.fgsm import FGSMMessageAttack
from ..baselines.base import build_defense, standard_baselines
from ..data.adapters import make_environment
from ..data.schemas import EvalRecord
from ..metrics.metrics import attack_success_rate
from ..utils.io import ensure_dir, save_dataframe, save_json
from ..utils.logging import get_logger

_log = get_logger(__name__)


def build_attack(cfg: dict[str, Any]) -> MessageAttack:
    a = cfg["attack"]
    attack_type = a.get("type", a.get("mode", "adaptive"))
    if attack_type == "bpda":
        from ..attacks.bpda import BPDAMessageAttack
        return BPDAMessageAttack(
            epsilon=a["epsilon"],
            steps=a["steps"],
            step_size=a["step_size"],
            norm=a["norm"],
            colluding=bool(a.get("colluding", False)),
        )
    elif attack_type == "spsa":
        from ..attacks.spsa import SPSAMessageAttack
        return SPSAMessageAttack(
            epsilon=a["epsilon"],
            steps=a["steps"],
            step_size=a["step_size"],
            norm=a["norm"],
            colluding=bool(a.get("colluding", False)),
        )
    return FGSMMessageAttack(
        epsilon=a["epsilon"],
        steps=a["steps"],
        step_size=a["step_size"],
        norm=a["norm"],
        mode="adaptive" if a.get("mode") == "adaptive" else "static",
        colluding=bool(a.get("colluding", False)),
    )


def _episode_returns(env, agg, attack, policy, byz, seed, episodes):
    returns = []
    for ep in range(episodes):
        ep_rng = np.random.default_rng(int(seed) * 10000 + ep)
        returns.append(env.rollout(agg, attack, policy, byz, ep_rng)["return"])
    return np.array(returns)


def run_evaluation(cfg: dict[str, Any], rng: np.random.Generator, out_dir: str | None = None) -> pd.DataFrame:
    env = make_environment(cfg, rng)
    n = env.n
    fractions = list(cfg["eval"]["fractions"])
    episodes = int(cfg["eval"]["episodes"])
    attack_enabled = bool(cfg["attack"]["enabled"])
    seed = int(cfg.get("seed", 42))

    specs = standard_baselines(cfg)
    train_f = int(cfg["attack"]["f"])
    train_byz = list(range(train_f))
    train_attack = build_attack(cfg) if attack_enabled else None

    records: list[EvalRecord] = []
    for spec in specs:
        try:
            agg, policy = build_defense(spec, env, cfg, train_attack, train_byz, rng)
        except ValueError as exc:  # e.g. Krum regime invalid for this topology
            _log.warning("skipping %s: %s", spec.name, exc)
            continue

        for phi in fractions:
            f_eval = int(round(phi * n))
            byz = list(range(f_eval))
            attack = build_attack(cfg) if (attack_enabled and f_eval > 0) else None
            clean = _episode_returns(env, agg, None, policy, byz, seed, episodes)
            adv = _episode_returns(env, agg, attack, policy, byz, seed, episodes)
            dev = float(
                np.mean([env.rollout(agg, attack, policy, byz, np.random.default_rng(seed * 5000 + ep))["mean_deviation"]
                         for ep in range(max(1, episodes // 5))])
            )
            asr = attack_success_rate(clean.tolist(), adv.tolist(), drop_threshold=0.1)
            records.append(
                EvalRecord(
                    operator=spec.name,
                    fraction=float(phi),
                    clean_reward=float(clean.mean()),
                    adv_reward=float(adv.mean()),
                    attack_success=float(asr) if not np.isnan(asr) else 0.0,
                    aggregate_deviation=dev,
                    n_agents=n,
                    msg_dim=env.d,
                    topology=cfg["env"]["topology"],
                    mode=cfg["attack"]["mode"],
                    colluding=bool(cfg["attack"]["colluding"]),
                    seed=int(cfg["seed"]),
                )
            )
            _log.info("%-18s phi=%.2f clean=%.3f adv=%.3f", spec.name, phi,
                      clean.mean(), adv.mean())

    df = pd.DataFrame([asdict(r) for r in records])
    if out_dir is not None and not df.empty:
        ensure_dir(out_dir)
        save_dataframe(df, f"{out_dir}/evaluation.csv")
        save_json(
            {
                "note": "synthetic smoke-test output, not a paper result",
                "methods": sorted(df["operator"].unique().tolist()),
                "fractions": fractions,
            },
            f"{out_dir}/evaluation_summary.json",
        )
    return df
