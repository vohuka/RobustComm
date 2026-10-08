"""Baseline / method specifications and a builder (Sec. 10.5).

A baseline is a *defense configuration*: an aggregation operator plus whether adversarial
training and ensembles are used. ``build_defense`` constructs the aggregator and (if
required) trains a policy ensemble via the minimax loop, returning ``(aggregator, policy)``
ready for evaluation. This keeps every baseline on equal footing under the same runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..aggregators import build_aggregator
from ..aggregators.base import RobustAggregator
from ..attacks.base import MessageAttack
from ..envs.synthetic_comm import IdentityPolicy, SyntheticCommEnv
from ..policy.ensembles import PolicyEnsemble
from ..training.adversarial import adversarial_train
from .no_defense import NoCommAggregator


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    operator: str  # mean | krum | median | trmean | nocomm
    f: int
    beta: float
    adversarial_training: bool
    ensembles: bool
    description: str


def build_defense(
    spec: BaselineSpec,
    env: Any,
    cfg: dict[str, Any],
    attack: MessageAttack | None,
    byzantine_ids: list[int],
    rng: np.random.Generator,
) -> tuple[RobustAggregator, Any]:
    """Construct (and optionally adversarially train) a defense from a spec."""
    from .sota_defenses import SunMessageAblationDefense, RMACRLReconstructionDefense

    if spec.operator == "nocomm":
        aggregator: RobustAggregator = NoCommAggregator(env.d)
    elif spec.operator == "sun_ensemble":
        aggregator = SunMessageAblationDefense(env.d, f=spec.f, num_samples=20)
    elif spec.operator == "rmacrl":
        aggregator = RMACRLReconstructionDefense(env.d, residual_threshold=0.8)
    else:
        aggregator = build_aggregator(spec.operator, f=spec.f, beta=spec.beta)

    if spec.adversarial_training:
        k = cfg["training"]["ensemble_size"] if spec.ensembles else 1
        ensemble = PolicyEnsemble(env.d, ensemble_size=k, rng=rng)
        adversarial_train(
            env,
            aggregator,
            ensemble,
            attack,
            byzantine_ids,
            episodes=cfg["training"]["episodes"],
            lr=cfg["training"]["lr"],
            rng=rng,
        )
        return aggregator, ensemble
    return aggregator, IdentityPolicy()


def standard_baselines(cfg: dict[str, Any]) -> list[BaselineSpec]:
    """The paper's baselines B1-B4 plus SOTA defenses and the full RobustComm method (Sec. 10.5)."""
    f = int(cfg["attack"]["f"])
    beta = float(cfg["aggregator"]["beta"])
    return [
        BaselineSpec("no_defense", "mean", 0, beta, False, False,
                     "mean aggregation (DIAL), no defense (B1)"),
        BaselineSpec("no_comm", "nocomm", 0, beta, False, False,
                     "no communication, lower reference (B2)"),
        BaselineSpec("krum_only", "krum", f, beta, False, False,
                     "Krum robust aggregation, no adversarial training (B3)"),
        BaselineSpec("median_only", "median", f, beta, False, False,
                     "Coordinate Median alone, no adversarial training"),
        BaselineSpec("adv_only", "mean", 0, beta, True, True,
                     "mean aggregation + adversarial training (B4)"),
        BaselineSpec("sun_ensemble", "sun_ensemble", f, beta, False, False,
                     "Message ablation ensemble defense (Sun et al., ICLR 2023)"),
        BaselineSpec("rmacrl", "rmacrl", f, beta, False, False,
                     "Message reconstruction residual filter (Xue et al., AAMAS 2022)"),
        BaselineSpec("robustcomm_krum", "krum", f, beta, True, True,
                     "RobustComm: Krum + adversarial training + ensembles"),
        BaselineSpec("robustcomm_median", "median", f, beta, True, True,
                     "RobustComm: coordinate-wise median + adversarial training + ensembles"),
    ]
