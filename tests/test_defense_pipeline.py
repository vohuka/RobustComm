"""End-to-end defense pipeline: build_defense, training, and the central claim."""
import numpy as np

from robustcomm.baselines.base import BaselineSpec, build_defense, standard_baselines
from robustcomm.data.adapters import make_environment
from robustcomm.experiments.run_evaluation import build_attack

CFG = {
    "seed": 0,
    "env": {"name": "synthetic_comm", "n_agents": 9, "msg_dim": 4, "topology": "broadcast",
            "neighbors": 8, "horizon": 5, "honest_dispersion": 0.5, "discrete": False,
            "codebook_levels": 5, "data_root": None},
    "aggregator": {"operator": "krum", "f": 2, "beta": 0.2},
    "attack": {"enabled": True, "f": 2, "epsilon": 2.0, "norm": "linf", "steps": 3,
               "step_size": 1.0, "mode": "adaptive", "colluding": False, "white_box": False},
    "training": {"enabled": True, "episodes": 5, "ensemble_size": 2, "lr": 0.05},
    "eval": {"episodes": 8, "fractions": [0.0, 0.22], "target_reward_tau": 0.5,
             "bootstrap_samples": 100},
    "theory": {"L_pi": 1.0, "L_r": 1.0, "gamma": 0.99},
    "output": {"dir": "outputs/test"},
}


def _mean_return(env, agg, attack, policy, byz, rng, k=8):
    return float(np.mean([env.rollout(agg, attack, policy, byz, rng)["return"] for _ in range(k)]))


def test_standard_baselines_present():
    names = {s.name for s in standard_baselines(CFG)}
    assert {"no_defense", "no_comm", "krum_only", "adv_only", "robustcomm_krum"} <= names


def test_build_defense_returns_aggregator_and_policy():
    rng = np.random.default_rng(0)
    env = make_environment(CFG, rng)
    spec = BaselineSpec("krum_only", "krum", 2, 0.2, False, False, "")
    agg, policy = build_defense(spec, env, CFG, None, [], rng)
    assert agg.name == "krum"
    assert hasattr(policy, "act")


def test_robust_holds_better_than_mean_under_attack():
    """Central claim (Prop 9.1 / Thm 9.4) on synthetic data: robust aggregation retains
    more reward under attack than the mean."""
    rng = np.random.default_rng(0)
    env = make_environment(CFG, rng)
    byz = [0, 1]
    attack = build_attack(CFG)
    mean_spec = BaselineSpec("mean", "mean", 0, 0.2, False, False, "")
    krum_spec = BaselineSpec("krum", "krum", 2, 0.2, False, False, "")
    mean_agg, mean_pol = build_defense(mean_spec, env, CFG, None, [], rng)
    krum_agg, krum_pol = build_defense(krum_spec, env, CFG, None, [], rng)
    mean_adv = _mean_return(env, mean_agg, attack, mean_pol, byz, np.random.default_rng(7))
    krum_adv = _mean_return(env, krum_agg, attack, krum_pol, byz, np.random.default_rng(7))
    assert krum_adv > mean_adv


def test_adversarial_training_runs_and_returns_history():
    rng = np.random.default_rng(0)
    env = make_environment(CFG, rng)
    spec = BaselineSpec("robustcomm", "median", 2, 0.2, True, True, "")
    agg, policy = build_defense(spec, env, CFG, build_attack(CFG), [0, 1], rng)
    assert hasattr(policy, "members") and len(policy.members) == 2
