"""Determinism under fixed seeds."""
import numpy as np

from robustcomm.aggregators import build_aggregator
from robustcomm.attacks import FGSMMessageAttack
from robustcomm.data.adapters import make_environment
from robustcomm.envs import IdentityPolicy
from robustcomm.policy.ensembles import PolicyEnsemble
from robustcomm.training.adversarial import adversarial_train
from tests.test_defense_pipeline import CFG


def test_env_rollout_deterministic_under_fixed_seed():
    env = make_environment(CFG, np.random.default_rng(0))
    agg = build_aggregator("median", f=2)
    r1 = env.rollout(agg, None, IdentityPolicy(), [0, 1], np.random.default_rng(123))
    r2 = env.rollout(agg, None, IdentityPolicy(), [0, 1], np.random.default_rng(123))
    assert r1 == r2


def test_adversarial_training_deterministic():
    env = make_environment(CFG, np.random.default_rng(0))
    agg = build_aggregator("krum", f=2)
    atk = FGSMMessageAttack(epsilon=1.0, steps=2, step_size=0.5, mode="adaptive")

    def train():
        ens = PolicyEnsemble(env.d, ensemble_size=2, rng=np.random.default_rng(5))
        adversarial_train(env, agg, ens, atk, [0, 1], episodes=4, lr=0.05,
                          rng=np.random.default_rng(9))
        return np.concatenate([m.W.ravel() for m in ens.members])

    assert np.allclose(train(), train())


def test_attack_deterministic():
    rng_seed = 11
    M = np.random.default_rng(0).normal(size=(9, 4))
    mask = np.array([True, True] + [False] * 7)
    agg = build_aggregator("krum", f=2)
    atk = FGSMMessageAttack(epsilon=1.0, steps=3, step_size=0.5, mode="adaptive")
    o1 = atk.craft(M, mask, agg, np.random.default_rng(rng_seed))
    o2 = atk.craft(M, mask, agg, np.random.default_rng(rng_seed))
    assert np.allclose(o1, o2)
