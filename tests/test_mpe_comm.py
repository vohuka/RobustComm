import numpy as np
import pytest

from robustcomm.aggregators import MedianAggregator, KrumAggregator, MeanAggregator
from robustcomm.attacks.fgsm import FGSMMessageAttack
from robustcomm.data.adapters import make_environment
from robustcomm.envs.mpe_comm import MPECommEnv, MPEParticlePolicy


def test_mpe_comm_env_creation_and_rollout():
    rng = np.random.default_rng(42)
    env = MPECommEnv(n_agents=5, msg_dim=4, horizon=5)
    assert env.n == 5
    assert env.d == 4
    assert env.horizon == 5

    agg = MedianAggregator()
    pol = MPEParticlePolicy(msg_dim=4, rng=rng)
    res = env.rollout(agg, None, pol, byzantine_ids=[], rng=rng)
    assert "return" in res
    assert 0.0 < res["return"] <= 1.0
    assert "mean_deviation" in res


def test_mpe_comm_under_fgsm_attack():
    rng = np.random.default_rng(123)
    env = MPECommEnv(n_agents=5, msg_dim=4, horizon=5)
    attack = FGSMMessageAttack(epsilon=0.5, steps=3, step_size=0.1, colluding=True)
    pol = MPEParticlePolicy(msg_dim=4, rng=rng)

    # Undefended mean aggregator under attack
    mean_agg = MeanAggregator()
    res_undef = env.rollout(mean_agg, attack, pol, byzantine_ids=[0, 1], rng=rng)

    # Defended median aggregator under attack
    median_agg = MedianAggregator()
    res_def = env.rollout(median_agg, attack, pol, byzantine_ids=[0, 1], rng=rng)

    assert res_def["return"] >= res_undef["return"] - 0.15


def test_mpe_adapter_via_make_environment():
    rng = np.random.default_rng(7)
    cfg = {
        "env": {
            "name": "mpe",
            "n_agents": 5,
            "msg_dim": 4,
            "topology": "broadcast",
            "horizon": 5,
        }
    }
    env = make_environment(cfg, rng)
    assert env.n == 5
    assert env.d == 4
    agg = MedianAggregator()
    pol = MPEParticlePolicy(msg_dim=4, rng=rng)
    res = env.rollout(agg, None, pol, byzantine_ids=[], rng=rng)
    assert "return" in res
