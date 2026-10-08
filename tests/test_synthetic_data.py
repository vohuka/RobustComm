"""Schemas, topologies, synthetic generator, and environment."""
import numpy as np
import pytest

from robustcomm.data import build_topology, generate_message_rounds
from robustcomm.data.schemas import CommRound
from robustcomm.envs import IdentityPolicy, SyntheticCommEnv


def test_topologies_have_correct_structure():
    n = 6
    for name in ["broadcast", "ring", "star", "neighborhood"]:
        topo = build_topology(name, n, neighbors=3, rng=np.random.default_rng(0))
        assert topo.n_agents == n
        assert set(topo.inbound) == set(range(n))
        for i, nbr in topo.inbound.items():
            assert i not in nbr  # inbound excludes self
    assert build_topology("broadcast", n).neighbourhood_size(0) == n - 1


def test_bad_topology_raises():
    with pytest.raises(ValueError):
        build_topology("mesh", 5)


def test_commround_validation():
    r = CommRound(messages=np.zeros((4, 2)), byzantine=np.array([1, 0, 0, 0]))
    assert r.n_senders == 4 and r.msg_dim == 2 and r.n_byzantine == 1
    assert r.honest_messages().shape == (3, 2)
    with pytest.raises(ValueError):
        CommRound(messages=np.zeros((4, 2)), byzantine=np.array([1, 0]))


def test_synthetic_generator_schema():
    df = generate_message_rounds(n_rounds=3, n_senders=5, msg_dim=2, n_byzantine=1, seed=0)
    assert len(df) == 15
    for col in ["round", "sender", "byzantine", "target_0", "message_0", "message_1"]:
        assert col in df.columns
    assert df["byzantine"].sum() == 3  # 1 per round * 3 rounds


def test_env_rollout_returns_valid_reward():
    topo = build_topology("broadcast", 7)
    env = SyntheticCommEnv(7, 4, topo, honest_dispersion=0.5, horizon=5)
    from robustcomm.aggregators import build_aggregator

    out = env.rollout(build_aggregator("median", f=2), None, IdentityPolicy(), [0, 1],
                      np.random.default_rng(0))
    assert 0.0 <= out["return"] <= 1.0
    assert out["mean_deviation"] >= 0.0


def test_env_krum_incompatible_topology_raises():
    # star: peripheral neighbourhood is tiny -> Krum regime fails.
    topo = build_topology("star", 7)
    env = SyntheticCommEnv(7, 4, topo, horizon=3)
    from robustcomm.aggregators import build_aggregator

    with pytest.raises(ValueError):
        env.rollout(build_aggregator("krum", f=2), None, IdentityPolicy(), [1],
                    np.random.default_rng(0))
