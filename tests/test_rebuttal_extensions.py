"""Tests for rebuttal extensions: PredatorPreyEnv, BoundaryPushingAttack, CamouflageAttack, LipschitzEstimator."""
import numpy as np
import pytest

from robustcomm.aggregators import MedianAggregator, MeanAggregator
from robustcomm.attacks.boundary_pushing import BoundaryPushingAttack
from robustcomm.attacks.camouflage import CamouflageAttack
from robustcomm.envs.predator_prey import PredatorPreyEnv, PredatorPreyPolicy
from robustcomm.theory.lipschitz_estimator import LipschitzEstimator


def test_predator_prey_env_creation_and_rollout():
    rng = np.random.default_rng(42)
    env = PredatorPreyEnv(n_agents=6, n_preys=2, msg_dim=4, horizon=10)
    assert env.n == 6
    assert env.d == 4

    agg = MedianAggregator()
    pol = PredatorPreyPolicy(msg_dim=4, rng=rng)
    res = env.rollout(agg, None, pol, byzantine_ids=[], rng=rng)
    assert "return" in res
    assert 0.0 <= res["return"] <= 1.0
    assert "capture_ratio" in res
    assert "mean_deviation" in res


def test_boundary_pushing_attack():
    rng = np.random.default_rng(123)
    env = PredatorPreyEnv(n_agents=6, n_preys=2, msg_dim=4, horizon=5)
    attack = BoundaryPushingAttack(epsilon=1.0, steps=5)
    pol = PredatorPreyPolicy(msg_dim=4, rng=rng)

    mean_agg = MeanAggregator()
    median_agg = MedianAggregator()

    res_mean = env.rollout(mean_agg, attack, pol, byzantine_ids=[0, 1], rng=rng)
    res_med = env.rollout(median_agg, attack, pol, byzantine_ids=[0, 1], rng=rng)

    # Median aggregator should sustain higher or equal return compared to undefended mean
    assert res_med["return"] >= res_mean["return"] - 0.10


def test_camouflage_attack():
    rng = np.random.default_rng(456)
    messages = rng.normal(0, 0.5, size=(7, 4))
    byz_mask = np.array([True, True, False, False, False, False, False])
    attack = CamouflageAttack(epsilon=1.0, dispersion_bound=0.5)

    crafted = attack.craft(messages, byz_mask, MedianAggregator(), rng)
    assert crafted.shape == messages.shape
    # Check that crafted messages remain finite and reasonable
    assert np.all(np.isfinite(crafted))


def test_lipschitz_estimator():
    estimator = LipschitzEstimator(msg_dim=4, n_agents=7, gamma=0.99, dt=0.1, damping=0.9)
    pol = PredatorPreyPolicy(msg_dim=4)

    L_pi = estimator.estimate_policy_lipschitz(pol, n_samples=500)
    L_r = estimator.estimate_reward_lipschitz(n_samples=500)
    L_P = estimator.estimate_transition_lipschitz()

    assert L_pi > 0.0
    assert L_r > 0.0
    assert L_P > 0.0

    info = estimator.compute_theoretical_bound(L_pi=L_pi, L_r=L_r, L_P=L_P, B=0.42, d=4, f=3, n=7)
    assert "theoretical_bound" in info
    assert info["theoretical_bound"] > 0.0
    assert info["L_Q"] > 0.0
