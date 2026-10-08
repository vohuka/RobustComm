"""Perturbation set and FGSM surrogate attack (Def 5.1, Alg 2)."""
import numpy as np

from robustcomm.aggregators import build_aggregator
from robustcomm.attacks import FGSMMessageAttack, PerturbationSet


def test_perturbation_projection_linf():
    ps = PerturbationSet(epsilon=1.0, norm="linf")
    out = ps.project(np.array([2.0, -3.0, 0.5]))
    assert np.allclose(out, [1.0, -1.0, 0.5])


def test_perturbation_projection_l2():
    ps = PerturbationSet(epsilon=1.0, norm="l2")
    out = ps.project(np.array([3.0, 4.0]))  # norm 5 -> scaled to 1
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_attack_only_perturbs_byzantine_and_within_budget():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(7, 4))
    mask = np.array([True, True, False, False, False, False, False])
    agg = build_aggregator("krum", f=2)
    atk = FGSMMessageAttack(epsilon=1.0, steps=3, step_size=0.5, norm="linf", mode="adaptive")
    out = atk.craft(M, mask, agg, rng)
    # honest rows unchanged
    assert np.allclose(out[~mask], M[~mask])
    # Byzantine perturbations within the linf budget
    delta = out[mask] - M[mask]
    assert np.all(np.abs(delta) <= 1.0 + 1e-9)


def test_attack_degrades_mean_more_than_krum():
    rng = np.random.default_rng(1)
    s = np.zeros(4)
    dirs = rng.normal(size=(9, 4))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    M = s + dirs * 0.3
    mask = np.zeros(9, dtype=bool)
    mask[:2] = True
    atk = FGSMMessageAttack(epsilon=3.0, steps=5, step_size=1.0, mode="static")
    mean_agg = build_aggregator("mean")
    krum_agg = build_aggregator("krum", f=2)
    M_att = atk.craft(M, mask, mean_agg, rng)
    mean_dev = np.linalg.norm(mean_agg.aggregate(M_att) - s)
    krum_dev = np.linalg.norm(krum_agg.aggregate(M_att) - s)
    assert mean_dev > krum_dev


def test_colluding_shares_direction():
    rng = np.random.default_rng(2)
    M = rng.normal(size=(7, 3))
    mask = np.array([True, True, True, False, False, False, False])
    agg = build_aggregator("median", f=3)
    atk = FGSMMessageAttack(epsilon=2.0, steps=2, step_size=1.0, mode="static", colluding=True)
    out = atk.craft(M, mask, agg, rng)
    deltas = out[mask] - M[mask]
    # colluding perturbations share the same sign pattern (coordinated direction)
    signs = np.sign(deltas)
    assert np.all(signs == signs[0])
