"""Minimax adversarial communication training (Alg 3) on the synthetic environment.

Alternates an inner attacker (crafts message perturbations against the current messages
and the deployed aggregator) with an outer learner (updates a sampled ensemble member to
recover the signal from the possibly-corrupted aggregate). This is the synthetic
instantiation of Algorithm 3; it is a heuristic for the minimax objective (Eq. minimax)
with no convergence guarantee (Remark 9.8), exactly as the paper states.

NOTE: outputs from this loop are synthetic smoke-test outputs, not paper results.
"""
from __future__ import annotations

import numpy as np

from ..aggregators.base import RobustAggregator
from ..attacks.base import MessageAttack
from ..envs.synthetic_comm import SyntheticCommEnv
from ..policy.ensembles import PolicyEnsemble
from ..utils.logging import get_logger

_log = get_logger(__name__)


def adversarial_train(
    env: SyntheticCommEnv,
    aggregator: RobustAggregator,
    ensemble: PolicyEnsemble,
    attack: MessageAttack | None,
    byzantine_ids: list[int],
    episodes: int,
    lr: float,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    """Train ``ensemble`` in place; return the per-episode training-return trace."""
    env.check_operator_compatible(aggregator)
    byz = set(int(b) for b in byzantine_ids)
    byz_mask = np.array([i in byz for i in range(env.n)], dtype=bool)
    honest_ids = [i for i in range(env.n) if i not in byz]

    history: list[float] = []
    for ep in range(episodes):
        member = ensemble.sample_member(rng)  # MADDPG-style: one sub-policy per episode
        ep_rewards: list[float] = []
        for _ in range(env.horizon):
            s = env.sample_target(rng)
            M = env.honest_messages(s, rng)
            if attack is not None and byz_mask.any():  # inner loop (attack)
                M = attack.craft(M, byz_mask, aggregator, rng)
            grad_W = np.zeros_like(member.W)
            count = 0
            for i in honest_ids:  # outer loop (defense update)
                m_hat = aggregator.aggregate(M[env.agg_neighbors[i]])
                grad_W += member.mse_grad(m_hat, s)
                ep_rewards.append(env._reward(member.act(m_hat), s))
                count += 1
            if count:
                member.apply_grad(grad_W / count, lr)
        history.append(float(np.mean(ep_rewards)) if ep_rewards else 0.0)
        if (ep + 1) % max(1, episodes // 5) == 0:
            _log.info("adversarial_train: episode %d/%d, train-return %.3f",
                      ep + 1, episodes, history[-1])
    return {"train_return": history}
