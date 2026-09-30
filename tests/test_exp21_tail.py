"""Tests for the EXP_0021 tail-repair pieces.

Covers: (1) no-progress stall truncation semantics in the vector and single
envs (cut at K, truncated-not-terminated, counters reset, success
semantics untouched); (2) adaptive_edge_gat padding equivalence now that the
edge layers mask padding slots; (3) the balanced training shell aggregates
stall bookkeeping; (4) the rule never fires while mass is being removed.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from environments.exp21_tail_env import (
    TailTruncationSingle, TailTruncationVector, make_training_env,
)
from marl.mappo_advanced import MAPPOAdvanced

ENV_KW = dict(scenario='straight', scenario_pool=['straight'],
              randomize_scenario=False, num_robots=8, active_robots=5,
              num_clots=3, horizon=300, robot_radius=.0011,
              obs_mode='geometric_predictive', contact_mode='geodesic',
              dynamic_intravascular_particles=True, particle_count=16,
              particle_seed=82)


def _vector(stall_limit):
    return TailTruncationVector(n_envs=2, seed=1, stall_limit=stall_limit, **ENV_KW)


def test_stall_cut_is_truncation_not_termination():
    env = _vector(10)
    env._observe()
    zero = np.zeros((2, 8, 3), np.float32)
    cut = None
    for t in range(20):
        obs, r, term, trunc, info = env.step(zero)
        if np.any(trunc):
            cut = t + 1
            break
        assert not np.any(term)
    assert cut == 10, f"expected cut at step 10, got {cut}"
    assert info['stall_truncated'].any()
    assert not np.asarray(term).any()


def test_stall_counter_resets_and_steps_restart():
    env = _vector(10)
    env._observe()
    zero = np.zeros((2, 8, 3), np.float32)
    for _ in range(9):
        obs, r, term, trunc, info = env.step(zero)
        assert not np.any(trunc)
    # The 10th zero-progress step is the cut itself.
    obs, r, term, trunc, info = env.step(zero)
    assert np.asarray(trunc).all()
    assert (env.steps == 0).all(), "steps must restart after the cut"
    assert (env._stall == 0).all(), "stall counter must clear on reset"


def test_no_cut_while_mass_decreases():
    # Mass decrease resets the counter even when the decrease is tiny.
    env = _vector(5)
    env._observe()
    zero = np.zeros((2, 8, 3), np.float32)
    for t in range(20):
        # Inject a small mass decrease each step before stepping.
        env.clot_masses[:] = np.maximum(
            env.clot_masses - 1e-3, 0.0)
        env._prev_total_mass[:] = env.clot_masses.sum(axis=1) + 1.0
        obs, r, term, trunc, info = env.step(zero)
        # the wrapper compares against the pre-step total: with the injected
        # decrease total < prev -> counter stays 0 -> no cut.
        assert not np.any(trunc), f"cut fired at step {t} despite progress"
    assert (env._stall == 0).all()


def test_single_env_cut_and_reset():
    env = TailTruncationSingle(scenario='straight', scenario_pool=['straight'],
                               randomize_scenario=False, seed=1, num_robots=5,
                               num_clots=3, horizon=300, robot_radius=.0011,
                               obs_mode='geometric_predictive', contact_mode='geodesic',
                               dynamic_intravascular_particles=True, particle_count=24,
                               particle_seed=82, stall_limit=10)
    env.reset(seed=1)
    zero = np.zeros((5, 3), np.float32)
    cut = None
    for t in range(20):
        obs, r, term, trunc, info = env.step(zero)
        if trunc:
            cut = t + 1
            break
    assert cut == 10
    assert info['stall_truncated']
    assert not term
    # After reset the counter clears.
    env.reset(seed=2)
    assert env._stall == 0 and not env.stall_truncated


def test_training_shell_aggregates_stall_info():
    env = make_training_env(56, 42, None, 'cpu', stall_limit=10)
    act = np.zeros((56, 8, 3), np.float32)
    for t in range(12):
        obs, r, term, trunc, info = env.step(act)
        assert info['stall_steps'].shape == (56,)
        if np.any(trunc):
            assert int(np.asarray(trunc).sum()) > 0
            assert (env.envs[0].steps == 0).all()
            break
    else:
        raise AssertionError("shell never cut with stall_limit=10")


def test_adaptive_edge_gat_padding_equivalence():
    torch.manual_seed(0)
    agent = MAPPOAdvanced(n_agents=10, obs_dim=52, action_dim=3, state_dim=24,
                          hidden_dim=32, num_layers=2, architecture='adaptive_edge_gat',
                          control_mode='local', critic_value_mode='v', device='cpu',
                          max_agents=10)
    real_n = 5
    obs = torch.randn(2, real_n, 52)
    ctx = {"adjacency": torch.ones(2, real_n, real_n),
           "positions": torch.rand(2, real_n, 3),
           "velocities": torch.zeros(2, real_n, 3)}
    ref_actions, _, _ = agent.actor.get_actions(obs, ctx, deterministic=True)
    obs_pad = torch.zeros(2, 10, 52)
    obs_pad[:, :real_n] = obs
    ctx_pad = {"adjacency": torch.zeros(2, 10, 10),
               "positions": torch.zeros(2, 10, 3),
               "velocities": torch.zeros(2, 10, 3)}
    ctx_pad["adjacency"][:, :real_n, :real_n] = ctx["adjacency"]
    ctx_pad["positions"][:, :real_n] = ctx["positions"]
    ctx_pad["agent_mask"] = torch.zeros(2, 10, dtype=torch.bool)
    ctx_pad["agent_mask"][:, :real_n] = True
    pad_actions, _, _ = agent.actor.get_actions(obs_pad, ctx_pad, deterministic=True)
    assert (pad_actions[:, :real_n] - ref_actions).abs().max() < 1e-5


def test_worker_eval_applies_frenet_transform():
    """The eval loop must not feed raw Frenet proposals to env.step.

    Regression for the EXP_0021 first-evaluation bug: the worker's evaluate()
    passed the actor output straight to ``env.step``, which reads actions as
    world-frame velocity commands. Training and the v2 evaluations always
    apply ``bounded`` + ``direct_local_action``; the eval loop source must do
    the same.
    """
    import inspect

    from scripts import exp21_worker

    src = inspect.getsource(exp21_worker.evaluate)
    calls = [l.strip() for l in src.splitlines() if 'env.step(' in l]
    assert calls, "evaluate() must call env.step"
    for call in calls:
        assert 'direct_local_action(' in call, (
            f"evaluate() env.step call bypasses the Frenet transform: {call}")
