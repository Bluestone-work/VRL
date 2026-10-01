"""Own route bearing in a fixed observation position (schema mca_point_routed_115_own_v9).

The vector is the same route bearing already written in the robot's assigned
target slot; only its position changes. The masked arm writes zeros there, so
with zero-initialized new input weights it reproduces the v8 policy exactly.
"""
from dataclasses import replace

import numpy as np
import torch

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig, TARGET_SLOT_DIMS
from marl.mca_physical_policy import (make_physical_agent, initialize_expanded_obstacle_policy,
                                      physical_policy_action)
from scripts.train_mca_physical import reset_with_valid_particles

CONFIG = 'configs/experiments/EXP_0039_ASSIGNED_DYNAMICS.json'
SEED = 940000000


def configs():
    base = DynamicsConfig.from_json(CONFIG)
    return base, replace(base, target_observation='routed_assigned_own'), replace(base, target_observation='routed_assigned_own_masked')


def test_schema_appends_three_columns_after_the_v8_block():
    base, own, masked = configs()
    assert CompiledMCAPhysicalEnv(base).obs_dim == 112
    for cfg in (own, masked):
        env = CompiledMCAPhysicalEnv(cfg)
        assert env.obs_dim == 115 and env.observation_schema == 'mca_point_routed_115_own_v9'


def test_prefix_is_identical_and_own_columns_copy_the_assigned_slot():
    base, own, masked = configs()
    nodes = {}
    for name, cfg in (('base', base), ('own', own), ('masked', masked)):
        env = CompiledMCAPhysicalEnv(cfg)
        obs, _ = reset_with_valid_particles(env, SEED)
        rng = np.random.default_rng(1)
        for _ in range(10):
            obs, *_ = env.step(rng.uniform(-1, 1, (cfg.num_robots, 3)))
        nodes[name] = obs['nodes']; assignment = env._assigned_targets()
    assert np.array_equal(nodes['own'][:, :112], nodes['base'])
    assert np.array_equal(nodes['masked'][:, :112], nodes['base'])
    assert not nodes['masked'][:, 112:].any()
    for i, j in enumerate(assignment):
        slot = 76+TARGET_SLOT_DIMS*j
        assert np.allclose(nodes['own'][i, 112:115], nodes['own'][i, slot:slot+3])
        assert np.linalg.norm(nodes['own'][i, 112:115]) > .99


def test_cleared_own_target_gives_zero_bearing_until_reassigned():
    _, own, _ = configs()
    env = CompiledMCAPhysicalEnv(own)
    reset_with_valid_particles(env, SEED)
    env.masses[:] = 0.
    assert not env._observation()['nodes'][:, 112:].any()


def test_expanded_masked_policy_acts_exactly_like_its_v8_parent():
    base, _, masked = configs()
    parent_env = CompiledMCAPhysicalEnv(base)
    obs_p, _ = reset_with_valid_particles(parent_env, SEED)
    parent = make_physical_agent(parent_env, seed=42, hidden_dim=128, device='cpu')
    payload = dict(meta=dict(parent.meta), actor=parent.actor.state_dict(), critic=parent.critic.state_dict())
    env = CompiledMCAPhysicalEnv(masked)
    obs_c, _ = reset_with_valid_particles(env, SEED)
    child = make_physical_agent(env, seed=7, hidden_dim=128, device='cpu')
    expanded = initialize_expanded_obstacle_policy(child, payload)
    assert expanded == ['actor.encoder.input_proj.weight', 'critic.inner.obs_encoder.input_proj.weight']
    for _ in range(5):
        with torch.no_grad():
            a = physical_policy_action(parent, parent_env, obs_p, deterministic=True)
            b = physical_policy_action(child, env, obs_c, deterministic=True)
        assert np.allclose(a[0], b[0], atol=1e-6) and np.allclose(a[2], b[2], atol=1e-6)
        obs_p, *_ = parent_env.step(a[3]); obs_c, *_ = env.step(b[3])
