"""Routed/assigned target observations, sticky allocation and monotone potential.

The allocation and the route bearing are observation features only: no waypoint,
velocity or residual command is derived from them, so the action path stays the
pure-RL Frenet transform. The diagnostic path follower is used here strictly as
a reference bearing in assertions, never inside the environment.
"""
from dataclasses import replace

import numpy as np
import pytest

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import (DynamicsConfig, TARGET_SLOT_DIMS, TARGET_SUMMARY_DIMS)
from marl.mca_physical_policy import make_physical_agent, initialize_expanded_obstacle_policy, physical_context
from scripts.train_mca_physical import reset_with_valid_particles
from scripts.validate_mca_surface_task import witness_action

CONFIG = 'configs/experiments/EXP_0029_MCA_ALL_RANDOM_DYNAMICS.json'
SEED = 940000000


def configs():
    base = DynamicsConfig.from_json(CONFIG)
    routed = replace(base, target_observation='routed_assigned', progress_potential='mass_weighted')
    masked = replace(base, target_observation='routed_assigned_masked', progress_potential='nearest')
    return base, routed, masked


def rollout(cfg, seed, steps=40):
    env = CompiledMCAPhysicalEnv(cfg)
    obs, _ = reset_with_valid_particles(env, seed)
    rng = np.random.default_rng(11)
    nodes, rewards = [obs['nodes'].copy()], []
    for _ in range(steps):
        obs, reward, terminated, truncated, _ = env.step(rng.uniform(-1, 1, (cfg.num_robots, 3)))
        nodes.append(obs['nodes'].copy()); rewards.append(reward)
        if terminated or truncated:
            break
    return np.stack(nodes), np.array(rewards)


def test_schema_and_dimensions_are_explicit_and_appended():
    base, routed, masked = configs()
    assert CompiledMCAPhysicalEnv(base).obs_dim == 76
    assert CompiledMCAPhysicalEnv(base).target_block == 0
    for cfg in (routed, masked):
        env = CompiledMCAPhysicalEnv(cfg)
        assert env.target_block == TARGET_SLOT_DIMS*env.num_clots+TARGET_SUMMARY_DIMS == 36
        assert env.obs_dim == 112
        assert env.observation_schema == 'mca_point_routed_112_v8'


def test_defaults_and_invalid_values_are_rejected():
    assert DynamicsConfig.from_json(CONFIG).target_observation == 'nearest_euclidean'
    assert DynamicsConfig.from_json(CONFIG).progress_potential == 'nearest'
    base = DynamicsConfig.from_json(CONFIG)
    for field, value in (('target_observation', 'routed'), ('progress_potential', 'monotone')):
        with pytest.raises(ValueError):
            replace(base, **{field: value})
    with pytest.raises(ValueError):
        CompiledMCAPhysicalEnv(replace(base, contact_model='stenosis_surface',
                                       target_observation='routed_assigned'))


def test_masked_arm_is_bit_identical_to_the_unrouted_baseline():
    base, _, masked = configs()
    reference, base_rewards = rollout(base, SEED)
    appended, masked_rewards = rollout(masked, SEED)
    assert appended.shape[-1] == 112 and reference.shape[-1] == 76
    assert np.array_equal(appended[..., :76], reference)
    assert np.array_equal(masked_rewards, base_rewards)
    assert not appended[..., 76:].any()


def test_routed_block_only_touches_appended_columns():
    base, routed, _ = configs()
    reference, base_rewards = rollout(base, SEED)
    features, routed_rewards = rollout(routed, SEED)
    assert np.array_equal(features[..., :76], reference)
    assert features[..., 76:].any()
    # The potential changed on purpose; the rest of the reward did not.
    assert routed_rewards.shape == base_rewards.shape


def test_route_bearing_is_a_unit_vector_and_never_points_away_from_the_route():
    _, routed, _ = configs()
    env = CompiledMCAPhysicalEnv(routed)
    cosines = []
    for episode in range(12):
        obs, _ = reset_with_valid_particles(env, SEED+episode)
        nodes = obs['nodes']; assignment = env._assigned_targets()
        distance = env._target_distances()
        reference = witness_action(env, assignment.copy())
        for i in range(routed.num_robots):
            j = int(assignment[i])
            local = nodes[i, 76+TARGET_SLOT_DIMS*j:76+TARGET_SLOT_DIMS*j+3]
            assert np.linalg.norm(local) == pytest.approx(1., abs=1e-5)
            if distance[i, j] >= .7 and np.linalg.norm(reference[i]) > 1e-9:
                frame = np.stack((env.tree.tangents[env.robot_stations[i]],
                                  env.tree.normals[env.robot_stations[i]],
                                  env.tree.binormals[env.robot_stations[i]]))
                world = frame.T@local
                cosines.append(float(world@reference[i]/np.linalg.norm(reference[i])))
    cosines = np.array(cosines)
    # The Euclidean bearing in the unchanged prefix scores about +0.32 here and
    # points into the wrong half-space for roughly a fifth of the pairs.
    assert cosines.min() > 0.
    assert cosines.mean() > .6


def test_cleared_targets_report_zero_slots_and_release_their_robots():
    _, routed, _ = configs()
    env = CompiledMCAPhysicalEnv(routed)
    reset_with_valid_particles(env, SEED)
    env.masses[1] = 0.
    nodes = env._observation()['nodes']
    slot = 76+TARGET_SLOT_DIMS
    assert not nodes[:, slot:slot+TARGET_SLOT_DIMS].any()
    assert (env._assigned_targets() != 1).all()
    summary = 76+TARGET_SLOT_DIMS*env.num_clots
    assert nodes[:, summary] == pytest.approx(3/4)


def test_allocation_covers_every_remaining_target():
    _, routed, _ = configs()
    env = CompiledMCAPhysicalEnv(routed)
    assert routed.num_robots >= 4
    for episode in range(40):
        reset_with_valid_particles(env, SEED+episode)
        assignment = env._assigned_targets()
        assert set(assignment.tolist()) == set(range(env.num_clots))
        counts = np.bincount(assignment, minlength=env.num_clots)
        assert counts.min() >= 1 and counts.max() <= 2


def test_allocation_is_sticky_until_the_remaining_set_changes():
    _, routed, _ = configs()
    env = CompiledMCAPhysicalEnv(routed)
    reset_with_valid_particles(env, SEED)
    first = env._assigned_targets().copy()
    rng = np.random.default_rng(3)
    for _ in range(25):
        env.step(rng.uniform(-1, 1, (routed.num_robots, 3)))
        if (env.masses > 0).sum() < env.num_clots:
            break
        assert np.array_equal(env._assigned_targets(), first)
    env.masses[int(first[0])] = 0.
    assert not np.array_equal(env._assigned_targets(), first)


def test_a_new_episode_never_inherits_the_previous_allocation():
    _, routed, _ = configs()
    env = CompiledMCAPhysicalEnv(routed)
    reset_with_valid_particles(env, SEED)
    stale = env._assigned_targets().copy()
    env._assignment[:] = -7
    # reset() observes, so the allocation is rebuilt rather than left unset.
    reset_with_valid_particles(env, SEED)
    assert (env._assigned_targets() != -7).all()
    assert np.array_equal(env._assigned_targets(), stale)
    env._reset_targets()
    assert env._assignment is None


def test_lost_robots_hold_no_target():
    _, routed, _ = configs()
    env = CompiledMCAPhysicalEnv(routed)
    reset_with_valid_particles(env, SEED)
    env.active[0] = False
    env._reset_targets()
    assignment = env._assigned_targets()
    assert assignment[0] == -1
    assert set(assignment[1:].tolist()) == set(range(env.num_clots))


def test_clearing_a_target_can_never_lower_the_monotone_potential():
    base, routed, _ = configs()
    nearest, monotone = [], []
    for cfg, sink in ((base, nearest), (routed, monotone)):
        env = CompiledMCAPhysicalEnv(cfg)
        for episode in range(8):
            reset_with_valid_particles(env, SEED+episode)
            for k in range(env.num_clots):
                before = env._reward_potential().copy()
                saved = env.masses.copy()
                env.masses = saved.copy(); env.masses[k] = 0.
                shaping = cfg.reward_discount*env._reward_potential()-before
                env.masses = saved
                sink.extend(shaping[env.active[:cfg.num_robots]].tolist())
    # The nearest-target potential is what punishes finishing a target.
    assert min(nearest) < -1.
    assert min(monotone) >= 0.


def test_expanded_weights_start_from_the_parent_policy_exactly():
    base, routed, _ = configs()
    parent_env = CompiledMCAPhysicalEnv(base)
    obs, _ = reset_with_valid_particles(parent_env, SEED)
    parent = make_physical_agent(parent_env, seed=42, hidden_dim=128, device='cpu')
    child_env = CompiledMCAPhysicalEnv(routed)
    child_obs, _ = reset_with_valid_particles(child_env, SEED)
    child = make_physical_agent(child_env, seed=7, hidden_dim=128, device='cpu')
    checkpoint = dict(meta=dict(parent.meta), actor=parent.actor.state_dict(),
                      critic=parent.critic.state_dict())
    expanded = initialize_expanded_obstacle_policy(child, checkpoint)
    assert expanded == ['actor.encoder.input_proj.weight', 'critic.inner.obs_encoder.input_proj.weight']
    state = obs['clot_state'].reshape(-1)
    reference = parent.act(obs['nodes'], physical_context(parent_env, obs), state, deterministic=True)
    padded = np.zeros((child_env.num_robots, child_env.obs_dim), np.float32)
    padded[:, :76] = obs['nodes']
    actual = child.act(padded, physical_context(parent_env, obs), state, deterministic=True)
    assert np.allclose(reference[0], actual[0], atol=1e-6)
    assert np.allclose(reference[2], actual[2], atol=1e-6)
