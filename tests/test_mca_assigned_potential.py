"""Own-target ('assigned') progress potential.

The potential is the distance to the target each robot already observes as
assigned to it. Within one allocation it is a pure state potential; on the
step the allocation changes (a target was cleared) the reassigned robots get
zero shaping. Other potentials keep the exact previous arithmetic.
"""
from dataclasses import replace

import numpy as np
import pytest

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig, MCAPhysicalEnv
from scripts.train_mca_physical import reset_with_valid_particles
from scripts.validate_mca_surface_task import witness_action

CONFIG = 'configs/experiments/EXP_0035_UNIT_SPEED_DYNAMICS.json'
SEED = 940000000


def configs():
    base = DynamicsConfig.from_json(CONFIG)
    return base, replace(base, progress_potential='assigned')


def test_assigned_is_a_valid_value_and_the_parent_config_is_unchanged():
    base, assigned = configs()
    assert base.progress_potential == 'mass_weighted'
    assert assigned.progress_potential == 'assigned'
    with pytest.raises(ValueError):
        replace(base, progress_potential='own')


def test_potential_is_the_distance_to_the_own_assigned_target():
    _, cfg = configs()
    env = CompiledMCAPhysicalEnv(cfg)
    reset_with_valid_particles(env, SEED)
    n = cfg.num_robots
    assignment = env._assigned_targets()
    distance = env._target_distances()[np.arange(n), assignment]
    _, _, radial, _ = env.transport.coordinates(env.positions_mm[:n], env.edges[:n], env.solution)
    expected = -cfg.progress_reward_scale*np.maximum(np.hypot(distance, radial)-cfg.contact_distance_mm, 0)
    assert np.allclose(env._reward_potential(), expected)


def test_route_following_toward_own_target_is_rewarded_for_every_robot():
    # The mass-weighted form rewarded 32 percent of robots for moving toward
    # their own target with negative shaping; the own-target form must not.
    _, cfg = configs()
    env = CompiledMCAPhysicalEnv(cfg)
    signs = []
    for episode in range(10):
        reset_with_valid_particles(env, SEED+episode)
        for _ in range(5):
            assignment = env._assigned_targets().copy()
            far = env._target_distances()[np.arange(cfg.num_robots), assignment] > 1.
            _, _, _, _, info = env.step(witness_action(env, assignment.copy()))
            if (env.masses > 0).sum() < env.num_clots:
                break
            signs.extend(info['shaping_rewards'][far & env.active[:cfg.num_robots]].tolist())
    assert len(signs) > 100
    assert np.mean(np.array(signs) > 0) > .95


def test_reassignment_step_gives_zero_shaping_only_to_reassigned_robots():
    _, cfg = configs()
    env = CompiledMCAPhysicalEnv(cfg)
    reset_with_valid_particles(env, SEED)
    before = env._reward_potential().copy()
    old = env._shaping_assignment()
    cleared = int(old[0])
    env.masses[cleared] = 0.
    shaping = env._shaping(before, old)
    new = env._shaping_assignment()
    changed = new != old
    assert changed.any()
    assert np.all(shaping[changed] == 0)
    same = ~changed
    expected = cfg.reward_discount*env._reward_potential()-before
    assert np.allclose(shaping[same], expected[same])


def test_success_step_shaping_is_zero_not_a_distance_windfall():
    _, cfg = configs()
    env = CompiledMCAPhysicalEnv(cfg)
    reset_with_valid_particles(env, SEED)
    before = env._reward_potential().copy()
    old = env._shaping_assignment()
    env.masses[:] = 0.
    assert np.all(env._shaping(before, old) == 0)


def test_other_potentials_keep_their_exact_arithmetic():
    base, _ = configs()
    env = CompiledMCAPhysicalEnv(base)
    reset_with_valid_particles(env, SEED)
    assert env._shaping_assignment() is None
    rng = np.random.default_rng(5)
    for _ in range(20):
        before = env._reward_potential().copy()
        _, _, term, trunc, info = env.step(rng.uniform(-1, 1, (base.num_robots, 3)))
        assert np.array_equal(info['shaping_rewards'], base.reward_discount*env._reward_potential()-before)
        if term or trunc:
            break


def test_compiled_and_reference_backends_agree_on_assigned_shaping():
    _, cfg = configs()
    rewards = []
    for cls in (CompiledMCAPhysicalEnv, MCAPhysicalEnv):
        env = cls(cfg)
        reset_with_valid_particles(env, SEED)
        out = []
        for _ in range(15):
            _, _, term, trunc, info = env.step(witness_action(env, env._assigned_targets().copy()))
            out.append(info['shaping_rewards'].copy())
            if term or trunc:
                break
        rewards.append(np.array(out))
    assert rewards[0].shape == rewards[1].shape
    assert np.allclose(rewards[0], rewards[1], atol=1e-6)
