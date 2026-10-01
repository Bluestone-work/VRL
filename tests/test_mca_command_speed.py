"""Constant-speed actuator option: the policy chooses direction, never slowness.

The default 'bounded' mode must be bit-identical to the historical clip-and-
cap behaviour; 'unit' only rescales nonzero commands to full speed.
"""
from dataclasses import replace

import numpy as np
import pytest

from environments.mca_compiled import CompiledMCAPhysicalEnv
from environments.mca_physical_env import DynamicsConfig, MCAPhysicalEnv, bound_command
from scripts.train_mca_physical import reset_with_valid_particles

CONFIG = 'configs/experiments/EXP_0034_ROUTED_DYNAMICS.json'
SEED = 940000000


def legacy(action):
    action = np.clip(action, -1, 1)
    return action/np.maximum(np.linalg.norm(action, axis=1, keepdims=True), 1)


def test_bounded_matches_legacy_and_unit_rescales_only_nonzero():
    rng = np.random.default_rng(3)
    action = rng.uniform(-2, 2, (64, 3))
    action[:5] = 0.
    action[5:10] *= 1e-3
    np.testing.assert_array_equal(bound_command(action.copy()), legacy(action.copy()))
    unit = bound_command(action.copy(), 'unit')
    norms = np.linalg.norm(unit, axis=1)
    assert np.all(norms[:5] == 0)
    np.testing.assert_allclose(norms[5:], 1, atol=1e-12)
    direction = legacy(action.copy())[5:]
    cos = (unit[5:]*direction).sum(1)/np.linalg.norm(direction, axis=1)
    np.testing.assert_allclose(cos, 1, atol=1e-12)


def test_default_and_invalid_values():
    base = DynamicsConfig.from_json(CONFIG)
    assert base.command_speed == 'bounded'
    assert replace(base, command_speed='unit').command_speed == 'unit'
    with pytest.raises(ValueError):
        replace(base, command_speed='fast')


@pytest.mark.parametrize('backend', [MCAPhysicalEnv, CompiledMCAPhysicalEnv])
def test_unit_mode_moves_at_full_speed_and_bounded_is_unchanged(backend):
    base = DynamicsConfig.from_json(CONFIG)
    small = np.full((base.num_robots, 3), .05)
    paths = {}
    for mode in ('bounded', 'unit'):
        env = backend(replace(base, command_speed=mode))
        reset_with_valid_particles(env, SEED)
        start = env.path_mm[:env.num_robots].copy()
        env.step(small)
        paths[mode] = env.path_mm[:env.num_robots]-start
    # Small commands move far less than a full-speed control interval.
    assert np.all(paths['unit'] > 2*paths['bounded'])


def test_backends_agree_in_unit_mode():
    cfg = replace(DynamicsConfig.from_json(CONFIG), command_speed='unit')
    rng = np.random.default_rng(5)
    envs = [MCAPhysicalEnv(cfg), CompiledMCAPhysicalEnv(cfg)]
    for env in envs:
        reset_with_valid_particles(env, SEED)
    for _ in range(10):
        action = rng.uniform(-.3, .3, (cfg.num_robots, 3))
        for env in envs:
            env.step(action)
    np.testing.assert_allclose(envs[0].positions_mm, envs[1].positions_mm, atol=1e-6)
