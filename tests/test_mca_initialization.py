"""Robot starts must vary in the simulator, including cached compiled resets."""
from dataclasses import replace
import numpy as np
import pytest

from environments.mca_physical_env import MCAPhysicalEnv, DynamicsConfig
from environments.mca_compiled import CompiledMCAPhysicalEnv
from scripts.train_mca_physical import reset_with_valid_particles


def config(**kwargs):
    return replace(DynamicsConfig.from_json(), robot_initialization='distributed_branches', **kwargs)


@pytest.mark.parametrize('cls', [MCAPhysicalEnv, CompiledMCAPhysicalEnv])
def test_starts_are_seeded_diverse_legal_and_separated(cls):
    env = cls(config())
    starts = []
    for seed in list(range(40)) + [0]:
        reset_with_valid_particles(env, seed)
        pos = env.positions_mm[:env.num_robots].copy()
        starts.append(pos)
        branches = env.tree.branch_ids[env.robot_stations]
        assert len(np.unique(branches)) == env.num_robots
        _, radius, distance, _ = env.transport.coordinates(pos, env.edges[:env.num_robots], env.solution)
        assert np.all(distance+env.config.robot_radius_mm < radius)
        separation = np.linalg.norm(pos[:, None]-pos[None], axis=-1)
        assert np.all(separation[np.triu_indices(env.num_robots, 1)] > 2*env.config.robot_radius_mm)
        assert np.isfinite(env._observation()['nodes']).all()
    np.testing.assert_array_equal(starts[0], starts[-1])
    assert all(not np.array_equal(starts[i], starts[i+1]) for i in range(39))


def test_cached_reset_matches_reference_rng_and_observation():
    ref, fast = MCAPhysicalEnv(config()), CompiledMCAPhysicalEnv(config())
    for seed in (42, 43, 44, 42):
        ro, ri = reset_with_valid_particles(ref, seed)
        fo, fi = reset_with_valid_particles(fast, seed)
        assert ri == fi
        assert ref.np_random.bit_generator.state == fast.np_random.bit_generator.state
        np.testing.assert_array_equal(ref.positions_mm, fast.positions_mm)
        for key in ro:
            np.testing.assert_array_equal(ro[key], fo[key])


@pytest.mark.parametrize('cls', [MCAPhysicalEnv, CompiledMCAPhysicalEnv])
def test_explicit_positions_override_sampling(cls):
    env = cls(config(particle_count=0))
    env.reset(seed=12)
    positions = env.positions_mm.copy()
    env.reset(seed=99, options={'robot_positions_mm': positions})
    np.testing.assert_array_equal(env.positions_mm, positions)


def test_old_protocol_keeps_original_starts():
    env = CompiledMCAPhysicalEnv(DynamicsConfig.from_json())
    reset_with_valid_particles(env, 42)
    before = env.positions_mm[:5].copy()
    reset_with_valid_particles(env, 43)
    np.testing.assert_array_equal(env.positions_mm[:5], before)


def test_invalid_initialization_mode_rejected():
    with pytest.raises(ValueError, match='robot_initialization'):
        replace(DynamicsConfig.from_json(), robot_initialization='typo')


@pytest.mark.parametrize('seed', [42, 43, 44])
def test_distributed_episodes_match_reference_dynamics(seed):
    from tests.test_mca_compiled import compare
    ref, fast = MCAPhysicalEnv(config()), CompiledMCAPhysicalEnv(config())
    reset_with_valid_particles(ref, seed)
    reset_with_valid_particles(fast, seed)
    rng = np.random.default_rng(seed)
    for _ in range(20):
        if compare(ref, fast, rng.uniform(-1, 1, (5, 3))):
            break
    assert ref._done and fast._done
