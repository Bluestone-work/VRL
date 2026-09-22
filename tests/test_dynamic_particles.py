"""Dynamic intravascular particles (EXP_0009): flow advection + records."""
import numpy as np
import pytest

from environments.dynamic_particles import DynamicIntravascularParticles
from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from environments.vector_env import VectorVascularEnv
from environments.vessel_geometry import build_vessel_tree


@pytest.fixture
def tree():
    return build_vessel_tree(
        "bifurcation", np.random.default_rng(3),
        base_radius=0.055, min_radius=0.0033,
    )


def _particles(tree, **kw):
    p = DynamicIntravascularParticles(
        n_envs=2, robot_radius=0.0011, count=20, seed=9, **kw
    )
    p.configure_flow(0.004, 0.055)
    p.reset(tree)
    return p


def test_particles_advect_with_flow(tree):
    """Displacement aligns with the local flow direction, not random walk."""
    p = _particles(tree)
    before = p.positions.copy()
    p.step(tree, tree.radii.copy())
    disp = (p.positions - before).reshape(-1, 3)
    st = tree.nearest_station(before.reshape(-1, 3))
    flow = tree.flow(
        before.reshape(-1, 3), st, 0.004, 0.055, radius_override=tree.radii[st]
    )
    cos = np.sum(flow * disp, axis=1) / (
        np.linalg.norm(flow, axis=1) * np.linalg.norm(disp, axis=1) + 1e-12
    )
    # Mean alignment must be strongly positive: advection dominates drift.
    assert cos.mean() > 0.8


def test_particles_stay_inside_lumen(tree):
    p = _particles(tree)
    for _ in range(10):
        p.step(tree, tree.radii.copy())
    flat = p.positions.reshape(-1, 3)
    axis, radius = tree._axis_point(flat, tree.nearest_station(flat))
    radial = np.linalg.norm(flat - axis, axis=1)
    assert np.all(radial <= radius + 1e-6)


def test_size_is_dimensionless_ratio(tree):
    p = _particles(tree, radius_ratio=3.0)
    assert p.radius == pytest.approx(3.0 * 0.0011)
    assert p.contact_distance == pytest.approx(3.0 * 0.0011 + 0.0011)


def test_dedicated_rng_leaves_env_stream_untouched():
    """Enabling particles must not perturb the env's own RNG draws."""
    a = VectorVascularEnv(
        n_envs=2, scenario="bifurcation", num_robots=3, num_clots=3,
        horizon=30, seed=99, randomize_scenario=False,
    )
    b = VectorVascularEnv(
        n_envs=2, scenario="bifurcation", num_robots=3, num_clots=3,
        horizon=30, seed=99, randomize_scenario=False,
        dynamic_intravascular_particles=True, particle_count=10,
        particle_seed=5,
    )
    act = np.zeros((2, 3, 3), np.float32)
    for _ in range(5):
        a.step(act)
        b.step(act.copy())
    assert a._rng.bit_generator.state["state"] == b._rng.bit_generator.state["state"]
    assert np.array_equal(a._rng.uniform(size=3), b._rng.uniform(size=3))


def test_flag_off_is_bit_identical():
    a = Vascular3DMARLEnv(
        scenario="bifurcation", num_robots=3, num_clots=3, horizon=30, seed=7,
    )
    b = Vascular3DMARLEnv(
        scenario="bifurcation", num_robots=3, num_clots=3, horizon=30, seed=7,
        dynamic_intravascular_particles=False,
    )
    a.reset(seed=5)
    b.reset(seed=5)
    acts = np.random.default_rng(2).uniform(-1, 1, (10, 3, 3))
    for t in range(10):
        oa, ra, _, _, ia = a.step(acts[t])
        ob, rb, _, _, ib = b.step(acts[t].copy())
        assert ra == rb
        assert np.array_equal(oa["nodes"], ob["nodes"])
        assert "particle_collisions" not in ia


def test_particle_records_in_info():
    env = Vascular3DMARLEnv(
        scenario="bifurcation", num_robots=3, num_clots=3, horizon=30, seed=7,
        dynamic_intravascular_particles=True, particle_count=15,
        particle_seed=3,
    )
    env.reset(seed=5)
    obs, reward, _term, _trunc, info = env.step(np.zeros((3, 3), np.float32))
    assert "particle_collisions" in info
    assert "particle_clearance_min" in info
    assert "particle_relative_speed_mean" in info
    assert np.isfinite(info["particle_clearance_min"])
    assert np.isfinite(info["particle_relative_speed_mean"])
    # rewards are unchanged by obstacles (no reward modification in v1)
    env_plain = Vascular3DMARLEnv(
        scenario="bifurcation", num_robots=3, num_clots=3, horizon=30, seed=7,
    )
    env_plain.reset(seed=5)
    _o, r_plain, _, _, _i = env_plain.step(np.zeros((3, 3), np.float32))
    assert reward == pytest.approx(r_plain)


def test_vector_env_particle_records():
    env = VectorVascularEnv(
        n_envs=2, scenario="bifurcation", num_robots=3, num_clots=3,
        horizon=30, seed=7, randomize_scenario=False,
        dynamic_intravascular_particles=True, particle_count=10,
        particle_seed=5,
    )
    env.reset_all()
    obs, reward, _t, _tr, info = env.step(np.zeros((2, 3, 3), np.float32))
    assert info["particle_collisions"].shape == (2, 3)
    assert info["particle_clearance"].shape == (2, 3)
    assert info["particle_relative_speed"].shape == (2, 3)
    assert np.isfinite(info["particle_clearance"]).all()
    # separation impulse actually moves robots out of overlap: place a robot
    # exactly on a particle and check the impulse is nonzero toward exit.
    env.particles.positions[0, 0] = env.robot_positions[0, 0].copy()
    impulse = env.particles.separation_impulse(env.robot_positions)
    assert np.linalg.norm(impulse[0, 0]) > 0.0


def test_state_dict_round_trip_with_particles():
    env = VectorVascularEnv(
        n_envs=2, scenario="bifurcation", num_robots=3, num_clots=3,
        horizon=30, seed=7, randomize_scenario=False,
        dynamic_intravascular_particles=True, particle_count=8,
        particle_seed=5,
    )
    env.reset_all()
    env.step(np.zeros((2, 3, 3), np.float32))
    state = env.state_dict()
    positions_before = env.particles.positions.copy()
    # advance, then restore
    env.step(np.ones((2, 3, 3), np.float32))
    env.load_state_dict(state)
    np.testing.assert_array_equal(env.particles.positions, positions_before)
