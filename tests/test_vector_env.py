"""The vectorized env must agree with the single env it replaces.

Equivalence is checked by construction rather than by porting the physics twice
and hoping: both are driven from an identical state with identical actions, and
the resulting positions, rewards and observations are compared. Anything that
drifts here means the batched rewrite changed the task, which would silently
invalidate every result collected with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.vascular_3d_marl_env import Vascular3DMARLEnv  # noqa: E402
from environments.vector_env import VectorVascularEnv  # noqa: E402
from environments.balanced_vector_env import BalancedVectorVascularEnv  # noqa: E402


def _mirror(single: Vascular3DMARLEnv, vec: VectorVascularEnv, slot: int = 0) -> None:
    """Force `single` to hold exactly the state of env `slot` in `vec`."""
    single.tree = vec.tree
    single._route_cache = {}
    alive = vec.clot_alive[slot]
    n_alive = int(alive.sum())
    single.clot_stations = vec.clot_stations[slot][alive].astype(np.int32)
    single.clot_positions = vec.clot_positions[slot][alive].copy()
    single.clot_masses = vec.clot_masses[slot][alive].copy()
    single.clot_initial_mass = vec.clot_initial[slot][alive].copy()
    single.active_clots = n_alive
    single.robot_positions = vec.robot_positions[slot].copy()
    single.robot_stations = vec.robot_stations[slot].copy()
    single.robot_velocities = vec.robot_velocities[slot].copy()
    single.steps = int(vec.steps[slot])
    single._first_contact_step = int(vec.first_contact[slot])
    single._milestones_hit = set()
    single.active_scenario = vec.active_scenario


def _make_pair(obs_mode: str = "geometric", n_envs: int = 4, seed: int = 0):
    vec = VectorVascularEnv(
        n_envs=n_envs, scenario="bifurcation", num_robots=3, num_clots=2,
        horizon=100, seed=seed, randomize_scenario=False, obs_mode=obs_mode,
    )
    single = Vascular3DMARLEnv(
        scenario="bifurcation", num_robots=3, num_clots=2, horizon=100,
        seed=seed, randomize_scenario=False, randomize_clots=False,
        use_pybullet=False, obs_mode=obs_mode,
    )
    single.reset(seed=seed)
    return vec, single


# ------------------------------------------------------------------ contracts


@pytest.mark.parametrize("obs_mode,dim", [("geometric", 36), ("legacy", 20)])
def test_shapes(obs_mode: str, dim: int) -> None:
    vec = VectorVascularEnv(n_envs=8, num_robots=4, num_clots=3, horizon=50,
                            seed=1, obs_mode=obs_mode)
    obs = vec.reset_all()
    assert obs["nodes"].shape == (8, 4, dim)
    assert obs["adjacency"].shape == (8, 4, 4)
    assert obs["clot_state"].shape == (8, 4, 6)

    obs, reward, term, trunc, info = vec.step(
        np.zeros((8, 4, 3), np.float32)
    )
    assert reward.shape == (8,)
    assert term.shape == (8,) and trunc.shape == (8,)
    assert info["agent_rewards"].shape == (8, 4)


def test_observations_stay_in_range() -> None:
    vec = VectorVascularEnv(n_envs=16, num_robots=4, num_clots=3, horizon=60,
                            seed=2, randomize_scenario=True)
    obs = vec.reset_all()
    rng = np.random.default_rng(0)
    for _ in range(120):
        assert np.isfinite(obs["nodes"]).all()
        assert obs["nodes"].min() >= -1.0 - 1e-5
        assert obs["nodes"].max() <= 1.0 + 1e-5
        obs, r, _t, _u, _i = vec.step(
            rng.uniform(-1, 1, (16, 4, 3)).astype(np.float32)
        )
        assert np.isfinite(r).all()


def test_robots_stay_inside_the_lumen() -> None:
    vec = VectorVascularEnv(n_envs=12, num_robots=5, num_clots=2, horizon=80, seed=3)
    vec.reset_all()
    push = np.tile(np.array([0.2, 1.0, 1.0], np.float32), (12, 5, 1))
    for _ in range(60):
        vec.step(push)
        flat = vec.robot_positions.reshape(-1, 3)
        axis, radius = vec.tree._axis_point(flat, vec.robot_stations.reshape(-1))
        radial = np.linalg.norm(flat - axis, axis=1)
        assert np.all(radial <= radius - vec.robot_radius + 1e-4)


def test_auto_reset_preserves_the_final_observation() -> None:
    """Bootstrapping needs the terminal obs, not the next episode's first obs."""
    vec = VectorVascularEnv(n_envs=4, num_robots=3, num_clots=1, horizon=5, seed=4)
    vec.reset_all()
    for _ in range(4):
        obs, _r, term, trunc, info = vec.step(np.zeros((4, 3, 3), np.float32))
    # Horizon 5 -> step 5 truncates every env at once.
    obs, _r, term, trunc, info = vec.step(np.zeros((4, 3, 3), np.float32))
    assert trunc.all(), "expected all envs to truncate at the horizon"
    assert "final_observation" in info
    assert info["final_observation"]["nodes"].shape == obs["nodes"].shape
    # Steps must have been reset for the new episode.
    assert np.all(vec.steps == 0)


def test_only_done_envs_are_reset() -> None:
    """A reset must not disturb the envs that are still running."""
    vec = VectorVascularEnv(n_envs=6, num_robots=3, num_clots=1, horizon=200, seed=5)
    vec.reset_all()
    for _ in range(10):
        vec.step(np.zeros((6, 3, 3), np.float32))
    # Force exactly one env to finish by clearing its clots.
    vec.clot_masses[2] = 0.0
    before = vec.robot_positions.copy()
    steps_before = vec.steps.copy()
    _obs, _r, term, _trunc, _info = vec.step(np.zeros((6, 3, 3), np.float32))
    assert term[2] and not term[[0, 1, 3, 4, 5]].any()
    assert vec.steps[2] == 0, "finished env was not reset"
    assert np.all(vec.steps[[0, 1, 3, 4, 5]] == steps_before[[0, 1, 3, 4, 5]] + 1)
    # Untouched envs kept moving rather than being snapped back to a reset state.
    assert not np.allclose(vec.robot_positions[0], before[0])


def test_difficulty_is_clamped() -> None:
    vec = VectorVascularEnv(n_envs=2, horizon=20, seed=6)
    vec.set_difficulty(9.0)
    assert vec._difficulty == 1.0
    vec.set_difficulty(-1.0)
    assert vec._difficulty == 0.0


def test_clot_slots_are_masked() -> None:
    """Inactive clot slots must be zeroed, not filled with stale geometry."""
    vec = VectorVascularEnv(n_envs=10, num_robots=3, num_clots=3, horizon=40, seed=7)
    obs = vec.reset_all()
    dead = ~vec.clot_alive
    if not dead.any():
        pytest.skip("every slot happened to be active")
    assert np.allclose(obs["clot_state"][dead], 0.0)


def test_anatomical_scenarios_use_declared_clot_sites() -> None:
    vec = VectorVascularEnv(
        n_envs=12,
        scenario="mca_m1_lvo",
        scenario_pool=["mca_m1_lvo"],
        randomize_scenario=False,
        num_robots=3,
        num_clots=3,
        robot_radius=0.0011,
        seed=8,
    )
    candidates = vec._territory_clot_candidates()
    assert candidates is not None
    declared_stations = set(candidates[0].tolist())
    for row in range(vec.n_envs):
        active = vec.clot_stations[row][vec.clot_alive[row]]
        assert active.size > 0
        assert set(active.tolist()) <= declared_stations


def test_reward_modes_match_single_env_constants() -> None:
    for mode in ("baseline", "milestone"):
        vec = VectorVascularEnv(n_envs=2, reward_mode=mode, seed=9)
        single = Vascular3DMARLEnv(reward_mode=mode, seed=9)
        assert vec.success_bonus == single.success_bonus
        assert vec.clot_cleared_bonus == single.clot_cleared_bonus
        assert vec._milestones == single._milestones


def test_geometry_id_advances_when_tree_is_resampled() -> None:
    vec = VectorVascularEnv(n_envs=2, seed=10)
    first = vec.active_geometry_id
    vec.reset_all()
    assert vec.active_geometry_id == first + 1


def test_balanced_vector_env_represents_every_territory() -> None:
    env = BalancedVectorVascularEnv(
        n_envs=28, scenario_pool="anatomical", num_robots=3, num_clots=2,
        horizon=20, tree_resample_interval=20, robot_radius=0.0011, seed=11,
    )
    obs = env.reset_all()
    assert obs["nodes"].shape == (28, 3, 36)
    counts = np.bincount(env.scenario_ids)
    assert counts.size == 14
    assert counts.min() == counts.max() == 2

    action = np.zeros((28, 3, 3), np.float32)
    obs, reward, term, trunc, info = env.step(action)
    assert reward.shape == (28,)
    assert info["scenario"].shape == (28,)
    assert np.unique(info["scenario"]).size == 14
    assert env.geometry_features.shape == (28, 12)


def test_balanced_vector_env_applies_difficulty_to_every_territory() -> None:
    env = BalancedVectorVascularEnv(
        n_envs=14, scenario_pool="anatomical", num_robots=3, num_clots=3,
        horizon=20, tree_resample_interval=20, robot_radius=0.0011, seed=111,
    )
    env.set_difficulty(0.6)
    assert env.difficulty == 0.6
    assert all(child._difficulty == 0.6 for child in env.envs)


def test_balanced_env_preserves_terminal_context() -> None:
    env = BalancedVectorVascularEnv(
        n_envs=14, scenario_pool="anatomical", num_robots=3, num_clots=1,
        horizon=1, tree_resample_interval=10, robot_radius=0.0011, seed=12,
    )
    action = np.zeros((14, 3, 3), np.float32)
    _obs, _reward, _term, trunc, info = env.step(action)
    assert trunc.all()
    assert info["final_observation"]["nodes"].shape == (14, 3, 36)
    assert info["final_context"]["positions"].shape == (14, 3, 3)


# ------------------------------------------------------- agreement with single


@pytest.mark.parametrize("obs_mode", ["geometric", "legacy"])
def test_observation_matches_single_env(obs_mode: str) -> None:
    """Same state, same observation."""
    vec, single = _make_pair(obs_mode=obs_mode, seed=11)
    vec.reset_all()
    _mirror(single, vec, slot=0)

    got = vec._observe()["nodes"][0]
    want = single._build_observation()["nodes"]
    assert np.allclose(got, want, atol=1e-5), np.abs(got - want).max()


def test_dynamics_match_single_env() -> None:
    """Same state and same action produce the same next positions.

    Brownian noise is disabled on both sides: it is drawn from each env's own RNG
    and is not part of what this test is checking.
    """
    vec, single = _make_pair(seed=13)
    vec.reset_all()
    vec.brownian_sigma = 0.0
    single.brownian_sigma = 0.0
    _mirror(single, vec, slot=0)

    rng = np.random.default_rng(0)
    for _ in range(15):
        action = rng.uniform(-1, 1, (vec.n_envs, 3, 3)).astype(np.float32)
        _o, reward, _t, _u, info = vec.step(action)
        s_obs, s_reward, s_term, s_trunc, s_info = single.step(action[0])

        assert np.allclose(
            vec.robot_positions[0], single.robot_positions, atol=1e-5
        ), np.abs(vec.robot_positions[0] - single.robot_positions).max()
        assert np.allclose(
            info["agent_rewards"][0], s_info["agent_rewards"], atol=1e-4
        )
        assert float(reward[0]) == pytest.approx(float(s_reward), abs=1e-4)
        if s_term or s_trunc:
            break


def test_lysis_matches_single_env() -> None:
    """Clot mass must decay identically, including the saturation term."""
    vec, single = _make_pair(seed=17)
    vec.reset_all()
    vec.brownian_sigma = 0.0
    single.brownian_sigma = 0.0

    # Park every robot on the first clot so lysis definitely fires.
    target = vec.clot_positions[:, 0, :]
    vec.robot_positions = np.repeat(target[:, None, :], 3, axis=1).astype(np.float32)
    vec.robot_stations = np.repeat(
        vec.clot_stations[:, 0:1], 3, axis=1
    ).astype(np.int32)
    _mirror(single, vec, slot=0)

    action = np.zeros((vec.n_envs, 3, 3), np.float32)
    for _ in range(5):
        _o, _r, _t, _u, info = vec.step(action)
        single.step(action[0])
        alive = vec.clot_alive[0]
        assert np.allclose(
            vec.clot_masses[0][alive], single.clot_masses, atol=1e-5
        ), (vec.clot_masses[0][alive], single.clot_masses)


def test_occlusion_matches_single_env() -> None:
    vec, single = _make_pair(seed=19)
    vec.reset_all()
    _mirror(single, vec, slot=0)
    got = vec._occluded_radius(vec.robot_stations)[0]
    want = single._occluded_radius(single.robot_stations)
    assert np.allclose(got, want, atol=1e-6)


def test_geodesic_matches_single_env() -> None:
    vec, single = _make_pair(seed=23)
    vec.reset_all()
    _mirror(single, vec, slot=0)
    target_v = vec._assign()
    target_s = single._assigned_clot()
    # The slot indexing agrees only when every slot is active; compare distances,
    # which is what the reward actually consumes.
    if not vec.clot_alive[0].all():
        pytest.skip("inactive slots shift the clot indexing")
    assert np.array_equal(target_v[0], target_s)
    assert np.allclose(
        vec._geodesic(target_v)[0], single._geodesic_to_target(target_s), atol=1e-6
    )


def test_vector_env_is_faster_than_the_loop() -> None:
    """The entire point of the batched env: it must beat N single steps.

    A modest threshold, so this stays a smoke test on machines under load rather
    than a flaky benchmark.
    """
    import time

    n_envs = 32
    vec = VectorVascularEnv(n_envs=n_envs, num_robots=3, num_clots=2,
                            horizon=200, seed=29)
    vec.reset_all()
    action = np.zeros((n_envs, 3, 3), np.float32)
    vec.step(action)  # warm up
    t0 = time.time()
    for _ in range(30):
        vec.step(action)
    vec_rate = 30 * n_envs / (time.time() - t0)

    single = Vascular3DMARLEnv(num_robots=3, num_clots=2, horizon=200, seed=29,
                               use_pybullet=False)
    single.reset(seed=29)
    one = np.zeros((3, 3), np.float32)
    single.step(one)
    t0 = time.time()
    for _ in range(100):
        _o, _r, t, u, _i = single.step(one)
        if t or u:
            single.reset()
    single_rate = 100 / (time.time() - t0)

    assert vec_rate > single_rate * 2.0, (
        f"vector env {vec_rate:.0f} steps/s vs single {single_rate:.0f} steps/s"
    )
