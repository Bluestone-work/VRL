"""Separated initialization (EXP_0007): geodesic FPS spawn constraints."""
import numpy as np
import pytest

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from environments.vector_env import VectorVascularEnv
from environments.vessel_geometry import resolve_pool


SCENARIOS = resolve_pool("anatomical")[:4]


def _env(scenario, robots=5, **kw):
    return Vascular3DMARLEnv(
        scenario=scenario, scenario_pool=[scenario], randomize_scenario=False,
        num_robots=robots, num_clots=3, horizon=60, robot_radius=0.0011,
        initialization_mode="separated", obs_mode="geometric", seed=42, **kw
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_separated_spawn_satisfies_distance_constraints(scenario):
    """Default constraints hold at relaxation level 0 and are reported."""
    env = _env(scenario)
    for episode in range(5):
        obs, info = env.reset(seed=1000 + episode)
        assert info["separated_relaxation_level"] == 0
        eu_req = max(8.0 * 2 * env.robot_radius, 1e-4)
        geo_req = 0.12 * env.tree.total_length
        assert info["separated_min_euclidean"] >= eu_req
        assert info["separated_min_geodesic"] >= geo_req


def test_relaxation_ladder_is_bounded_and_recorded():
    """Infeasible constraints relax through recorded levels, no deadlock."""
    env = _env("pulmonary_saddle", robots=10,
               separated_min_euclidean_radii=500.0,
               separated_min_geodesic_fraction=0.9)
    for episode in range(5):
        _obs, info = env.reset(seed=2000 + episode)
        # The ladder terminated (no hang) and recorded a relaxation level.
        assert 0 <= info["separated_relaxation_level"] <= 3
        assert info["separated_relaxation_level"] > 0  # strict impossible here


def test_separated_spread_exceeds_legacy_cluster():
    """Separated spawns are more spread out than the legacy trunk cluster.

    The legacy mode packs robots along the proximal trunk; FPS over the whole
    tree should achieve a strictly larger minimum pairwise geodesic distance
    on essentially every scenario.
    """
    wins = 0
    for scenario in SCENARIOS:
        env_sep = _env(scenario)
        env_leg = Vascular3DMARLEnv(
            scenario=scenario, scenario_pool=[scenario],
            randomize_scenario=False, num_robots=5, num_clots=3,
            horizon=60, robot_radius=0.0011,
            initialization_mode="legacy", obs_mode="geometric", seed=42,
        )
        env_sep.reset(seed=1234)
        env_leg.reset(seed=1234)

        def min_geo(env):
            best = np.inf
            for i in range(env.num_robots):
                d, _ = env.tree.route_to(int(env.robot_stations[i]))
                for j in range(i + 1, env.num_robots):
                    best = min(best, float(d[int(env.robot_stations[j])]))
            return best

        if min_geo(env_sep) > min_geo(env_leg):
            wins += 1
    assert wins >= len(SCENARIOS) - 1


def test_legacy_mode_unchanged():
    """Legacy init reproduces its historical trunk placement."""
    env = Vascular3DMARLEnv(
        scenario="bifurcation", num_robots=5, num_clots=3, horizon=60,
        robot_radius=0.0011, initialization_mode="legacy", seed=7,
    )
    obs, info = env.reset(seed=7)
    assert "separated_relaxation_level" not in info
    trunk = env.tree.stations_of_branch(0)
    assert np.all(np.isin(env.robot_stations, trunk))


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="initialization_mode"):
        Vascular3DMARLEnv(initialization_mode="bogus")


def test_vector_env_separated_mode():
    """The vector env runs the same spawn rule for its shared tree."""
    env = VectorVascularEnv(
        n_envs=3, scenario="pulmonary_saddle", num_robots=5, num_clots=3,
        horizon=50, seed=11, initialization_mode="separated",
        robot_radius=0.0011, randomize_scenario=False,
    )
    obs = env.reset_all()
    assert obs["agent_mask"].all()
    # All env rows share one tree; each reset draws its own FPS seed station,
    # so stations can differ per row. What must hold everywhere is the
    # separation constraint itself.
    for row in range(3):
        pts = env.robot_positions[row]
        eu = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(eu, np.inf)
        assert eu.min() > 2.0 * 2 * env.robot_radius
    _, _, _, _, info = env.step(np.zeros((3, 5, 3), np.float32))
    assert "separated_relaxation_level" in info
