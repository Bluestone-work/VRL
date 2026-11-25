"""Regression tests for finite-vessel mechanics and PPO diagnostics."""

from __future__ import annotations

import numpy as np
import torch

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from environments.vector_env import VectorVascularEnv
from environments.vessel_geometry import build_vessel_tree
from marl.mappo_policy import GaussianActor


def test_terminal_projection_clamps_axial_escape() -> None:
    tree = build_vessel_tree("straight", np.random.default_rng(7))
    terminal = tree.branches[0].stop
    probe = tree.points[terminal] + tree.tangents[terminal] * 0.25
    projected, outside, station, _axis = tree.project(
        probe[None], robot_radius=0.001, hint=np.asarray([terminal])
    )
    assert outside[0]
    assert station[0] == terminal
    axial = np.dot(projected[0] - tree.points[terminal], tree.tangents[terminal])
    assert axial <= 1e-6


def test_no_action_flow_and_diagnostics_are_finite() -> None:
    env = Vascular3DMARLEnv(
        scenario="straight", scenario_pool=["straight"], randomize_scenario=False,
        randomize_clots=False, num_robots=1, num_clots=1, seed=4,
        obs_mode="geometric_v2",
    )
    env.brownian_sigma = 0.0
    env.reset(seed=5)
    start = env.robot_positions.copy()
    _obs, _reward, _terminated, _truncated, info = env.step(np.zeros((1, 3), np.float32))
    displacement = env.robot_positions[0] - start[0]
    assert np.dot(displacement, env.tree.tangents[env.robot_stations[0]]) >= -1e-7
    for key in (
        "local_flow_speed", "available_control_speed", "axial_flow_speed",
        "control_authority_ratio", "distance_to_target", "geodesic_progress",
    ):
        assert np.isfinite(np.asarray(info[key])).all(), key


def test_adaptive_one_substep_matches_legacy_without_noise() -> None:
    kwargs = dict(
        scenario="bifurcation", scenario_pool=["bifurcation"], randomize_scenario=False,
        randomize_clots=False, num_robots=2, num_clots=1, seed=11,
        obs_mode="geometric_v2",
    )
    legacy = Vascular3DMARLEnv(**kwargs, integration_mode="legacy")
    adaptive = Vascular3DMARLEnv(**kwargs, integration_mode="adaptive",
                                 max_substep_displacement=1.0)
    legacy.brownian_sigma = adaptive.brownian_sigma = 0.0
    legacy.reset(seed=12)
    adaptive.reset(seed=12)
    action = np.asarray([[0.2, 0.0, 0.0], [0.0, 0.1, 0.0]], np.float32)
    legacy.step(action)
    adaptive.step(action)
    assert np.allclose(legacy.robot_positions, adaptive.robot_positions, atol=2e-6)


def test_vector_and_single_environment_match_for_one_slot() -> None:
    kwargs = dict(
        scenario="straight", scenario_pool=["straight"], randomize_scenario=False,
        randomize_clots=False, num_robots=1, num_clots=1, seed=21,
        obs_mode="geometric_v2",
    )
    single = Vascular3DMARLEnv(**kwargs)
    vector = VectorVascularEnv(
        n_envs=1, scenario=kwargs["scenario"], scenario_pool=kwargs["scenario_pool"],
        randomize_scenario=False, num_robots=1, num_clots=1, seed=kwargs["seed"],
        obs_mode=kwargs["obs_mode"],
    )
    single.brownian_sigma = vector.brownian_sigma = 0.0
    single.reset(seed=22)
    vector.tree = single.tree
    vector.robot_positions[0] = single.robot_positions
    vector.robot_stations[0] = single.robot_stations
    vector.robot_velocities[0] = single.robot_velocities
    vector.clot_masses[0, : single.active_clots] = 0.0
    single.clot_masses[:] = 0.0
    action = np.zeros((1, 1, 3), np.float32)
    single.step(action[0])
    vector.step(action)
    assert np.allclose(single.robot_positions, vector.robot_positions[0], atol=2e-6)


def test_v2_flow_feature_retains_intensity_difference() -> None:
    env = Vascular3DMARLEnv(
        scenario="straight", scenario_pool=["straight"], randomize_scenario=False,
        randomize_clots=False, num_robots=1, num_clots=1, seed=31,
        obs_mode="geometric_v2",
    )
    env.reset(seed=32)
    env.set_flow_scale(0.2)
    weak = env._build_observation()["nodes"][0, 36]
    env.set_flow_scale(1.0)
    strong = env._build_observation()["nodes"][0, 36]
    assert strong > weak + 1e-5


def test_squashed_entropy_gradient_and_pre_update_ratio() -> None:
    torch.manual_seed(3)
    actor = GaussianActor(5, 3, log_std_init=-0.3)
    features = torch.randn(4, 2, 5)
    actions, old_log_prob, _ = actor.get_actions(features)
    new_log_prob, entropy = actor.evaluate_actions(features, actions)
    assert torch.allclose(torch.exp(new_log_prob - old_log_prob),
                          torch.ones_like(old_log_prob), atol=2e-5)
    entropy.mean().backward()
    assert actor.mean.weight.grad is not None
    assert torch.isfinite(actor.mean.weight.grad).all()


def test_infeasible_flow_terminates_terminal_cap_stall() -> None:
    env = Vascular3DMARLEnv(
        scenario="straight", scenario_pool=["straight"], randomize_scenario=False,
        randomize_clots=False, num_robots=1, num_clots=1, seed=44,
        infeasible_flow_patience=2,
    )
    env.brownian_sigma = 0.0
    env.flow_speed = 0.02
    env.reset(seed=45)
    terminal = int(np.flatnonzero(env.tree._terminal_cap)[-1])
    env.robot_stations[0] = terminal
    env.robot_positions[0] = env.tree.points[terminal]
    env.clot_masses[:] = 1.0
    _, _, terminated, truncated, _ = env.step(np.zeros((1, 3), np.float32))
    assert not terminated and not truncated
    _, _, terminated, truncated, info = env.step(np.zeros((1, 3), np.float32))
    assert terminated and not truncated
    assert info["physics_failure"]
    assert info["termination_reason"] == "infeasible_flow"
