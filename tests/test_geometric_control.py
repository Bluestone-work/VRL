from types import SimpleNamespace

import numpy as np
import pytest
import torch

from marl.geometric_control import flow_guidance, local_to_world, policy_action, route_guidance
from marl.mappo_advanced import MAPPOAdvanced
from marl.policy_loader import load_policy
from scripts.eval_success_study import summarize
from scripts.report_success_study import paired_comparison


def _environment(batch=False):
    tree = SimpleNamespace(
        tangents=np.asarray([[0, 1, 0]], np.float32),
        normals=np.asarray([[-1, 0, 0]], np.float32),
        binormals=np.asarray([[0, 0, 1]], np.float32),
    )
    stations = np.zeros((2, 3) if batch else (3,), np.int32)
    return SimpleNamespace(tree=tree, robot_stations=stations, n_envs=2)


def _observation():
    nodes = np.zeros((3, 36), np.float32)
    nodes[:, 6] = 0.1
    nodes[:, 24] = 1.0
    nodes[:, 29] = 0.5
    return {"nodes": nodes, "adjacency": np.eye(3, dtype=np.float32),
            "clot_state": np.zeros((4, 6), np.float32)}


def test_local_frame_conversion_single_vector_and_balanced():
    action = np.asarray([[0.2, 0.3, 0.4]] * 3, np.float32)
    expected = np.asarray([[-0.3, 0.2, 0.4]] * 3, np.float32)
    np.testing.assert_allclose(local_to_world(action, _environment()), expected)
    batch = np.stack([action, action])
    np.testing.assert_allclose(local_to_world(batch, _environment(True)), np.stack([expected] * 2))
    balanced = SimpleNamespace(envs=[_environment(True), _environment(True)])
    np.testing.assert_allclose(local_to_world(np.stack([action] * 4), balanced), np.stack([expected] * 4))


def test_legacy_world_actions_remain_unchanged():
    action = np.asarray([[1.0, -1.0, 0.5]] * 3, np.float32)
    np.testing.assert_array_equal(policy_action(action, {}, None), action)


def test_guidance_cancels_flow_while_holding_contact():
    obs = _observation()
    obs["nodes"][:, 29] = 0.0
    obs["nodes"][:, 21] = 0.2
    obs["nodes"][:, 24] = 0.5
    command = route_guidance(obs["nodes"])
    np.testing.assert_allclose(command[:, 0], -0.4)
    np.testing.assert_allclose(command[:, 1:], 0.0)


def test_guidance_bounded_and_zero_residual_equals_controller():
    obs = _observation()
    raw = np.zeros((3, 3), np.float32)
    command = route_guidance(obs["nodes"])
    assert np.all(np.linalg.norm(command, axis=-1) <= 1.0)
    np.testing.assert_allclose(
        policy_action(raw, obs, _environment(), mode="guided"),
        local_to_world(command, _environment()),
    )
    with pytest.raises(ValueError, match="36-dimensional"):
        route_guidance(np.zeros((3, 20)))
    with pytest.raises(ValueError, match="nonnegative"):
        policy_action(raw, obs, _environment(), residual_scale=-1)


def test_flow_guidance_moves_off_axis_under_saturated_advection():
    obs = _observation()
    obs["nodes"][:, 19] = 0.5
    obs["nodes"][:, 20] = 1.0
    obs["nodes"][:, 18] = 1.0
    obs["nodes"][:, 21] = 1.0
    command = flow_guidance(obs["nodes"])
    assert np.all(command[:, 1] > 0)
    assert np.isfinite(command).all()
    assert np.all(np.linalg.norm(command, axis=-1) <= 1.000001)
    obs["nodes"][:, 19:21] = 0.0
    assert np.isfinite(flow_guidance(obs["nodes"])).all()


def test_guided_checkpoint_roundtrip_and_wrong_control_rejected(tmp_path):
    torch.set_num_threads(1)
    env = _environment()
    obs = _observation()
    env.num_robots = 3
    env.observation_space = {"nodes": np.zeros((3, 36))}
    agent = MAPPOAdvanced(
        n_agents=3, obs_dim=36, action_dim=3, state_dim=24,
        hidden_dim=32, architecture="gat", control_mode="guided", device="cpu",
    )
    path = tmp_path / "guided.pt"
    agent.save(path)
    policy = load_policy(str(path), env)
    expected = policy_action(np.zeros((3, 3)), obs, env, mode="guided")
    np.testing.assert_allclose(policy(obs, env), expected, atol=1e-6)
    agent.load(path)
    agent.meta["control_mode"] = "world"
    with pytest.raises(ValueError, match="control_mode"):
        agent.load(path)


def test_control_wrapper_does_not_mutate_rollout_actions():
    obs = _observation()
    raw = np.full((3, 3), 0.8, np.float32)
    original = raw.copy()
    original_nodes = obs["nodes"].copy()
    for mode in ("local", "guided", "flow_guided"):
        executed = policy_action(raw, obs, _environment(), mode=mode)
        assert np.all(np.linalg.norm(executed, axis=-1) <= 1.000001)
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(obs["nodes"], original_nodes)


def test_local_controller_cannot_use_old_world_model():
    agent = MAPPOAdvanced(
        n_agents=3, obs_dim=36, action_dim=3, hidden_dim=32,
        architecture="gat", control_mode="local", device="cpu",
    )
    with pytest.raises(ValueError, match="world-frame actions"):
        agent.configure_world_model(None)


def test_paired_comparison_rejects_nonmatching_seeds():
    reference = [{"scenario": "example", "seed": 1, "success": 0}]
    candidate = [{"scenario": "example", "seed": 2, "success": 1}]
    with pytest.raises(ValueError, match="identical unique episode keys"):
        paired_comparison([candidate], reference)


def test_summary_uses_cumulative_not_last_step_collisions():
    records = [{
        "scenario": "example", "success": success, "removal_rate": success,
        "return": 1.0, "wall_hits_total": 100, "wall_hits_per_step": 0.5,
        "robot_collisions_total": 2, "steps": 200, "contact_miss": 0,
    } for success in (0, 1)]
    summary = summarize(records)
    assert summary["successes"] == 1
    assert summary["macro"]["wall_hits_total"] == 100
    assert summary["macro"]["wall_hits_per_step"] == 0.5
    assert summary["success_wilson95"][0] < 0.5 < summary["success_wilson95"][1]
