from types import SimpleNamespace

import numpy as np
import pytest
import torch

from marl.geometric_control import direct_local_action, local_to_world, policy_action
from marl.mappo_advanced import MAPPOAdvanced


def _env():
    tree = SimpleNamespace(
        tangents=np.asarray([[1.0, 0.0, 0.0]], np.float32),
        normals=np.asarray([[0.0, 1.0, 0.0]], np.float32),
        binormals=np.asarray([[0.0, 0.0, 1.0]], np.float32),
    )
    return SimpleNamespace(tree=tree, robot_stations=np.zeros(2, dtype=np.int32))


def _obs():
    return {"nodes": np.zeros((2, 36), np.float32)}


def test_local_control_has_no_guidance(monkeypatch):
    import marl.geometric_control as control

    def forbidden(*_args, **_kwargs):
        raise AssertionError("guidance controller was called in local mode")

    monkeypatch.setattr(control, "route_guidance", forbidden)
    monkeypatch.setattr(control, "flow_guidance", forbidden)
    monkeypatch.setattr(control, "spread_flow_guidance", forbidden)
    raw = np.asarray([[0.2, 0.3, 0.4], [-0.4, 0.1, 0.2]], np.float32)
    np.testing.assert_allclose(
        policy_action(raw, _obs(), _env(), mode="local", residual_scale=999.0),
        direct_local_action(raw, _env()),
    )


def test_local_control_matches_frenet_transform():
    env = _env()
    raw = np.asarray([[2.0, -1.0, 0.5], [0.2, 0.3, 0.4]], np.float32)
    bounded = np.clip(raw, -1.0, 1.0)
    bounded /= np.maximum(np.linalg.norm(bounded, axis=-1, keepdims=True), 1.0)
    expected = local_to_world(bounded, env)
    np.testing.assert_allclose(policy_action(raw, _obs(), env, mode="local"), expected)


def test_local_control_ignores_residual_scale():
    env = _env()
    raw = np.asarray([[0.4, -0.1, 0.7], [-0.2, 0.8, 0.1]], np.float32)
    first = policy_action(raw, _obs(), env, mode="local", residual_scale=0.0, guidance_speed=0.65)
    second = policy_action(raw, _obs(), env, mode="local", residual_scale=-123.0, guidance_speed=0.0)
    np.testing.assert_array_equal(first, second)


def test_local_checkpoint_uses_direct_action_semantics(tmp_path):
    torch.set_num_threads(1)
    agent = MAPPOAdvanced(
        n_agents=2, obs_dim=36, action_dim=3, state_dim=12,
        hidden_dim=16, num_layers=1, architecture="gat",
        control_mode="local", device="cpu",
    )
    assert agent.meta["action_semantics"] == "direct_local_frenet"
    path = tmp_path / "direct_local.pt"
    agent.save(path)
    restored = MAPPOAdvanced(
        n_agents=2, obs_dim=36, action_dim=3, state_dim=12,
        hidden_dim=16, num_layers=1, architecture="gat",
        control_mode="local", device="cpu",
    )
    restored.load(path)
    assert restored.meta["action_semantics"] == "direct_local_frenet"
    restored.meta["action_semantics"] = "controller_residual"
    with pytest.raises(RuntimeError, match="direct_local_frenet"):
        restored.env_action(np.zeros((2, 3), np.float32), _obs(), _env())


def test_local_rejects_nonfinite_policy_action():
    with pytest.raises(FloatingPointError, match="non-finite"):
        policy_action(np.asarray([[np.nan, 0.0, 0.0]], np.float32), _obs(), _env(), mode="local")
