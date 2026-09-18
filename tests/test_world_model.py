from __future__ import annotations

import numpy as np
import torch

from marl.mappo_advanced import MAPPOAdvanced
from marl.transition_dataset import (
    EpisodeTransitionWriter,
    load_transition_shards,
    split_by_geometry,
)
from marl.world_model import GraphWorldModelEnsemble, WorldModelConfig
from scripts.train_vector_mappo import curriculum_difficulty


def _world_batch(batch_size=8, n_agents=3, obs_dim=10, state_dim=12):
    return {
        "obs": torch.rand(batch_size, n_agents, obs_dim) * 2 - 1,
        "next_obs": torch.rand(batch_size, n_agents, obs_dim) * 2 - 1,
        "action": torch.rand(batch_size, n_agents, 3) * 2 - 1,
        "rewards": torch.randn(batch_size, n_agents),
        "state": torch.rand(batch_size, state_dim) * 2 - 1,
        "next_state": torch.rand(batch_size, state_dim) * 2 - 1,
        "adjacency": torch.ones(batch_size, n_agents, n_agents),
        "positions": torch.rand(batch_size, n_agents, 3),
        "next_positions": torch.rand(batch_size, n_agents, 3),
        "velocities": torch.randn(batch_size, n_agents, 3) * 0.01,
        "next_velocities": torch.randn(batch_size, n_agents, 3) * 0.01,
        "geometry_features": torch.rand(batch_size, 12),
        "scenario_id": torch.arange(batch_size) % 4,
        "done": torch.zeros(batch_size),
    }


def test_transition_writer_roundtrip_and_geometry_split(tmp_path):
    writer = EpisodeTransitionWriter(tmp_path, n_envs=2, shard_size=4)
    for step in range(3):
        writer.append(
            done=np.asarray([step == 2, False]),
            obs=np.full((2, 3, 4), step, np.float32),
            geometry_id=np.asarray([1, 2], np.int64),
        )
    writer.close()
    data = load_transition_shards(tmp_path)
    assert data["obs"].shape == (6, 3, 4)
    assert np.unique(data["episode_id"]).size == 2
    split = split_by_geometry(data["geometry_id"], seed=0)
    joined = np.concatenate(list(split.values()))
    assert sorted(joined.tolist()) == list(range(6))


def test_graph_world_model_loss_and_prediction_shapes():
    config = WorldModelConfig(
        obs_dim=10, state_dim=12, hidden_dim=32, ensemble_size=2,
    )
    model = GraphWorldModelEnsemble(config)
    batch = _world_batch()
    loss = model.members[0].loss(batch)
    assert torch.isfinite(loss["total"])
    prediction = model.predict(batch)
    assert prediction["obs"].shape == (8, 3, 10)
    assert prediction["state"].shape == (8, 12)
    assert prediction["rewards"].shape == (8, 3)
    assert prediction["uncertainty"].shape == (8,)


def test_world_model_does_not_clip_anatomical_positions_at_zero():
    config = WorldModelConfig(
        obs_dim=10, state_dim=12, hidden_dim=32, ensemble_size=2,
    )
    model = GraphWorldModelEnsemble(config)
    for member in model.members:
        torch.nn.init.zeros_(member.position_delta.weight)
        torch.nn.init.zeros_(member.position_delta.bias)
    batch = _world_batch()
    batch["positions"] -= 1.0
    prediction = model.predict(batch)
    assert torch.allclose(prediction["positions"], batch["positions"])
    assert (prediction["positions"] < 0.0).any()


def test_world_model_value_expansion_only_changes_critic_target():
    n_envs, n_agents, obs_dim, state_dim = 2, 3, 10, 12
    agent = MAPPOAdvanced(
        n_agents=n_agents, obs_dim=obs_dim, action_dim=3,
        state_dim=state_dim, architecture="gat", hidden_dim=32, device="cpu",
    )
    model = GraphWorldModelEnsemble(WorldModelConfig(
        obs_dim=obs_dim, state_dim=state_dim, hidden_dim=32, ensemble_size=2,
    ))
    agent.configure_world_model(model, horizon=2, blend=0.25, uncertainty_threshold=1e6)
    for _ in range(3):
        batch = _world_batch(n_envs, n_agents, obs_dim, state_dim)
        obs = batch["obs"].numpy()
        ctx = {
            "adjacency": batch["adjacency"].numpy(),
            "positions": batch["positions"].numpy(),
            "velocities": batch["velocities"].numpy(),
        }
        next_ctx = {
            "adjacency": batch["adjacency"].numpy(),
            "positions": batch["next_positions"].numpy(),
            "velocities": batch["next_velocities"].numpy(),
        }
        actions, log_probs, values = agent.act_batch(
            obs, ctx, batch["state"].numpy()
        )
        agent.buffer.store(
            obs, actions, batch["rewards"].numpy(),
            np.zeros((n_envs, n_agents), np.float32), log_probs, values,
            ctx, batch["state"].numpy(),
            next_obs=batch["next_obs"].numpy(), next_ctx=next_ctx,
            next_state=batch["next_state"].numpy(),
            geometry_features=batch["geometry_features"].numpy(),
            next_geometry_features=batch["geometry_features"].numpy(),
            terminals=np.zeros((n_envs, n_agents), np.float32),
            scenario_id=batch["scenario_id"].numpy(),
        )
    metrics = agent.update(n_epochs=1, batch_size=12)
    assert metrics["mve_fraction"] == 1.0
    assert metrics["mve_target_delta"] > 0.0


def test_fixed_budget_curriculum_has_three_expected_stages():
    boundaries = (0.2, 0.5)
    difficulties = (0.2, 0.6, 1.0)
    assert curriculum_difficulty(0, 1_000, boundaries, difficulties) == 0.2
    assert curriculum_difficulty(199, 1_000, boundaries, difficulties) == 0.2
    assert curriculum_difficulty(200, 1_000, boundaries, difficulties) == 0.6
    assert curriculum_difficulty(499, 1_000, boundaries, difficulties) == 0.6
    assert curriculum_difficulty(500, 1_000, boundaries, difficulties) == 1.0
