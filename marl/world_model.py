"""Graph-structured stochastic dynamics ensemble for vascular MARL."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from marl.gat_policy import GATEncoder


@dataclass
class WorldModelConfig:
    obs_dim: int = 36
    action_dim: int = 3
    state_dim: int = 24
    geometry_dim: int = 12
    hidden_dim: int = 192
    num_layers: int = 2
    num_heads: int = 4
    num_scenarios: int = 32
    scenario_embed_dim: int = 16
    ensemble_size: int = 5
    neighbor_radius: float = 0.06


class GraphDynamicsMember(nn.Module):
    NODE_SCALE = 0.10
    STATE_SCALE = 0.05
    POSITION_SCALE = 0.025
    VELOCITY_SCALE = 0.025
    REWARD_SCALE = 20.0

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.scenario_embedding = nn.Embedding(
            config.num_scenarios, config.scenario_embed_dim
        )
        global_dim = (
            config.state_dim + config.geometry_dim + config.scenario_embed_dim
        )
        node_input_dim = config.obs_dim + config.action_dim + 6 + global_dim
        self.encoder = GATEncoder(
            node_input_dim, config.hidden_dim,
            config.num_layers, config.num_heads,
        )
        self.node_delta = nn.Linear(config.hidden_dim, config.obs_dim)
        self.next_velocity = nn.Linear(config.hidden_dim, 3)
        self.position_delta = nn.Linear(config.hidden_dim, 3)
        self.reward = nn.Linear(config.hidden_dim, 1)
        self.global_trunk = nn.Sequential(
            nn.Linear(config.hidden_dim + global_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.state_delta = nn.Linear(config.hidden_dim, config.state_dim)
        self.done_logit = nn.Linear(config.hidden_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]):
        obs = batch["obs"]
        n_agents = obs.shape[1]
        scenario = batch["scenario_id"].long().clamp(
            0, self.config.num_scenarios - 1
        )
        scenario_embedding = self.scenario_embedding(scenario)
        global_features = torch.cat([
            batch["state"], batch["geometry_features"], scenario_embedding
        ], dim=-1)
        expanded_global = global_features[:, None, :].expand(-1, n_agents, -1)
        node_input = torch.cat([
            obs, batch["action"], batch["positions"], batch["velocities"],
            expanded_global,
        ], dim=-1)
        encoded = self.encoder(node_input, batch["adjacency"])
        pooled = encoded.mean(dim=1)
        global_hidden = self.global_trunk(torch.cat([pooled, global_features], dim=-1))
        return {
            "node_delta_norm": self.node_delta(encoded),
            "state_delta_norm": self.state_delta(global_hidden),
            "position_delta_norm": self.position_delta(encoded),
            "next_velocity_norm": self.next_velocity(encoded),
            "reward_norm": self.reward(encoded).squeeze(-1),
            "done_logit": self.done_logit(global_hidden).squeeze(-1),
        }

    def targets(self, batch: dict[str, torch.Tensor]):
        return {
            "node_delta_norm": (batch["next_obs"] - batch["obs"]) / self.NODE_SCALE,
            "state_delta_norm": (
                batch["next_state"] - batch["state"]
            ) / self.STATE_SCALE,
            "position_delta_norm": (
                batch["next_positions"] - batch["positions"]
            ) / self.POSITION_SCALE,
            "next_velocity_norm": batch["next_velocities"] / self.VELOCITY_SCALE,
            "reward_norm": batch["rewards"] / self.REWARD_SCALE,
            "done_logit": batch["done"].float(),
        }

    def loss(self, batch: dict[str, torch.Tensor]):
        prediction = self(batch)
        target = self.targets(batch)
        losses = {
            "node": F.smooth_l1_loss(
                prediction["node_delta_norm"], target["node_delta_norm"]
            ),
            "state": F.smooth_l1_loss(
                prediction["state_delta_norm"], target["state_delta_norm"]
            ),
            "position": F.smooth_l1_loss(
                prediction["position_delta_norm"], target["position_delta_norm"]
            ),
            "velocity": F.smooth_l1_loss(
                prediction["next_velocity_norm"], target["next_velocity_norm"]
            ),
            "reward": F.smooth_l1_loss(
                prediction["reward_norm"], target["reward_norm"]
            ),
            "done": F.binary_cross_entropy_with_logits(
                prediction["done_logit"], target["done_logit"]
            ),
        }
        losses["total"] = (
            losses["node"] + losses["state"] + losses["position"]
            + losses["velocity"] + 2.0 * losses["reward"]
            + 0.25 * losses["done"]
        )
        return losses


class GraphWorldModelEnsemble(nn.Module):
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.members = nn.ModuleList([
            GraphDynamicsMember(config) for _ in range(config.ensemble_size)
        ])

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]):
        predictions = [member(batch) for member in self.members]
        means = {
            key: torch.stack([item[key] for item in predictions]).mean(dim=0)
            for key in predictions[0]
        }
        disagreement = torch.stack([
            torch.stack([item[key] for item in predictions]).var(dim=0, unbiased=False).mean(
                dim=tuple(range(1, predictions[0][key].ndim))
            ) if predictions[0][key].ndim > 1 else
            torch.stack([item[key] for item in predictions]).var(dim=0, unbiased=False)
            for key in ("node_delta_norm", "state_delta_norm", "reward_norm")
        ]).mean(dim=0)

        member = self.members[0]
        next_obs = (
            batch["obs"] + member.NODE_SCALE * means["node_delta_norm"]
        ).clamp(-1.0, 1.0)
        next_state = (
            batch["state"] + member.STATE_SCALE * means["state_delta_norm"]
        ).clamp(-1.0, 1.0)
        next_positions = (
            batch["positions"]
            + member.POSITION_SCALE * means["position_delta_norm"]
        )
        next_velocities = member.VELOCITY_SCALE * means["next_velocity_norm"]
        distances = torch.cdist(next_positions, next_positions)
        next_adjacency = (distances <= self.config.neighbor_radius).float()
        return {
            "obs": next_obs,
            "state": next_state,
            "positions": next_positions,
            "velocities": next_velocities,
            "adjacency": next_adjacency,
            "rewards": member.REWARD_SCALE * means["reward_norm"],
            "done_probability": torch.sigmoid(means["done_logit"]),
            "uncertainty": disagreement,
        }

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        checkpoint = {
            "config": asdict(self.config),
            "model": self.state_dict(),
        }
        if extra is not None:
            checkpoint["extra"] = extra
        torch.save(checkpoint, path)


def load_world_model(path: str | Path, device: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = GraphWorldModelEnsemble(WorldModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint.get("extra", {})
