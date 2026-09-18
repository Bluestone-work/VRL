"""High-level clot assignment policy for hierarchical MARL experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def allocator_features(env, assignments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build per-robot/per-clot features and an action mask."""
    obs = env._build_observation()
    n = env.num_robots
    slots = env._max_clot_slots
    robot = np.asarray(obs["nodes"], dtype=np.float32)
    clot = np.asarray(obs["clot_state"], dtype=np.float32)
    pair = np.zeros((n, slots, 8), dtype=np.float32)
    loads = np.bincount(
        assignments[assignments >= 0], minlength=max(env.active_clots, 1)
    ).astype(np.float32)
    for c in range(min(env.active_clots, slots)):
        delta = env.clot_positions[c][None, :] - env.robot_positions
        station = env.robot_stations
        tangent = env.tree.tangents[station]
        normal = env.tree.normals[station]
        binormal = env.tree.binormals[station]
        local = np.stack([
            np.sum(delta * tangent, axis=1),
            np.sum(delta * normal, axis=1),
            np.sum(delta * binormal, axis=1),
        ], axis=1)
        distance, _ = env._route(c)
        geo = distance[station] / max(float(env.tree.total_length), 1e-8)
        eta = geo * float(env.tree.total_length) / max(env.max_speed, 1e-8)
        eta /= max(float(env.horizon), 1.0)
        pair[:, c, 0:3] = np.clip(local / 0.5, -1.0, 1.0)
        pair[:, c, 3] = np.clip(geo, 0.0, 1.0)
        pair[:, c, 4] = np.clip(eta, 0.0, 2.0)
        pair[:, c, 5] = env.clot_masses[c] / max(float(env.clot_initial_mass[c]), 1e-8)
        pair[:, c, 6] = 1.0 if env.clot_masses[c] > 0 else 0.0
        pair[:, c, 7] = loads[c] / max(float(n), 1.0)
    robot_context = np.repeat(robot[:, None, :], slots, axis=1)
    clot_context = np.repeat(clot[None, :, :], n, axis=0)
    features = np.concatenate([robot_context, clot_context, pair], axis=-1)
    valid = np.zeros((n, slots + 1), dtype=bool)
    valid[:, : min(env.active_clots, slots)] = (
        env.clot_masses[: min(env.active_clots, slots)] > 0
    )[None, :]
    valid[:, -1] = True
    return features.astype(np.float32), valid


class HierarchicalAllocator(nn.Module):
    """Shared robot allocator with target-pair attention and a centralized critic."""

    def __init__(self, feature_dim: int, slots: int, hidden_dim: int = 128):
        super().__init__()
        self.slots = int(slots)
        self.pair_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.robot_context = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.idle_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_dim * slots, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def logits_and_value(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pair_encoder(features)
        robot = self.robot_context(features.mean(dim=2))
        context = robot.unsqueeze(2).expand_as(pair)
        scores = self.score_head(torch.cat([pair, context], dim=-1)).squeeze(-1)
        idle = self.idle_head(robot).squeeze(-1).unsqueeze(-1)
        logits = torch.cat([scores, idle], dim=-1)
        pooled = features.mean(dim=1).reshape(features.shape[0], -1)
        value = self.critic(pooled).squeeze(-1)
        return logits, value

    def distribution(
        self, features: torch.Tensor, valid: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        logits, value = self.logits_and_value(features)
        masked = logits.masked_fill(~valid, -1e9)
        return Categorical(logits=masked), value


@dataclass
class AllocationAction:
    assignments: np.ndarray
    log_probability: float
    value: float


def sample_allocation(
    model: HierarchicalAllocator,
    features: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
    deterministic: bool = False,
) -> AllocationAction:
    with torch.no_grad():
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        valid_tensor = torch.as_tensor(valid, dtype=torch.bool, device=device).unsqueeze(0)
        distribution, value = model.distribution(feature_tensor, valid_tensor)
        actions = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        log_probability = distribution.log_prob(actions).sum(dim=-1)
    selected = actions[0].detach().cpu().numpy().astype(np.int32)
    assignments = np.where(selected < model.slots, selected, -1).astype(np.int32)
    return AllocationAction(
        assignments=assignments,
        log_probability=float(log_probability.item()),
        value=float(value.item()),
    )
