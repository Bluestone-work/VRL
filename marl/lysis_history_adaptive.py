"""Small deployable history encoder for lysis navigation.

The module consumes only the measured token and the action that was executed.
The auxiliary target is the next measured velocity (token columns 19:22), so
it does not require simulator position, flow or response-gain labels.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HistoryAdaptivePolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 3, history: int = 8,
                 hidden: int = 64, aux_dim: int = 3):
        super().__init__()
        self.obs_dim, self.action_dim, self.history, self.hidden = obs_dim, action_dim, history, hidden
        self.input = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.actor = nn.Sequential(nn.Linear(obs_dim + hidden, 128), nn.Tanh(), nn.Linear(128, action_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim + hidden, 128), nn.Tanh(), nn.Linear(128, 1))
        self.aux = nn.Sequential(nn.Linear(hidden, 64), nn.Tanh(), nn.Linear(64, aux_dim))

    def forward(self, obs, actions, current_obs):
        """Return action mean, value and next-observable-motion prediction.

        Shapes are ``[B,H,D]``, ``[B,H,A]`` and ``[B,D]``. Padding is handled
        by the caller by supplying zero rows after reset.
        """
        x = self.input(torch.cat((obs, actions), dim=-1))
        z = self.gru(x)[0][:, -1]
        fused = torch.cat((current_obs, z), dim=-1)
        return self.actor(fused), self.critic(fused).squeeze(-1), self.aux(z)


def observable_motion_target(obs, next_obs, velocity_slice=slice(19, 22), scale=1.0):
    """Construct a normalized target entirely from consecutive observations."""
    return (next_obs[..., velocity_slice] - obs[..., velocity_slice]) / float(scale)


def auxiliary_loss(prediction, target, weight=0.05):
    """Bounded auxiliary contribution; diagnostics expose its unweighted MSE."""
    raw = F.smooth_l1_loss(prediction, target)
    return weight * raw, {"aux_loss": float(raw.detach()), "aux_weighted": float((weight * raw).detach())}
