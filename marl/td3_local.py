"""Deterministic continuous local-control networks for a TD3 ablation.

The encoder and action interface match the image/history SAC controller.  The
actor only emits a bounded local residual; execution, TPG, WallGuard and the
spacing shield remain outside the learned network.  This module contains no
privileged state or simulator access.
"""
import torch
from torch import nn
from marl.image_flow_sac import Encoder


class TD3Actor(nn.Module):
    action_mode = 'td3_residual'

    def __init__(self, action_dim=3):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Sequential(nn.Linear(160, 128), nn.Tanh(), nn.Linear(128, action_dim))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, hist, mask, ctx, image):
        return torch.tanh(self.head(self.encoder(hist, mask, ctx, image)))


class TD3Critic(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()
        self.encoder = Encoder()
        def q():
            return nn.Sequential(nn.Linear(160 + action_dim, 128), nn.Tanh(),
                                 nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        self.q1, self.q2 = q(), q()

    def forward(self, hist, mask, ctx, image, action):
        z = torch.cat([self.encoder(hist, mask, ctx, image), action], -1)
        return self.q1(z).squeeze(-1), self.q2(z).squeeze(-1)
