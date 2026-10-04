"""Learned branch (frontier) selector for fair local navigation.

Scores each visible frontier from fair features only (marl.edge_follower.LocalFollower.FEATURES);
trained with a privileged label (the frontier geodesically closest to the target on the known map),
so the map is a training-time teacher and never an input. Softmax over the candidates of one decision.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


class FrontierSelector(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.register_buffer('mu', torch.zeros(n_feat)); self.register_buffer('sd', torch.ones(n_feat))

    def forward(self, x):
        return self.net((x-self.mu)/self.sd).squeeze(-1)


def make_scorer(path):
    p = torch.load(path, map_location='cpu', weights_only=False)
    m = FrontierSelector(p['n_feat']); m.load_state_dict(p['state']); m.eval()
    def score(F):
        with torch.no_grad():
            return m(torch.as_tensor(np.asarray(F), dtype=torch.float32)).numpy()
    return score
