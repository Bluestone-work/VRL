"""Matched original temporal policy with optional short GRU context."""
import torch
from torch import nn
from marl.obstacle_control import TemporalPolicy


class FlowPolicy(TemporalPolicy):
    def __init__(self, history=False, **kw):
        super().__init__(**kw)
        self.history_enabled = history
        dim = kw['dim']
        if history:
            self.context = nn.GRU(dim, 32, batch_first=True)
            self.fuse = nn.Linear(160, 128)
        self.motion = nn.Sequential(nn.Linear(131, 64), nn.Tanh(), nn.Linear(64, 3))

    def backbone(self, seq, mask):
        h = super().backbone(seq, mask)
        if self.history_enabled:
            z = seq.new_zeros((1, len(seq), 32))
            for k in range(max(0, seq.shape[1]-8), seq.shape[1]):
                _, candidate = self.context(seq[:, k:k+1], z)
                z = torch.where(mask[:, k][None, :, None], z, candidate)
            h = self.fuse(torch.cat((h, z[0]), -1))
        return h

    def predict_motion(self, seq, mask, executed):
        return self.motion(torch.cat((self.backbone(seq, mask), executed), -1))
