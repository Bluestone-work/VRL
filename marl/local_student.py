"""Fair-observation learned student: parameter-shared MLP over the 111-d local observation + target slot.

Same input as the fair local follower (marl.partial_obs via the cluster sensor adapter) plus the
allocation-A target slot (one-hot over the 4 clot slots). Output: Frenet direction + stop logit.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from marl.partial_obs import OBS_DIM, CLOT_SLOTS


class LocalStudent(nn.Module):
    def __init__(self, hidden=256, layers=3):
        super().__init__()
        dims = [OBS_DIM+CLOT_SLOTS]+[hidden]*layers
        mods = []
        for a, b in zip(dims, dims[1:]):
            mods += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
        self.body = nn.Sequential(*mods)
        self.head = nn.Linear(hidden, 4)
        self.config = dict(hidden=hidden, layers=layers)

    def forward(self, nav, slot):
        onehot = nn.functional.one_hot(slot.clamp(min=0), CLOT_SLOTS).float()*(slot >= 0)[..., None]
        out = self.head(self.body(torch.cat((nav, onehot), -1)))
        return out[..., :3], out[..., 3]


def local_student_action(model, nav, slot, device='cpu'):
    with torch.no_grad():
        d, s = model(torch.as_tensor(nav, dtype=torch.float32, device=device),
                     torch.as_tensor(slot, dtype=torch.long, device=device))
    d = nn.functional.normalize(d, dim=-1); stop = torch.sigmoid(s) > .5
    a = torch.where(stop[..., None], torch.zeros_like(d), d).cpu().numpy().astype(np.float64)
    a[np.asarray(slot) < 0] = 0.
    return a


def save_local(model, path, **meta):
    torch.save(dict(config=model.config, state=model.state_dict(), meta=meta), path)


def load_local(path, device='cpu'):
    p = torch.load(path, map_location=device, weights_only=False)
    m = LocalStudent(**p['config']).to(device); m.load_state_dict(p['state']); m.eval()
    return m
