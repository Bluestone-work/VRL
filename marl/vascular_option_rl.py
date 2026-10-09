"""Vascular memory option RL, EXP0060. No simulator state in the policy API.

The actor chooses a local feedback option every 0.5 s, while route feedback and
the common measured-state coordination/filter run at 10 Hz. This is NOT task
reallocation. All options, including fixed-option baselines, share this API.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from marl.deployable_sensing import DeployablePursuit
from marl.edge_follower import _safety

OBS_DIM = 66
HISTORY = 8
OPTIONS = ('pursuit', 'speed_055', 'speed_080', 'short_turn', 'careful_turn',
           'long_look', 'repulsion_half', 'repulsion_150', 'wait')


class OptionPolicy(nn.Module):
    def __init__(self, memory=True, central=True):
        super().__init__()
        self.memory, self.central = memory, central
        self.encoder = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.LayerNorm(64), nn.Tanh())
        self.gru = nn.GRU(64, 64, batch_first=True) if memory else None
        self.actor = nn.Linear(64, len(OPTIONS))
        self.critic = nn.Sequential(nn.Linear(128 if central else 64, 128), nn.Tanh(), nn.Linear(128, 1))
        # An explicitly declared navigation prior, not learned imitation.
        nn.init.zeros_(self.actor.weight); nn.init.zeros_(self.actor.bias)
        with torch.no_grad():
            self.actor.bias[0] = 2.

    def forward(self, history, active):
        # history: [batch, padded_agents=3, history=8, features=66]
        b, n, h, _ = history.shape
        z = self.encoder(history.reshape(b*n, h, OBS_DIM))
        z = self.gru(z)[0][:, -1] if self.memory else z[:, -1]
        z = z.reshape(b, n, 64)
        logits = self.actor(z)
        if self.central:
            pooled = (z*active[..., None]).sum(1)/active.sum(1, keepdim=True).clamp(min=1)
            z = torch.cat((z, pooled[:, None].expand(-1, n, -1)), -1)
        return torch.distributions.Categorical(logits=logits), self.critic(z).squeeze(-1)


class OptionController(DeployablePursuit):
    """Fixed preoperative map + measured Estimate -> candidate local commands.

The env supplied here is a read-only preoperative snapshot, not the simulator.
No masses, true current positions/edges/velocities/flow are available.
"""
    def prepare(self, targets, est, hold, elapsed, horizon):
        base = super().act(targets, est)
        n = len(targets); f = self.frames(est)
        candidates = np.repeat(base[:, None], len(OPTIONS), axis=1)
        candidates[:, 1] *= .55; candidates[:, 2] *= .8
        candidates[:, 8] = 0.
        x = np.zeros((n, OBS_DIM), np.float32)
        for i, t in enumerate(targets):
            if t < 0 or not est.active[i]:
                candidates[i] = 0.; continue
            pos = est.pos[i]; R = self.route[i]; P = self.pts[R]; k = self.prog[i]
            parts = [(f[i]@r, f[i]@v) for r, v in est.particles[i]]
            for option, look, speed in ((3, .1, 1.), (4, .2, .65), (5, .8, 1.)):
                acc, j = 0., k
                while j+1 < len(P) and acc < look:
                    acc += np.linalg.norm(P[j+1]-P[j]); j += 1
                goal = self.env.clot_positions_mm[t]
                carrot = goal if j == len(P)-1 and np.linalg.norm(goal-pos) < .7 else P[j]
                direction = f[i]@(carrot-pos)/max(np.linalg.norm(carrot-pos), 1e-9)
                candidates[i, option] = _safety(speed*direction, parts)
            # Change repulsion only, keep the same closest-approach wait rule.
            from marl.fair_reactive import _closest
            push = np.zeros(3); waiting = False
            for rp, rv in parts:
                tc, closest, clear = _closest(rp, rv)
                push += np.clip(1-clear, 0, 1)**2*(-closest/max(np.linalg.norm(closest), 1e-9))
                waiting |= clear < .3 and tc < .5
            for option, gain in ((6, -3.), (7, 3.)):
                a = np.clip(base[i]+gain*push, -1, 1)
                a /= max(np.linalg.norm(a), 1.)
                candidates[i, option] = 0. if waiting or np.linalg.norm(a) < .35 else a
            axis, radius, radial = self.sensor.map_coordinates(est, i)
            rel = self.env.clot_positions_mm[t]-pos
            x[i, :3] = base[i]; x[i, 3:6] = f[i]@est.vel[i]
            x[i, 6:9] = f[i]@rel/10.
            x[i, 9:12] = f[i]@(pos-axis)/max(radius, 1e-6)
            x[i, 12:18] = [(radius-radial-self.body)/max(radius, 1e-6), radius,
                           np.linalg.norm(rel)/10., elapsed/horizon, float(hold[i]), float(est.clot_alive[t])]
            # Route curvature and local progress are computed only from the preop map.
            j = min(k+4, len(P)-1)
            x[i, 18:21] = f[i]@(P[j]-P[k]); x[i, 21] = k/max(len(P)-1, 1)
            x[i, 22] = np.linalg.norm(est.vel[i]); x[i, 23] = len(est.particles[i])/32.
            for q, (rp, rv) in enumerate(sorted(parts, key=lambda p: np.linalg.norm(p[0]))[:4]):
                s = 24+q*7; x[i, s:s+7] = np.r_[1., rp/1.5, rv]
            js = np.flatnonzero(est.peers_vis[i])
            js = sorted(js, key=lambda j: np.linalg.norm(est.peers_rel[i, j]))[:2]
            for q, j in enumerate(js):
                s = 52+q*7
                x[i, s:s+7] = np.r_[1., f[i]@est.peers_rel[i, j]/6., f[i]@(est.vel[j]-est.vel[i])]
        # Common coordinator is non-negotiable for every option.
        candidates[hold] = 0.
        return np.clip(x, -10, 10), candidates
