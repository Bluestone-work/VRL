"""Learned flow/response beliefs; privileged labels are training-only.

Flow is a world-frame vector in robot-speed units. Response is the effective
scalar gain (including lubrication/shape/adhesion, excluding rotation and lag).
Gaussian variance is learned by NLL; calibration requires independent evaluation.
"""
import torch
from torch import nn
from marl.obstacle_control import TemporalPolicy


class MatrixPolicy(TemporalPolicy):
    def __init__(self, arm, **kw):
        self.matrix_arm = arm
        super().__init__(**kw)
        # Construct after the baseline heads so equal seeds give identical baseline initialization.
        self.belief_enabled = arm in ('belief', 'belief_rolling_tpg')
        if arm != 'nav_tf_v3':
            self.dynamics = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 8))
            nn.init.zeros_(self.dynamics[-1].weight)
            nn.init.zeros_(self.dynamics[-1].bias)
            with torch.no_grad(): self.dynamics[-1].bias[3] = 1.
        if self.belief_enabled:
            self.fuse = nn.Linear(136, 128)
            with torch.no_grad():
                self.fuse.weight.zero_(); self.fuse.bias.zero_()
                self.fuse.weight[:, :128].copy_(torch.eye(128))

    def belief(self, h):
        p = self.dynamics(h)
        return p[:, :4], p[:, 4:].clamp(-6., 2.)

    def features(self, seq, mask):
        h = super().backbone(seq, mask)
        if self.belief_enabled:
            mean, logvar = self.belief(h)
            # PPO cannot inflate confidence simply to change residual authority.
            h = self.fuse(torch.cat((h, mean.detach(), logvar.detach()), -1))
        return h

    def dist(self, seq, mask):
        h = super().backbone(seq, mask)
        gate = h.new_ones((len(h), 1))
        if self.belief_enabled:
            mean, logvar = self.belief(h)
            gate = (1./(1.+logvar.detach().exp().mean(-1, keepdim=True))).clamp(.1, 1.)
            h = self.fuse(torch.cat((h, mean.detach(), logvar.detach()), -1))
        return torch.distributions.Normal(self.pi(h)*gate, self.log_std.exp()*gate), self.v(h).squeeze(-1)

    def auxiliary(self, seq, mask, labels, valid):
        mean, logvar = self.belief(super().backbone(seq, mask))
        dims = 4 if self.belief_enabled else 3
        error = (mean[:, :dims]-labels[:, :dims]).square()
        nll = .5*(error*torch.exp(-logvar[:, :dims])+logvar[:, :dims]).mean(-1)
        weight = valid.float(); den = weight.sum().clamp(min=1.)
        return (nll*weight).sum()/den, (error.mean(-1)*weight).sum()/den


def training_dynamics_labels(ep):
    """Read simulation truth only to supervise auxiliary heads, never actor input."""
    import numpy as np
    env, var, n = ep.env, ep.var, ep.n
    tp = env.transport; pos = env.positions_mm[:n]; edges = env.edges[:n]
    sol = dict(env.solution)
    sol['station_inflow_mm3_s'] = sol['station_inflow_mm3_s']*(var.q(env.elapsed_s) if var else 1.)
    flow = tp.velocity_mm_s(pos, edges, sol)/env.config.robot_speed_mm_s
    _, lumen, radial, _ = tp.coordinates(pos, edges, env.solution)
    gap = np.maximum(lumen-radial-env.body_radius[:n], 0.)
    gain = tp.lubrication_floor+(1-tp.lubrication_floor)*np.clip(gap/(4*env.body_radius[:n]), 0., 1.)
    if var:
        gain *= var.gain0*np.exp(var.log_drift)
        gain *= np.clip((lumen-env.body_radius[:n])/.3, .4, 1.)**var.beta
        gain *= np.where(var.touch, 1-var.adhesion, 1.)
    return np.column_stack((flow, gain)).astype(np.float32)
