"""EXP0090: action-conditioned short-horizon motion predictor (learned belief / world model) - deployable inputs.

Inputs (all deployable; built by DynFeatures from the estimate, the sent-command log and the pre-operative map):
  history  H_HIST steps x HIST_DIM, expressed in the CURRENT local route frame F_t (tangent, normal, binormal):
           measured image velocity (3), sent world command of that step (3, after hold / shield; the complete queue,
           so long latencies are covered), delayed position relative to the newest estimate (3), camera frame age
           (1), TPG hold (1)
  context  CTX_DIM: radial offset from the map axis / healthy radius (3), map wall clearance (1), healthy radius (1),
           route tangent now / +1 mm / +2 mm (9), direction (3) and distance (1) to the target clot, frame age (1)
  plan     H_FUT steps x 3: candidate world command sequence (local frame) from now on
Outputs (local frame, mm): displacement of the TRUE position at t + h*dt relative to the newest (delayed) estimate,
  for HORIZONS = (0, 1, 3, 6, 12) control steps (0 = delay compensation; 1.2 s max), Gaussian mean + log variance.
  Auxiliary head (diagnostic only, never used for control): flow velocity at the cluster (3, robot-speed units) and
  the effective response gain (1). Labels (future truth, flow, gain) come from the simulator and are used ONLY as
  supervised targets; nothing from the simulator enters the input.
Ensembles of independently seeded models give an epistemic spread; predicted variance gives the aleatoric part.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn

H_HIST, H_FUT = 24, 12
HORIZONS = (0, 1, 3, 6, 12)
HIST_DIM, CTX_DIM = 11, 19


def _ahead_tangent(P, k, dist):
    acc, j = 0., k
    while j+1 < len(P) and acc < dist:
        acc += float(np.linalg.norm(P[j+1]-P[j])); j += 1
    d = P[j+1]-P[j] if j+1 < len(P) else (P[j]-P[j-1] if j > 0 else np.zeros(3))
    return d/max(np.linalg.norm(d), 1e-9)


class DynFeatures:
    """Per-episode history buffers of deployable quantities (world frame); call push() once per control step,
    after the command of the step is known (sent_world = the command actually sent at the previous step)."""
    def __init__(self, ep):
        self.n = ep.n; self.buf = [deque(maxlen=H_HIST) for _ in range(ep.n)]

    def push(self, ep, est, hold):
        age = max(0., float(ep.env.elapsed_s)-float(getattr(est, 'frame_time_s', ep.env.elapsed_s)))
        for i in range(self.n):
            self.buf[i].append((est.vel[i].astype(float).copy(), ep.sent_world[i].astype(float).copy(),
                                est.pos[i].astype(float).copy(), age, float(hold[i])))

    def frame(self, ep, est, i):
        return ep.ctl.frames(est)[i]                         # rows: tangent, normal, binormal (world -> local)

    def history(self, ep, est, i, F=None):
        F = self.frame(ep, est, i) if F is None else F
        X = np.zeros((H_HIST, HIST_DIM), np.float32); M = np.ones(H_HIST, bool)
        b = list(self.buf[i]); now = est.pos[i]
        for k, (v, c, p, age, h) in enumerate(b[-H_HIST:]):
            j = H_HIST-len(b[-H_HIST:])+k
            X[j, 0:3] = F@v; X[j, 3:6] = F@c; X[j, 6:9] = F@(p-now); X[j, 9] = age/.3; X[j, 10] = h; M[j] = False
        return X, M

    def context(self, ep, est, tgt, i, F=None):
        F = self.frame(ep, est, i) if F is None else F
        ctl = ep.ctl; o = np.zeros(CTX_DIM, np.float32)
        ax, r, rad = ep.sensor.map_coordinates(est, i); body = float(ep.env.config.robot_radius_mm)
        o[0:3] = F@(est.pos[i]-ax)/max(r, 1e-9); o[3] = np.clip((r-rad-body)/.5, -1, 3); o[4] = r/(r+.5)
        if ctl.route[i] is not None:
            P = ctl.pts[ctl.route[i]]; k = int(ctl.prog[i])
            o[5:8] = F@_ahead_tangent(P, k, 0.); o[8:11] = F@_ahead_tangent(P, k, 1.); o[11:14] = F@_ahead_tangent(P, k, 2.)
        if tgt[i] >= 0:
            d = ep.env.clot_positions_mm[tgt[i]]-est.pos[i]; L = float(np.linalg.norm(d))
            o[14:17] = F@d/max(L, 1e-9); o[17] = min(L/2., 3.)
        o[18] = max(0., float(ep.env.elapsed_s)-float(getattr(est, 'frame_time_s', ep.env.elapsed_s)))/.3
        return o


class Predictor(nn.Module):
    def __init__(self, d=96):
        super().__init__()
        self.gru = nn.GRU(HIST_DIM, d, batch_first=True)
        self.plan = nn.Sequential(nn.Linear(H_FUT*3, d), nn.GELU())
        self.ctx = nn.Sequential(nn.Linear(CTX_DIM, d), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(3*d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        self.mean = nn.Linear(256, len(HORIZONS)*3); self.logvar = nn.Linear(256, len(HORIZONS)*3)
        self.aux = nn.Linear(256, 4)

    def forward(self, hist, mask, ctx, plan):
        x = hist.masked_fill(mask[..., None], 0.)
        h = self.gru(x)[0][:, -1]
        z = self.trunk(torch.cat([h, self.ctx(ctx), self.plan(plan.reshape(len(plan), -1))], -1))
        B = len(z)
        return (self.mean(z).view(B, len(HORIZONS), 3), self.logvar(z).view(B, len(HORIZONS), 3).clamp(-9., 3.),
                self.aux(z))


class Ensemble:
    """Load K predictor checkpoints; predict(hist, mask, ctx, plan) -> mean [B,5,3], aleatoric var, epistemic var."""
    def __init__(self, paths):
        torch.set_num_threads(1)
        self.models = []
        for p in paths:
            c = torch.load(p, map_location='cpu', weights_only=False)
            m = Predictor(c.get('d', 96)); m.load_state_dict(c['state']); m.eval(); self.models.append(m)
            self.norm = c['norm']

    def predict(self, hist, mask, ctx, plan):
        with torch.no_grad():
            H = torch.as_tensor(hist); M = torch.as_tensor(mask); C = torch.as_tensor(ctx); P = torch.as_tensor(plan)
            outs = [m(H, M, C, P) for m in self.models]
        s = self.norm
        mu = np.stack([o[0].numpy() for o in outs])*s; var = np.stack([np.exp(o[1].numpy()) for o in outs])*s**2
        self.last_aux = np.stack([o[2].numpy() for o in outs]).mean(0)   # [B, 4]: flow (local, robot-speed units), gain
        return mu.mean(0), var.mean(0), mu.var(0)


class OnlineAffine:
    """Deployable non-learned predictor (baseline F): v = gain * command * speed + drift, identified online from the
    timestamp-aligned sent-command log and image velocities (marl.observed_motion.ObservedMotion); the delay is
    compensated by integrating the sent commands of the last frame age."""
    def __init__(self, ep):
        from marl.observed_motion import ObservedMotion
        self.om = ObservedMotion(ep.n, dt=float(ep.env.config.control_dt_s)); self.dt = float(ep.env.config.control_dt_s)
        self.speed = float(ep.env.config.robot_speed_mm_s)

    def update(self, ep, est):
        self.om.update(ep.env.elapsed_s, ep.sent_world, getattr(est, 'frame_time_s', ep.env.elapsed_s), est.pos,
                       est.active, est.edge)

    def predict(self, ep, est, i, F, plans_world):
        """plans_world [K, H_FUT, 3] -> mean [K, 5, 3] local, var [K, 5, 3]."""
        g, drift = float(self.om.gain[i]), self.om.drift[i]
        age = max(0., float(ep.env.elapsed_s)-float(getattr(est, 'frame_time_s', ep.env.elapsed_s)))
        lag = (g*ep.sent_world[i]*self.speed+drift)*age            # motion during the image delay (sent command)
        cum = np.cumsum((g*plans_world*self.speed+drift)*self.dt, axis=1)  # [K, H, 3]
        out = np.zeros((len(plans_world), len(HORIZONS), 3))
        for j, h in enumerate(HORIZONS):
            out[:, j] = lag+(cum[:, h-1] if h > 0 else 0.)
        sig2 = (float(self.om.rmse[i])*self.dt)**2*np.array([max(h, 1) for h in HORIZONS])[None, :, None]+1e-4
        return np.einsum('ab,khb->kha', F, out), np.broadcast_to(sig2, out.shape).copy()
