"""Image-conditioned residual DRL controller (IR-PPO) for the deployable pipeline.

Motivation (measured, benchmark v2 + image probe): the hand-tuned local safety layer of the deployable
pursuit (particle repulsion gain 6, wait if clearance < 0.3 mm within 0.5 s) is the dominant remaining
failure: particle contact in 7.9 / 10.7 / 11.4 % of episodes (N = 1/2/3), and with image-based perception
particle collisions rise further (+0.39 events/episode at N=3) because a particle merging into the
cluster's blob drops out of the detection list exactly when it is closest. A detection list cannot express
"something is there but I cannot resolve it"; the pixels can. Learning-from-scratch is not an option
(pure PPO: Safe <= 0.7 %), so the policy is a residual on the rule (Johannink et al. 2019; Silver et al.
2018): at initialisation its output is exactly the rule, and PPO only learns corrections.

Hierarchy (each layer uses deployable information only)
  pre-operative   allocation A (makespan-optimal tours on the CTA map)
  coordination    TPG schedule: hold before conflict zones (hard constraint, never overridden)
  local control   pursuit rule u_rule  +  learned residual  du = 0.6 * pi(image crops, detections, map)
  safety filter   spacing shield (N > 1), stop dead-zone

Observation per cluster (all from imaging + pre-operative map; world frame, so it is aligned with the image)
  image   [2, 32, 32] background-subtracted crops of the top (x-y) and side (x-z) camera views, 1.6 mm field
          around the cluster's own tracking estimate, with the same latency as the detections
  vector  [VEC_DIM] nominal route direction, rule command, estimated velocity, 4 nearest detected particles
          (relative position / velocity, mask), 2 nearest peers (relative position, mask), map features
          (radial offset / healthy radius, wall clearance, near-junction flag, distance to target, TPG hold,
          time fraction)
Network: CNN image encoder || MLP vector encoder -> fused trunk -> Gaussian actor (zero-initialised mean)
and value head. Parameters shared across clusters (decentralised execution).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from marl.image_sensing import CROP_PX

PARTS, PEERS = 4, 2
VEC_DIM = 3+3+3+7*PARTS+4*PEERS+6
RES_SCALE, DEADZONE = .6, .35


class IRPolicy(nn.Module):
    def __init__(self, hidden=256, use_image=True):
        super().__init__()
        self.use_image = use_image                 # ablation: False = detections + map only (no pixels)
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.GELU(),          # 32 -> 16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),         # 16 -> 8
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GELU(),         # 8 -> 4
            nn.Flatten(), nn.Linear(32*4*4, 128), nn.LayerNorm(128), nn.GELU())
        self.vec = nn.Sequential(nn.Linear(VEC_DIM, 128), nn.LayerNorm(128), nn.GELU())

        def trunk():
            return nn.Sequential(nn.Linear(256, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.pi_trunk, self.v_trunk = trunk(), trunk()
        self.mu, self.v = nn.Linear(hidden, 3), nn.Linear(hidden, 1)
        nn.init.zeros_(self.mu.weight); nn.init.zeros_(self.mu.bias)     # residual starts at exactly zero
        self.log_std = nn.Parameter(torch.full((3,), -1.2))

    def features(self, img, vec):
        im = self.cnn(img*2.) if self.use_image else torch.zeros(vec.shape[0], 128, device=vec.device)
        return torch.cat((im, self.vec(vec)), -1)

    def dist(self, img, vec):
        f = self.features(img, vec)
        return torch.distributions.Normal(self.mu(self.pi_trunk(f)), self.log_std.exp()), self.v(self.v_trunk(f)).squeeze(-1)


def observe(env, sensor, est, ctl, rule_local, hold, targets):
    """(img [n,2,32,32], vec [n,VEC_DIM], rule_world [n,3]) from imaging estimates and the map only."""
    n = env.num_robots; F = ctl.frames(est)
    to_w = lambda i, v: F[i].T@v
    vec = np.zeros((n, VEC_DIM), np.float32); rule_w = np.zeros((n, 3))
    for i in range(n):
        if not est.active[i] or targets[i] < 0:
            continue
        o = vec[i]; k = 0
        rule_w[i] = to_w(i, rule_local[i])
        o[0:3] = to_w(i, ctl.nominal[i]); o[3:6] = rule_w[i]; o[6:9] = est.vel[i]/env.config.robot_speed_mm_s
        k = 9
        P = sorted(est.particles[i], key=lambda x: np.linalg.norm(x[0]))[:PARTS]
        for j, (rp, rv) in enumerate(P):
            s = k+7*j; o[s:s+3] = rp/1.5; o[s+3:s+6] = np.clip(rv/env.config.robot_speed_mm_s, -3, 3); o[s+6] = 1.
        k += 7*PARTS
        vis = np.flatnonzero(est.peers_vis[i])
        vis = vis[np.argsort(np.linalg.norm(est.peers_rel[i, vis], axis=1))][:PEERS]
        for j, q in enumerate(vis):
            s = k+4*j; o[s:s+3] = est.peers_rel[i, q]/6.; o[s+3] = 1.
        k += 4*PEERS
        ax, r, rad = sensor.map_coordinates(est, i)
        R = ctl.route[i]; pk = ctl.prog[i]
        junction = R is not None and bool(np.any(ctl.deg[R[max(pk-3, 0):pk+6]] >= 3))
        o[k] = rad/max(r, 1e-9); o[k+1] = (r-rad-env.config.robot_radius_mm)/max(r, 1e-9)
        o[k+2] = float(junction); o[k+3] = np.linalg.norm(env.clot_positions_mm[targets[i]]-est.pos[i])/10.
        o[k+4] = float(hold[i]); o[k+5] = max(0., 1-env.elapsed_s/env.config.episode_duration_s)
    return sensor.crops.copy(), vec, rule_w


def compose(rule_w, a, scale=RES_SCALE):
    """Executed world command: rule + bounded residual, unit-norm clip, stop dead-zone. With scale >= 1 the
    residual can cancel or reverse the rule entirely (the learned layer has full authority locally)."""
    u = rule_w+scale*np.clip(a, -1, 1)
    nrm = np.linalg.norm(u, axis=-1, keepdims=True)
    u = u/np.maximum(nrm, 1.)
    u[np.linalg.norm(u, axis=-1) < DEADZONE] = 0.
    return u


class IRController:
    """Deterministic evaluation wrapper (mean residual)."""
    def __init__(self, env, sensor, ctl, ckpt):
        torch.set_num_threads(1)
        self.env, self.sensor, self.ctl = env, sensor, ctl
        ck = torch.load(ckpt, map_location='cpu')
        self.net = IRPolicy(use_image=ck.get('use_image', True)); self.net.load_state_dict(ck['state']); self.net.eval()
        self.scale = float(ck.get('res_scale', RES_SCALE))

    def act(self, est, rule_local, hold, targets):
        img, vec, rule_w = observe(self.env, self.sensor, est, self.ctl, rule_local, hold, targets)
        with torch.no_grad():
            f = self.net.features(torch.as_tensor(img), torch.as_tensor(vec))
            a = self.net.mu(self.net.pi_trunk(f)).numpy().astype(np.float64)
        live = (np.asarray(targets) >= 0) & est.active
        a[~live] = 0.
        u = compose(rule_w, a, self.scale); u[~live] = 0.
        F = self.ctl.frames(est)
        return np.einsum('nij,nj->ni', F, u)                 # back to the local frame for gate/shield
