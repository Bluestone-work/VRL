"""Controllers for environment v3 (microscope-visible obstacles, marl.obstacle_field).

Perception (deployable): cluster estimates from DeployableSensor (or the image chain), obstacle boxes from a
YOLO-like detector, with the cluster's latency; obstacle velocities are inferred by frame-to-frame
nearest-neighbour association (detectors give no identities). Nothing here reads simulator truth.

  APFPursuit      classical baseline: pure pursuit on the pre-operative route + artificial potential field
                  (closest-approach repulsion, wait for an approaching moving obstacle, stop dead-zone)
  token()         per-step observation token (world frame), the analogue of Turbo's state (S19): route
                  target direction/distance, own velocity and previous action, map clearance features, the
                  K_OBS nearest obstacles (relative position, size, relative velocity, surface gap) and peers
  TemporalPolicy  shared backbone over a memory window of tokens: 'mlp' (current token only), 'gru', or
                  'transformer' (causal self-attention over time, as Turbo); actor / critic heads
  compose()       residual (u_rule + scale * a) or direct (a) command
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from marl.deployable_sensing import DeployablePursuit
from marl.obstacle_field import detect_obstacles

K_OBS, K_PEER, WINDOW = 6, 2, 16
TOKEN_DIM = 3+3+3+3+1+5+9*K_OBS+4*K_PEER
# APF tuned on training anatomies with tuning seeds 1414000000+ (2 grids, 27 settings x 54 episodes):
# gain 10, range 0.7 mm: Safe 40.7 %, obstacle-collision episodes 20.4 %, wall >= 1 s 44.4 % (stronger
# repulsion trades obstacle hits for wall contact: the classical APF dilemma inside a lumen).
APF_GAIN, APF_RANGE, APF_HORIZON, WAIT_GAP, WAIT_T, DEADZONE = 10., .7, 1., .04, .6, .35


class ObstacleTracker:
    """Per-cluster obstacle detections with latency and finite-difference velocities."""
    def __init__(self, field, n, rng, latency, dt):
        self.field, self.rng, self.lat, self.dt = field, rng, latency, dt
        self.buf, self.prev = [], [np.zeros((0, 3)) for _ in range(n)]

    def observe(self, est):
        self.buf.append(detect_obstacles(self.field, est.pos, est.active, self.rng))
        dets = self.buf[max(len(self.buf)-1-self.lat, 0)]
        if len(self.buf) > self.lat+2:
            self.buf.pop(0)
        out = []
        for i, D in enumerate(dets):
            cur = np.array([c for c, _ in D]).reshape(-1, 3); prev = self.prev[i]; L = []
            for c, r in D:
                v = np.zeros(3)
                if len(prev):
                    k = int(np.argmin(np.linalg.norm(prev-c, axis=1)))
                    if np.linalg.norm(prev[k]-c) < .25:
                        v = (c-prev[k])/self.dt
                L.append((c-est.pos[i], v-est.vel[i], r))
            self.prev[i] = cur; out.append(L)
        est.obstacles = out
        return est


def apf(local, obstacles_local, body):
    """Artificial potential field in the Frenet frame on top of the route direction."""
    push = np.zeros(3); wait = False
    for rel, relv, r in obstacles_local:
        t = float(np.clip(-(rel@relv)/max(relv@relv, 1e-12), 0, APF_HORIZON))
        closest = rel+t*relv
        gap = np.linalg.norm(closest)-body-r
        push += np.clip(1-gap/APF_RANGE, 0, 1)**2*(-closest/max(np.linalg.norm(closest), 1e-9))
        moving = np.linalg.norm(relv) > .15
        wait |= moving and gap < WAIT_GAP and t < WAIT_T and t > 0
    u = np.clip(local+APF_GAIN*push, -1, 1); u /= max(np.linalg.norm(u), 1.)
    return np.zeros(3) if wait or np.linalg.norm(u) < DEADZONE else u


class APFPursuit(DeployablePursuit):
    def __init__(self, env, sensor, avoid=True):
        super().__init__(env, sensor); self.avoid = avoid

    def act(self, targets, est):
        local = super().act(targets, est)                 # route direction (no debris: particles empty)
        if not self.avoid:
            return local
        F = self.frames(est)
        for i in range(len(local)):
            if local[i].any() and est.obstacles[i]:
                obs = [(F[i]@rel, F[i]@relv, r) for rel, relv, r in est.obstacles[i]]
                local[i] = apf(self.nominal[i], obs, self.body)
        return local


def token(env, sensor, est, ctl, rule_local, hold, targets, prev_cmd):
    """[n, TOKEN_DIM] world-frame tokens and the rule command in world coordinates."""
    n = env.num_robots; F = ctl.frames(est); spd = env.config.robot_speed_mm_s; body = float(env.config.robot_radius_mm)
    T = np.zeros((n, TOKEN_DIM), np.float32); rule_w = np.zeros((n, 3))
    for i in range(n):
        if not est.active[i] or targets[i] < 0:
            continue
        o = T[i]; rule_w[i] = F[i].T@rule_local[i]
        o[0:3] = F[i].T@ctl.nominal[i]; o[3:6] = rule_w[i]; o[6:9] = est.vel[i]/spd; o[9:12] = prev_cmd[i]
        o[12] = np.linalg.norm(env.clot_positions_mm[targets[i]]-est.pos[i])/10.
        ax, r, rad = sensor.map_coordinates(est, i)
        R = ctl.route[i]; pk = ctl.prog[i]
        o[13] = rad/max(r, 1e-9); o[14] = (r-rad-body)/max(r, 1e-9)
        o[15] = float(R is not None and np.any(ctl.deg[R[max(pk-3, 0):pk+6]] >= 3))
        o[16] = float(hold[i]); o[17] = max(0., 1-env.elapsed_s/env.config.episode_duration_s)
        k = 18
        obs = sorted(est.obstacles[i], key=lambda x: np.linalg.norm(x[0])-x[2])[:K_OBS]
        for j, (rel, relv, ro) in enumerate(obs):
            s = k+9*j; o[s:s+3] = rel/3.; o[s+3] = ro/.5; o[s+4:s+7] = np.clip(relv/spd, -3, 3)
            o[s+7] = (np.linalg.norm(rel)-body-ro); o[s+8] = 1.
        k += 9*K_OBS
        vis = np.flatnonzero(est.peers_vis[i]); vis = vis[np.argsort(np.linalg.norm(est.peers_rel[i, vis], axis=1))][:K_PEER]
        for j, q in enumerate(vis):
            s = k+4*j; o[s:s+3] = est.peers_rel[i, q]/6.; o[s+3] = 1.
    return T, rule_w


class TemporalPolicy(nn.Module):
    def __init__(self, arch='transformer', d=128, layers=2, heads=4, window=WINDOW):
        super().__init__()
        self.arch, self.window = arch, window
        self.embed = nn.Sequential(nn.Linear(TOKEN_DIM, d), nn.LayerNorm(d), nn.GELU())
        if arch == 'transformer':
            self.pos = nn.Parameter(torch.zeros(window, d)); nn.init.normal_(self.pos, std=.02)
            self.pad = nn.Parameter(torch.zeros(1, 1, d))
            layer = nn.TransformerEncoderLayer(d, heads, 2*d, dropout=0., batch_first=True, norm_first=True, activation='gelu')
            self.core = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.register_buffer('causal', torch.triu(torch.full((window, window), float('-inf')), 1))
        elif arch == 'gru':
            self.core = nn.GRU(d, d, batch_first=True)
        else:
            self.core = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.GELU())

        def head(out):
            return nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, out))
        self.pi, self.v = head(3), head(1)
        nn.init.zeros_(self.pi[-1].weight); nn.init.zeros_(self.pi[-1].bias)
        self.log_std = nn.Parameter(torch.full((3,), -1.0))
        self.last_attn = None

    def backbone(self, seq, mask):
        """seq [B, W, TOKEN_DIM] oldest..newest; mask [B, W] True = padded (before the episode start)."""
        x = self.embed(seq)
        if self.arch == 'transformer':
            # steps before the episode start are a learned pad token (no key-padding mask: rows whose keys are
            # all masked give NaN that would propagate through the next layer)
            x = torch.where(mask[..., None], self.pad.expand_as(x), x)
            h = self.core(x+self.pos, mask=self.causal)
            return h[:, -1]
        if self.arch == 'gru':
            x = x.masked_fill(mask[..., None], 0.)
            return self.core(x)[0][:, -1]
        return self.core(x[:, -1])

    def dist(self, seq, mask):
        h = self.backbone(seq, mask)
        return torch.distributions.Normal(self.pi(h), self.log_std.exp()), self.v(h).squeeze(-1)


class History:
    def __init__(self, n, window=WINDOW):
        self.seq = np.zeros((n, window, TOKEN_DIM), np.float32); self.mask = np.ones((n, window), bool)

    def push(self, T):
        self.seq = np.roll(self.seq, -1, 1); self.mask = np.roll(self.mask, -1, 1)
        self.seq[:, -1] = T; self.mask[:, -1] = False
        return self.seq.copy(), self.mask.copy()


def compose(rule_w, a, residual=True, scale=1.):
    u = rule_w+scale*np.clip(a, -1, 1) if residual else np.clip(a, -1, 1)
    u = u/np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1.)
    u[np.linalg.norm(u, axis=-1) < DEADZONE] = 0.
    return u


class DRLController:
    """Deterministic evaluation wrapper around a trained TemporalPolicy checkpoint."""
    def __init__(self, env, sensor, ctl, ckpt):
        torch.set_num_threads(1)
        c = torch.load(ckpt, map_location='cpu', weights_only=False)
        self.cfg = c['cfg']; self.net = TemporalPolicy(self.cfg['arch']); self.net.load_state_dict(c['state']); self.net.eval()
        self.env, self.sensor, self.ctl = env, sensor, ctl
        n = env.num_robots; self.hist = History(n); self.prev = np.zeros((n, 3))

    def act(self, est, rule_local, hold, targets):
        T, rule_w = token(self.env, self.sensor, est, self.ctl, rule_local, hold, targets, self.prev)
        seq, mask = self.hist.push(T)
        with torch.no_grad():
            a = self.net.pi(self.net.backbone(torch.as_tensor(seq), torch.as_tensor(mask))).numpy().astype(np.float64)
        live = (np.asarray(targets) >= 0) & est.active
        u = compose(rule_w, a, self.cfg['residual'], self.cfg['scale']); u[~live] = 0.
        self.prev = u.copy()
        return np.einsum('nij,nj->ni', self.ctl.frames(est), u)
