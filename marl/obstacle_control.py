"""Controllers for environment v3 (microscope-visible obstacles, marl.obstacle_field).

Perception (deployable): cluster estimates from DeployableSensor (or the image chain), obstacle boxes from a
YOLO-like detector, with the cluster's latency; obstacle velocities are inferred by frame-to-frame
nearest-neighbour association (detectors give no identities). Nothing here reads simulator truth.

  APFPursuit      classical baseline: pure pursuit on the pre-operative route + artificial potential field
                  (closest-approach repulsion, wait for an approaching moving obstacle, stop dead-zone)
  token()         per-step observation token (world frame), the analogue of Turbo's state (S19): route
                  target direction/distance, own velocity and previous action, map clearance features and the
                  radial offset vector to the map axis (wall direction, v3.1), the
                  K_OBS nearest obstacles (relative position, size, relative velocity, surface gap) and peers
  TemporalPolicy  shared backbone over a memory window of tokens: 'mlp' (current token only), 'gru', or
                  'transformer' (causal self-attention over time, as Turbo); actor / critic heads
  compose()       residual (u_rule + scale * a) or direct (a) command
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from marl.deployable_sensing import DeployablePursuit
from marl.obstacle_field import detect_obstacles

K_OBS, K_PEER, WINDOW = 6, 2, 16
BASE_DIM = 3+3+3+3+1+5+3
# obstacle slot (v3.4, aligned with Turbo S19: relative position, size, no velocity): rel pos (3), radius,
# surface gap, mask = 6; with obs_vel (ablation) the finite-difference relative velocity is appended = 9
SLOT = {False: 6, True: 9}
token_dim = lambda obs_vel=False: BASE_DIM+SLOT[obs_vel]*K_OBS+4*K_PEER
TOKEN_DIM = token_dim(False)
# APF tuned on training anatomies with tuning seeds 1414000000+ (2 grids, 27 settings x 54 episodes):
# gain 10, range 0.7 mm: Safe 40.7 %, obstacle-collision episodes 20.4 %, wall >= 1 s 44.4 % (stronger
# repulsion trades obstacle hits for wall contact: the classical APF dilemma inside a lumen).
APF_GAIN, APF_RANGE, APF_HORIZON, WAIT_GAP, WAIT_T, DEADZONE = 10., .7, 1., .04, .6, .35
# Wall potential (v3.6): axis direction (towards the map centreline) and clearance from the map
# lumen edge, both from the sensor's map match. WALL_GAIN is in units of the APF push so the two
# terms are comparable; WALL_RANGE activates the term only within 0.35 mm of the wall.
WALL_GAIN, WALL_RANGE = 1.2, .35


class ObstacleTracker:
    """Per-cluster obstacle detections with latency and finite-difference velocities."""
    def __init__(self, field, n, rng, latency, dt, cfg=None):
        from marl.obstacle_field import DetectorConfig
        self.field, self.rng, self.lat, self.dt = field, rng, latency, dt
        self.cfg = cfg if cfg is not None else DetectorConfig()
        self.buf, self.prev = [], [np.zeros((0, 3)) for _ in range(n)]

    def observe(self, est):
        self.buf.append(detect_obstacles(self.field, est.pos, est.active, self.rng, self.cfg))
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


def apf(local, obstacles_local, body, wall_dir=None, wall_gap=None):
    """Artificial potential field in the Frenet frame on top of the route direction.

    v3.6: wall term. The APF's obstacle repulsion points from the obstacle to the cluster; for a
    wall-adherent static obstacle that direction points INTO the wall, and the pursuit term keeps
    pushing along the route, so the cluster gets pinned against the wall for seconds (485 / 535
    wall failures of the tuned APF on the v3 dev set are avoidance-induced; the same scenes
    without avoidance have wall < 1 s). The wall is on the pre-operative map, so the fix stays
    deployable: an axis-attracting potential from the map's radial offset, scaled like the
    obstacle term and only active near the wall (inside WALL_RANGE of the lumen edge)."""
    push = np.zeros(3); wait = False
    for rel, relv, r in obstacles_local:
        t = float(np.clip(-(rel@relv)/max(relv@relv, 1e-12), 0, APF_HORIZON))
        closest = rel+t*relv
        gap = np.linalg.norm(closest)-body-r
        push += np.clip(1-gap/APF_RANGE, 0, 1)**2*(-closest/max(np.linalg.norm(closest), 1e-9))
        moving = np.linalg.norm(relv) > .15
        wait |= moving and gap < WAIT_GAP and t < WAIT_T and t > 0
    if wall_dir is not None and wall_gap is not None and np.isfinite(wall_gap):
        push += WALL_GAIN*np.clip(1-wall_gap/WALL_RANGE, 0, 1)**2*wall_dir
    u = np.clip(local+APF_GAIN*push, -1, 1); u /= max(np.linalg.norm(u), 1.)
    return np.zeros(3) if wait or np.linalg.norm(u) < DEADZONE else u


class APFPursuit(DeployablePursuit):
    def __init__(self, env, sensor, avoid=True, topo=False):
        super().__init__(env, sensor, topo=topo); self.avoid = avoid

    def act(self, targets, est):
        local = super().act(targets, est)                 # route direction (no debris: particles empty)
        if not self.avoid:
            return local
        F = self.frames(est)
        for i in range(len(local)):
            if local[i].any() and est.obstacles[i]:
                obs = [(F[i]@rel, F[i]@relv, r) for rel, relv, r in est.obstacles[i]]
                # wall term from the pre-operative map: axis point, healthy radius, radial offset
                ax, r_map, rad = self.sensor.map_coordinates(est, i)
                off = ax-est.pos[i]; d = float(np.linalg.norm(off))
                wall_dir = off/d if d > 1e-9 else np.zeros(3)
                wall_gap = r_map-rad-self.body
                local[i] = apf(self.nominal[i], obs, self.body, F[i]@wall_dir, wall_gap)
        return local


def token(env, sensor, est, ctl, rule_local, hold, targets, prev_cmd, obs_vel=False):
    """[n, token_dim(obs_vel)] world-frame tokens and the rule command in world coordinates.
    Default (obs_vel=False) gives obstacles as Turbo does: where and how large, not how fast; motion has to be
    inferred from the memory window."""
    n = env.num_robots; F = ctl.frames(est); spd = env.config.robot_speed_mm_s; body = float(env.config.robot_radius_mm)
    T = np.zeros((n, token_dim(obs_vel)), np.float32); rule_w = np.zeros((n, 3)); W = SLOT[obs_vel]
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
        o[18:21] = (est.pos[i]-ax)/max(r, 1e-9)       # wall direction: radial offset from the map axis (v3.1)
        k = 21
        obs = sorted(est.obstacles[i], key=lambda x: np.linalg.norm(x[0])-x[2])[:K_OBS]
        for j, (rel, relv, ro) in enumerate(obs):
            s = k+W*j; o[s:s+3] = rel/3.; o[s+3] = ro/.5; o[s+4] = np.linalg.norm(rel)-body-ro; o[s+5] = 1.
            if obs_vel:
                o[s+6:s+9] = np.clip(relv/spd, -3, 3)
        k += W*K_OBS
        vis = np.flatnonzero(est.peers_vis[i]); vis = vis[np.argsort(np.linalg.norm(est.peers_rel[i, vis], axis=1))][:K_PEER]
        for j, q in enumerate(vis):
            s = k+4*j; o[s:s+3] = est.peers_rel[i, q]/6.; o[s+3] = 1.
    return T, rule_w


class TemporalPolicy(nn.Module):
    def __init__(self, arch='transformer', d=128, layers=2, heads=4, window=WINDOW, dim=TOKEN_DIM):
        super().__init__()
        self.arch, self.window = arch, window
        self.embed = nn.Sequential(nn.Linear(dim, d), nn.LayerNorm(d), nn.GELU())
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
    def __init__(self, n, window=WINDOW, dim=TOKEN_DIM):
        self.seq = np.zeros((n, window, dim), np.float32); self.mask = np.ones((n, window), bool)

    def push(self, T):
        self.seq = np.roll(self.seq, -1, 1); self.mask = np.roll(self.mask, -1, 1)
        self.seq[:, -1] = T; self.mask[:, -1] = False
        return self.seq.copy(), self.mask.copy()


def compose_lateral(rule_w, a, d, lat=.8):
    """v3.5 constrained residual: a[0:2] steer in the plane perpendicular to the route direction d, a[2] scales
    the rule command to 0.5-1.0 of its magnitude. The forward progress the rule makes can be slowed but never
    cancelled or reversed, so the policy cannot trade task progress for shaping reward (the v3.2 and v3.4
    exploits); what remains to learn is how to pass obstacles: sideways, and how fast."""
    a = np.clip(a, -1, 1); u = np.zeros_like(rule_w)
    for i in range(len(a)):
        di = d[i]/max(np.linalg.norm(d[i]), 1e-9)
        h = np.eye(3)[int(np.argmin(np.abs(di)))]
        e1 = np.cross(di, h); e1 /= max(np.linalg.norm(e1), 1e-9); e2 = np.cross(di, e1)
        u[i] = (.75+.25*a[i, 2])*rule_w[i]+lat*(a[i, 0]*e1+a[i, 1]*e2)
    u = u/np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1.)
    u[np.linalg.norm(u, axis=-1) < DEADZONE] = 0.
    return u


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
        self.cfg = c['cfg']; self.vel = bool(self.cfg.get('obs_vel', False)); dim = token_dim(self.vel)
        self.net = TemporalPolicy(self.cfg['arch'], dim=dim); self.net.load_state_dict(c['state']); self.net.eval()
        self.env, self.sensor, self.ctl = env, sensor, ctl
        n = env.num_robots; self.hist = History(n, dim=dim); self.prev = np.zeros((n, 3))

    def act(self, est, rule_local, hold, targets):
        T, rule_w = token(self.env, self.sensor, est, self.ctl, rule_local, hold, targets, self.prev, self.vel)
        seq, mask = self.hist.push(T)
        with torch.no_grad():
            a = self.net.pi(self.net.backbone(torch.as_tensor(seq), torch.as_tensor(mask))).numpy().astype(np.float64)
        live = (np.asarray(targets) >= 0) & est.active
        u = (compose_lateral(rule_w, a, T[:, 0:3]) if self.cfg.get('action') == 'lateral'
             else compose(rule_w, a, self.cfg['residual'], self.cfg['scale'])); u[~live] = 0.
        self.prev = u.copy()
        return np.einsum('nij,nj->ni', self.ctl.frames(est), u)

class DiscreteTemporalPolicy(TemporalPolicy):
    """Temporal deployable policy whose head selects among local teacher options."""
    def __init__(self, arch='transformer', d=128, layers=2, heads=4, window=WINDOW,
                 dim=TOKEN_DIM, actions=8):
        super().__init__(arch=arch, d=d, layers=layers, heads=heads, window=window, dim=dim)
        self.actions = int(actions)
        self.pi = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, self.actions))
        self.value = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 1))
        nn.init.zeros_(self.pi[-1].weight); nn.init.zeros_(self.pi[-1].bias)

    def logits(self, seq, mask):
        return self.pi(self.backbone(seq, mask))

    def dist(self, seq, mask):
        h = self.backbone(seq, mask)
        return torch.distributions.Categorical(logits=self.pi(h)), self.value(h).squeeze(-1)


def save_discrete_checkpoint(path, policy, cfg):
    """Write a tensor-only checkpoint; configuration belongs in the JSON sidecar."""
    torch.save({k: v.detach().cpu() for k, v in policy.state_dict().items()}, path)
    Path(str(path) + '.json').write_text(json.dumps(dict(cfg), sort_keys=True))


def load_discrete_checkpoint(path, cfg):
    """Load a tensor-only checkpoint with explicit architecture and dimension checks."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError('discrete checkpoint must contain tensors only')
    expected = dict(arch=cfg['arch'], dim=int(cfg['dim']), actions=int(cfg.get('actions', 8)))
    if any(k not in cfg for k in ('arch', 'dim')) or expected['actions'] not in (8, 9):
        raise ValueError(f'invalid discrete policy config: {cfg}')
    net = DiscreteTemporalPolicy(cfg['arch'], dim=int(cfg['dim']), actions=expected['actions'])
    net.load_state_dict(state, strict=True); net.eval()
    return net
