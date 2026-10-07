"""Microscope-visible obstacles in the vessel tree (environment v3, 2026-10-06).

Design follows An et al. 2026 (Turbo, "Autonomous navigation of intelligent microrobotic swarms in unknown
environments"): obstacles are objects a digital microscope resolves clearly and a detector (YOLO there)
reports as bounding boxes; there are static and dynamic ones; their number, static/dynamic ratio, size and
speed are randomised per episode; touching one is a safety violation. Turbo navigates an open plane with
1-3 mm obstacles; our lumen radius is 0.3-13 mm, so sizes scale with the local healthy radius R.

  static   wall-adherent obstacle (residual mural thrombus / plaque not on the pre-operative map): a sphere
           of radius r = U[0.40, 0.70] R attached to the wall, centre at radial distance R - 0.6 r; its tip
           reaches 0.36 R ... -0.12 R from the axis, so larger ones cross the centreline and force a detour.
           The opposite-side gap is >= 0.88 R, passable by a cluster (diameter 0.16 mm) for R >= 0.18 mm.
  where    75 % on the segments of the treatment routes (start -> clots; lesion-adjacent residual thrombus and
           fragments), 25 % anywhere by length. Placement uses the scene, never the policy.
  dynamic  free fragment (embolus) moving along the vessels at U[0.2, 0.8] mm/s with a fixed radial offset,
           random branch choice at junctions, reflection at terminal ends; radius U[0.08, 0.20] mm, capped
           at 0.35 R so it always fits the lumen.
  count    U[12, 24] obstacles, static fraction U[0.1, 0.9] (Turbo, Supplementary Table 4), placed on
           vessel segments by length, >= 1.5 mm from cluster starts and >= 1.0 mm from clots.
Collision: cluster-obstacle centre distance < r_cluster + r_obstacle. Obstacles do not alter the flow or
block bodies physically: a collision is a terminal safety violation (as in Turbo), so post-contact dynamics
do not enter Safe Success.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObstacleConfig:
    count_min: int = 12
    count_max: int = 24
    static_frac_min: float = .1
    static_frac_max: float = .9
    static_r_frac: tuple = (.40, .70)
    on_route_frac: float = .75            # share of obstacles placed on segments of the treatment routes
    dynamic_r_mm: tuple = (.08, .20)
    dynamic_r_cap_frac: float = .35
    dynamic_speed_mm_s: tuple = (.2, .8)
    keep_out_start_mm: float = 1.5
    keep_out_clot_mm: float = 1.0
    near_margin_mm: float = .15


class ObstacleField:
    def __init__(self, env, rng, cfg=ObstacleConfig()):
        self.env, self.rng, self.cfg = env, rng, cfg
        tr = env.tree
        self.pts = env.transport.points.astype(np.float64)
        self.R = np.asarray(env.flow_model.healthy_radius_mm, np.float64)
        self.nb = [[j for j, _ in tr.station_graph[i]] for i in range(tr.n_stations)]
        self.tan, self.nor, self.bin = tr.tangents.astype(float), tr.normals.astype(float), tr.binormals.astype(float)
        self.body = float(env.config.robot_radius_mm)
        n = env.num_robots
        starts = env.positions_mm[:n].astype(float); clots = env.clot_positions_mm.astype(float)
        seg = [(a, b) for a in range(tr.n_stations) for b in self.nb[a] if a < b]
        L = np.array([np.linalg.norm(self.pts[b]-self.pts[a]) for a, b in seg]); w = L/L.sum()
        import scripts.benchmark_multicluster as bm
        _, sp = bm._station_paths(env); on = set()
        for st in np.asarray(env.robot_stations):
            for c in np.asarray(env.clot_stations):
                path = sp(int(st), int(c))
                on |= {(min(u, v), max(u, v)) for u, v in zip(path[:-1], path[1:])}
        w_route = np.array([L[k] if sg in on else 0. for k, sg in enumerate(seg)])
        w_route = w_route/w_route.sum() if w_route.sum() > 0 else w
        total = int(rng.integers(cfg.count_min, cfg.count_max+1))
        n_static = int(round(total*rng.uniform(cfg.static_frac_min, cfg.static_frac_max)))
        self.static_c, self.static_r = [], []
        self.dyn = []                                   # dict(a, b, s, speed, dir, off, r)
        for k in range(total):
            for _ in range(200):
                a, b = seg[int(rng.choice(len(seg), p=w_route if rng.uniform() < cfg.on_route_frac else w))]; s = rng.uniform()
                p = self.pts[a]+s*(self.pts[b]-self.pts[a]); Rl = (1-s)*self.R[a]+s*self.R[b]
                if np.linalg.norm(starts-p, axis=1).min() < cfg.keep_out_start_mm:
                    continue
                if len(clots) and np.linalg.norm(clots-p, axis=1).min() < cfg.keep_out_clot_mm:
                    continue
                ang = rng.uniform(0, 2*np.pi)
                e = np.cos(ang)*self.nor[a]+np.sin(ang)*self.bin[a]
                if k < n_static:
                    r = rng.uniform(*cfg.static_r_frac)*Rl
                    self.static_c.append(p+(Rl-.6*r)*e); self.static_r.append(r)
                else:
                    r = min(rng.uniform(*cfg.dynamic_r_mm), cfg.dynamic_r_cap_frac*Rl)
                    off = rng.uniform(0., .5)*(Rl-r)
                    self.dyn.append(dict(a=a, b=b, s=s, speed=rng.uniform(*cfg.dynamic_speed_mm_s),
                                         ang=ang, off=off, r=r))
                break
        self.static_c = np.asarray(self.static_c, float).reshape(-1, 3); self.static_r = np.asarray(self.static_r, float)
        self.dyn_r = np.array([d['r'] for d in self.dyn], float)
        self._overlap = np.zeros((n, len(self.static_r)+len(self.dyn)), bool)
        self.events = np.zeros(n, int); self.contact_s = np.zeros(n); self.events_static = 0; self.events_dynamic = 0
        self.min_clear = np.full(n, np.inf)

    # ---------- kinematics ----------
    def _dyn_pos(self, d):
        a, b, s = d['a'], d['b'], d['s']
        p = self.pts[a]+s*(self.pts[b]-self.pts[a])
        e = np.cos(d['ang'])*self.nor[a]+np.sin(d['ang'])*self.bin[a]
        return p+d['off']*e

    def advance(self, dt):
        for d in self.dyn:
            left = d['speed']*dt
            for _ in range(20):
                L = max(np.linalg.norm(self.pts[d['b']]-self.pts[d['a']]), 1e-9)
                room = (1-d['s'])*L
                if left < room:
                    d['s'] += left/L; break
                left -= room
                nxt = [j for j in self.nb[d['b']] if j != d['a']]
                if not nxt:                                   # terminal end: reflect
                    d['a'], d['b'], d['s'] = d['b'], d['a'], 0.
                else:
                    d['a'], d['b'], d['s'] = d['b'], int(self.rng.choice(nxt)), 0.
                Rl = self.R[d['a']]                           # keep the fragment inside a narrower vessel
                d['off'] = min(d['off'], max(Rl-d['r'], 0.)*.5)

    def positions(self):
        dp = np.array([self._dyn_pos(d) for d in self.dyn], float).reshape(-1, 3)
        return np.concatenate((self.static_c, dp)), np.concatenate((self.static_r, self.dyn_r)), \
            np.r_[np.zeros(len(self.static_r), bool), np.ones(len(self.dyn), bool)]

    # ---------- safety accounting (simulator side; never a policy input) ----------
    def account(self, cluster_pos, active, dt):
        """Returns per-cluster (new collision events, contact seconds, near-miss seconds)."""
        P, r, dyn = self.positions()
        n = len(cluster_pos)
        if not len(r):
            return np.zeros(n, int), np.zeros(n), np.zeros(n)
        d = np.linalg.norm(cluster_pos[:, None]-P[None], axis=-1)
        gap = d-self.body-r[None]
        ov = (gap < 0) & active[:, None]
        new = ov & ~self._overlap
        self._overlap = ov
        ev = new.sum(1); self.events += ev
        self.events_static += int(new[:, ~dyn].sum()); self.events_dynamic += int(new[:, dyn].sum())
        cs = ov.any(1)*dt; self.contact_s += cs
        near = (np.maximum(1-np.maximum(gap, 0)/self.cfg.near_margin_mm, 0)**2*active[:, None]).sum(1)*dt
        self.min_clear = np.where(active, np.minimum(self.min_clear, gap.min(1)), self.min_clear)
        return ev, cs, near


@dataclass(frozen=True)
class DetectorConfig:
    """YOLO-like obstacle detector on the digital-microscope image (Turbo: YOLOv5 boxes -> mm)."""
    fov_radius_mm: float = 3.             # obstacles reported within this distance of a cluster
    pos_sigma_mm: float = .02             # plus 2.5 % of the obstacle size (Turbo's observation noise)
    size_sigma_frac: float = .025
    miss_prob: float = .03
    pos_size_frac: float = .025           # position noise proportional to obstacle diameter


def detect_obstacles(field, est_pos, active, rng, cfg=DetectorConfig()):
    """Per cluster: list of (centre_world [3], radius_est). No identities and no velocities: like a detector,
    it reports boxes per frame; motion must be inferred from successive frames."""
    P, r, _ = field.positions()
    out = []
    for i in range(len(est_pos)):
        if not active[i] or not len(r):
            out.append([]); continue
        d = np.linalg.norm(P-est_pos[i], axis=1)
        idx = np.flatnonzero((d-r < cfg.fov_radius_mm) & (rng.uniform(size=len(r)) >= cfg.miss_prob))
        out.append([(P[k]+rng.normal(0., cfg.pos_sigma_mm+cfg.pos_size_frac*2*r[k], 3),
                     max(r[k]*(1+rng.normal(0., cfg.size_sigma_frac)), .01)) for k in idx])
    return out
