"""Deployable information model (2026-10-05): what a controller may know at run time.

Modelled on the two Nature Machine Intelligence references (Medany et al. 2025, ultrasound microscope
images; An et al. 2026 "Turbo", camera + YOLO detections): the policy only sees quantities an imaging
system measures, plus pre-operative information. Simulator internals never enter the controller.

Allowed
  pre-operative map      vessel centreline graph, healthy lumen radii, clot positions (registered CTA);
                         allocation A is computed from it once
  imaging (10 Hz)        each cluster's position with Gaussian noise (sigma_mm) and one-frame latency,
                         occasional dropout (last estimate held); clusters that left the field are lost
  derived estimates      velocity by finite differences of the estimates; the map edge a cluster is on by
                         map-matching the estimated position to the pre-operative centreline (continuity
                         prior: stay on the previous edge or an edge sharing its junction unless clearly
                         farther); lumen offset relative to the map centreline and HEALTHY map radius
  local detections       particles inside `particle_radius_mm` of a cluster, noisy positions, velocity by
                         nearest-neighbour association between frames; peers inside `peer_radius_mm`
  clot status            cleared / not cleared (visible on imaging: lumen re-opened)
Forbidden (never read by a controller): true positions, the simulator's edge index, flow field, particle
velocities, the solved (occluded) lumen radii, route tables, and the simulator's edge-switching rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeployableConfig:
    position_sigma_mm: float = .05
    latency_steps: int = 1
    dropout_prob: float = .02
    particle_radius_mm: float = 1.5
    particle_sigma_mm: float = .03
    peer_radius_mm: float = 6.


@dataclass
class Estimate:
    pos: np.ndarray          # [n, 3] estimated positions (mm)
    vel: np.ndarray          # [n, 3] finite-difference velocity (mm/s)
    edge: np.ndarray         # [n] map edge from map-matching
    station: np.ndarray      # [n] nearest map station on that edge (for the local frame)
    active: np.ndarray       # [n] cluster still visible
    particles: list          # per cluster: list of (rel_pos_mm, rel_vel_mm_s) in WORLD frame
    peers_rel: np.ndarray    # [n, n, 3] world-frame relative positions (zero if not visible)
    peers_vis: np.ndarray    # [n, n]
    clot_alive: np.ndarray   # [m]


class DeployableSensor:
    def __init__(self, env, cfg=DeployableConfig(), seed=0):
        self.env, self.cfg = env, cfg
        self.rng = np.random.default_rng(seed)
        t = env.transport
        self.a, self.ab, self.length = t.a, t.ab, t.length
        self.ends, self.groups, self.points = t.ends, t.groups, t.points.astype(np.float64)
        self.healthy = np.asarray(env.flow_model.healthy_radius_mm, np.float64)   # pre-operative lumen
        n = env.num_robots
        self.buffer = []
        self.prev_pos = None; self.prev_edge = None
        self.prev_particles = [np.zeros((0, 3)) for _ in range(n)]
        self.dt = env.config.control_dt_s

    # --- map matching (pre-operative centreline only) ---
    def _project(self, p):
        t = np.clip(((p-self.a)*self.ab).sum(1)/self.length**2, 0, 1)
        axis = self.a+t[:, None]*self.ab
        return np.linalg.norm(p-axis, axis=1), t

    def _match(self, p, prev):
        """Map matching with a topology prior and lumen-normalised distance.

        Candidates are the previous edge and the edges sharing one of its junctions (vessels connect only
        at junctions). Each is scored by the distance to its centreline divided by its healthy lumen radius
        there (a point is plausibly inside a vessel when the ratio is < 1), so a narrow side branch wins
        once the estimate is near its axis even while still inside the wide parent. Hysteresis keeps the
        previous edge unless another candidate is clearly better. If no candidate contains the estimate
        (ratio > 1.5), fall back to the globally best edge (re-localisation)."""
        d, t = self._project(p)
        rad = (1-t)*self.healthy[self.ends[:, 0]]+t*self.healthy[self.ends[:, 1]]
        ratio = d/np.maximum(rad, 1e-6)
        if prev is None:
            return int(np.argmin(ratio))
        g = set(self.groups[self.ends[prev]].tolist())
        cand = [e for e in range(len(d)) if e == prev or (set(self.groups[self.ends[e]].tolist()) & g)]
        best = min(cand, key=lambda e: ratio[e])
        if ratio[best] > 1.5:
            return int(np.argmin(ratio))
        banned = getattr(self, '_banned', {}).get(self._who, set())
        cand = [e for e in cand if e not in banned] or cand
        best = min(cand, key=lambda e: ratio[e])
        if prev not in banned and ratio[prev] < 1. and ratio[prev] <= ratio[best]+.15:
            return prev
        return int(best)

    def report_stall(self, i, edge):
        """Imaging shows cluster i not moving although commanded: the matched edge is probably wrong
        (near a take-off the cluster can sit inside the parent lumen while confined to the daughter).
        Ban that hypothesis for a while so the matcher switches to the next candidate."""
        self._banned = getattr(self, '_banned', {})
        self._banned.setdefault(i, set()).add(int(edge))
        self._ban_age = getattr(self, '_ban_age', {}); self._ban_age[i] = 0

    def observe(self):
        env, c = self.env, self.cfg
        n = env.num_robots
        truth = env.positions_mm[:n].astype(np.float64).copy()
        meas = truth+self.rng.normal(0., c.position_sigma_mm, truth.shape)
        self.buffer.append(meas)
        lagged = self.buffer[max(len(self.buffer)-1-c.latency_steps, 0)]
        if len(self.buffer) > c.latency_steps+2:
            self.buffer.pop(0)
        pos = lagged.copy()
        if self.prev_pos is not None:
            drop = self.rng.uniform(size=n) < c.dropout_prob
            pos[drop] = self.prev_pos[drop]
        active = env.active[:n].copy()       # a cluster that left the imaged field is reported lost
        vel = np.zeros_like(pos) if self.prev_pos is None else (pos-self.prev_pos)/self.dt
        edge = []
        for i in range(n):
            self._who = i
            age = getattr(self, '_ban_age', {})
            if i in age:
                age[i] += 1
                if age[i] > 30:                      # bans expire after 3 s
                    self._banned.pop(i, None); age.pop(i)
            edge.append(self._match(pos[i], None if self.prev_edge is None else int(self.prev_edge[i])))
        edge = np.array(edge)
        station = np.array([int(self.ends[e][int(np.argmin(np.linalg.norm(self.points[self.ends[e]]-pos[i], axis=1)))])
                            for i, e in enumerate(edge)])
        # particles inside the detection radius; velocity by nearest-neighbour association
        pid = np.flatnonzero(env.active[n:])+n
        P = env.positions_mm[pid].astype(np.float64)+self.rng.normal(0., c.particle_sigma_mm, (len(pid), 3))
        parts = []
        for i in range(n):
            if not active[i] or not len(P):
                parts.append([]); continue
            rel = P-pos[i]; near = np.linalg.norm(rel, axis=1) < c.particle_radius_mm
            cur = P[near]; prev = self.prev_particles[i]; out = []
            for q in cur:
                if len(prev):
                    k = int(np.argmin(np.linalg.norm(prev-q, axis=1)))
                    v = (q-prev[k])/self.dt if np.linalg.norm(prev[k]-q) < .5 else np.zeros(3)
                else:
                    v = np.zeros(3)
                out.append((q-pos[i], v-vel[i]))
            self.prev_particles[i] = cur
            parts.append(out)
        rel = pos[None, :, :]-pos[:, None, :]
        vis = (np.linalg.norm(rel, axis=2) <= c.peer_radius_mm) & active[:, None] & active[None, :]
        np.fill_diagonal(vis, False)
        self.prev_pos, self.prev_edge = pos.copy(), edge.copy()
        return Estimate(pos=pos, vel=vel, edge=edge, station=station, active=active, particles=parts,
                        peers_rel=np.where(vis[..., None], rel, 0.), peers_vis=vis, clot_alive=env.masses > 0)

    def map_coordinates(self, est, i):
        """Axis point, healthy map radius and radial offset of estimate i on its matched edge."""
        e = int(est.edge[i]); p = est.pos[i]
        t = float(np.clip(((p-self.a[e])@self.ab[e])/self.length[e]**2, 0, 1))
        axis = self.a[e]+t*self.ab[e]
        r = (1-t)*self.healthy[self.ends[e, 0]]+t*self.healthy[self.ends[e, 1]]
        return axis, r, float(np.linalg.norm(p-axis))
