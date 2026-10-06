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


class DeployablePursuit:
    """Classical pure pursuit along the pre-operative route, from imaging estimates only.

    Route = shortest station path on the pre-operative centreline from the station nearest the
    estimated position to the target clot's station, re-planned when the target changes or the estimate
    leaves the route by > 1 mm. A carrot `lookahead_mm` ahead on the route (shorter near junctions so the
    body follows the daughter axis through a take-off) gives the direction; particles detected nearby add
    repulsion and the forecast wait rule; commands shorter than 0.35 stop. Valid for the union-of-tubes
    lumen (junction_model='union'): no knowledge of the simulator's edge or switching rule is used."""
    LOOK, LOOK_NEAR, ZONE = .4, .1, 1.
    CENTRE, SLOW_MM = 1., .3

    def __init__(self, env, sensor, slow=False):
        self.slow = slow; self.body = float(env.config.robot_radius_mm)
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        tr = env.tree; n = tr.n_stations; sc = float(tr.physical_mm_per_unit)
        r, c, w = zip(*[(i, j, d*sc) for i, nb in enumerate(tr.station_graph) for j, d in nb])
        _, self.pred = shortest_path(csr_matrix((w, (r, c)), shape=(n, n)), directed=False, return_predecessors=True)
        self.env, self.sensor = env, sensor
        self.pts = env.transport.points.astype(np.float64)
        self.deg = np.array([len(x) for x in tr.station_graph])
        k = env.num_robots
        self.route = [None]*k; self.goal = np.full(k, -1); self.prog = np.zeros(k, int)
        self.station = np.zeros(k, int)
        self.nominal = np.zeros((k, 3))

    def _plan(self, i, pos, target):
        a = int(np.argmin(np.linalg.norm(self.pts-pos, axis=1))); b = int(self.env.clot_stations[target])
        path = [b]
        while path[-1] != a and self.pred[a, path[-1]] >= 0:
            path.append(int(self.pred[a, path[-1]]))
        self.route[i] = np.array(path[::-1]); self.goal[i] = target; self.prog[i] = 0

    def frames(self, est):
        tr = self.env.tree; s = self.station
        return np.stack((tr.tangents[s], tr.normals[s], tr.binormals[s]), axis=1).astype(np.float64)

    def act(self, targets, est):
        from marl.edge_follower import _safety
        n = self.env.num_robots; out = np.zeros((n, 3))
        for i in range(n):
            t = int(targets[i]); pos = est.pos[i]
            if t < 0 or not est.active[i]:
                self.station[i] = int(np.argmin(np.linalg.norm(self.pts-pos, axis=1)))
                self.nominal[i] = 0.
                continue
            if self.goal[i] != t or self.route[i] is None:
                self._plan(i, pos, t)
            R = self.route[i]; P = self.pts[R]
            w = P[self.prog[i]:self.prog[i]+15]
            k = self.prog[i]+int(np.argmin(np.linalg.norm(w-pos, axis=1)))
            if np.linalg.norm(P[k]-pos) > 1.:
                self._plan(i, pos, t); R = self.route[i]; P = self.pts[R]; k = 0
            self.prog[i] = k; self.station[i] = int(R[k])
            junction = np.any(self.deg[R[max(k-3, 0):k+6]] >= 3)
            look = self.LOOK_NEAR if junction else self.LOOK
            acc, j = 0., k
            while j+1 < len(P) and acc < look:
                acc += float(np.linalg.norm(P[j+1]-P[j])); j += 1
            carrot = self.env.clot_positions_mm[t] if j == len(P)-1 and np.linalg.norm(self.env.clot_positions_mm[t]-pos) < .7 else P[j]
            d = carrot-pos; d = d/max(np.linalg.norm(d), 1e-9)
            F = self.frames(est)[i]
            if self.slow:
                # Narrow lumen: clearance between body and healthy wall is comparable to the localisation
                # error. Steer back to the map axis and slow down so a position error is corrected before
                # it becomes contact (clearance-proportional speed, floor 40 %, above the 0.35 stop deadzone).
                ax, r, rad = self.sensor.map_coordinates(est, i) if est.edge is not None else (pos, 1., 0.)
                clearance = r-self.body
                if rad > 1e-9:
                    d = d+self.CENTRE*max(rad/max(clearance, 1e-6)-.3, 0.)*(ax-pos)/rad
                    d = d/max(np.linalg.norm(d), 1e-9)
                d = d*float(np.clip(clearance/self.SLOW_MM, .4, 1.))
            parts = [(F@r_, F@v) for r_, v in est.particles[i]]
            self.nominal[i] = F@d                  # route direction before the safety layer (for learners)
            out[i] = _safety(F@d, parts)
        return out

    def to_world(self, local, est):
        return np.einsum('nji,nj->ni', self.frames(est), local)*est.active[:, None]


class DeployableObserver:
    """The 111-d fair observation layout (marl.partial_obs) built from DeployableSensor estimates and the
    pre-operative map only, for learned policies. Frenet frames come from the map station matched to the
    estimate; lumen offset/clearance use the healthy map radius; velocity, particles and peers are imaging
    estimates; clots are pre-operative positions with observed cleared status."""
    def __init__(self, env, sensor):
        from marl.partial_obs import PartialObsConfig, PartialObserver
        self.env, self.sensor = env, sensor
        self.base = PartialObserver(env, PartialObsConfig(noise=0.))
        self.prev = np.zeros((env.num_robots, 3))

    def frames(self, est):
        tr = self.env.tree; s = est.station
        return np.stack((tr.tangents[s], tr.normals[s], tr.binormals[s]), axis=1).astype(np.float64)

    def observe(self, est):
        from marl.partial_obs import CLOT0, CLOT_SLOTS, MATE0, MATE_SLOTS, OBS_DIM, PART0, PARTICLE_SLOTS, PATH0, PATH_SLOTS
        env = self.env; n = env.num_robots; F = self.frames(est); R = self.base.cfg.sensing_radius_mm
        obs = np.zeros((n, OBS_DIM), np.float32); ids = np.full((n, CLOT_SLOTS), -1, np.int32)
        _, pts = self.base._graph()
        for i in range(n):
            if not est.active[i]:
                continue
            o = obs[i]; pos = est.pos[i]; loc = lambda v: F[i]@v
            o[0:3] = self.prev[i]; o[3:6] = loc(est.vel[i])/env.config.robot_speed_mm_s
            ax, r, rad = self.sensor.map_coordinates(est, i)
            o[6:9] = loc((pos-ax)/max(r, 1e-9)); o[9] = (r-rad-env.config.robot_radius_mm)/max(r, 1e-9)
            o[10] = max(0., 1-env.elapsed_s/env.config.episode_duration_s)
            rel = env.clot_positions_mm-pos; d = np.linalg.norm(rel, axis=1); alive = est.clot_alive
            for k, j in enumerate(np.argsort(np.where(alive, d, np.inf))[:CLOT_SLOTS]):
                if not alive[j]:
                    continue
                s = CLOT0+6*k; ids[i, k] = j
                o[s] = 1.; o[s+1] = d[j]/10.; o[s+2:s+5] = loc(rel[j]/max(d[j], 1e-9)); o[s+5] = 1.
            info = []
            for e, steer, far, dead in self.base._local_paths(int(est.station[i]), pos):
                sd = loc(pts[steer]-pos); fd = loc(pts[e]-pos)
                info.append((sd/max(np.linalg.norm(sd), 1e-9), fd/max(np.linalg.norm(fd), 1e-9), far, dead,
                             np.log(max(float(self.sensor.healthy[e]), 1e-6))))
            info.sort(key=lambda x: -x[0][0])
            for k, (sd, fd, far, dead, lr) in enumerate(info[:PATH_SLOTS]):
                s = PATH0+10*k; o[s] = 1.; o[s+1:s+4] = sd; o[s+4:s+7] = fd; o[s+7] = far; o[s+8] = float(dead); o[s+9] = lr
            parts = sorted(est.particles[i], key=lambda x: np.linalg.norm(x[0]))[:PARTICLE_SLOTS]
            for k, (rp, rv) in enumerate(parts):
                s = PART0+7*k; o[s] = 1.; o[s+1:s+4] = loc(rp)/R; o[s+4:s+7] = loc(rv)/env.config.robot_speed_mm_s
            vis = np.flatnonzero(est.peers_vis[i] & (np.linalg.norm(est.peers_rel[i], axis=1) < R))
            vis = vis[np.argsort(np.linalg.norm(est.peers_rel[i, vis], axis=1))][:MATE_SLOTS]
            for k, j in enumerate(vis):
                s = MATE0+4*k; o[s] = 1.; o[s+1:s+4] = loc(est.peers_rel[i, j])/R
        return obs, ids

    def record(self, local):
        self.prev = np.asarray(local, np.float64).copy()
