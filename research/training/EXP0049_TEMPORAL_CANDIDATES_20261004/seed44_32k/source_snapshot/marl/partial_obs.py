"""Fair partial observation (Turbo-style) shared by learned and hand-written controllers.

Modelled on An et al., Nature Machine Intelligence 2026 ("Turbo"): previous action,
noisy ego state, straight-line goal distance and bearing, and a few nearest obstacles
inside a sensing radius, with ~2.5 % multiplicative observation noise. No global map,
no route, no geodesic distance, no allocation. Adapted to vessels:

  * goals are the clots: straight-line (Euclidean) vector to every clot, as Turbo's
    goal distance/bearing (clot positions are known from pre-operative imaging)
  * obstacles are the particles inside the sensing radius, plus the vessel wall
  * local vessel geometry inside the field of view (the lumen visible around the
    robot): every centreline path leaving the robot's position, truncated at the
    view radius, with its steering direction, far-end direction, whether it ends
    in a dead end inside the view, and its lumen radius

Per-robot vector (OBS_DIM = 111), every vector in the robot's Frenet frame:
  0:3    previous action
  3:6    own velocity / robot speed                              (noisy)
  6:9    offset from the centreline / lumen radius               (noisy)
  9      wall clearance / lumen radius                           (noisy)
  10     remaining time fraction
  11:35  4 clot slots, nearest first: alive, distance/10 mm, direction (3), mass fraction
  35:75  4 local path slots: valid, steer direction (3), far direction (3),
         far geodesic distance / view radius, dead end inside view, log lumen radius
  75:103 4 particle slots inside the sensing radius, nearest first:
         valid, relative position / radius (3), relative velocity / robot speed (3)
  103:111 2 teammate slots inside the sensing radius: valid, relative position / radius (3)

`tests/test_partial_obs.py` makes every route/allocation accessor raise while this is built.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

OBS_DIM = 111
CLOT_SLOTS, PATH_SLOTS, PARTICLE_SLOTS, MATE_SLOTS = 4, 4, 4, 2
CLOT0, PATH0, PART0, MATE0 = 11, 35, 75, 103


@dataclass(frozen=True)
class PartialObsConfig:
    view_radius_mm: float = 4.0        # lumen visible around the robot (geodesic, along the centreline)
    sensing_radius_mm: float = 1.5     # particles / teammates (Euclidean)
    steer_lookahead_mm: float = .06    # first path station beyond this distance gives the steering direction
    noise: float = .025                # multiplicative Gaussian, as in Turbo
    position_noise_mm: float = 0.      # additive, for stress tests


class PartialObserver:
    """Builds the fair observation from geometry visible near each robot and physical state."""

    def __init__(self, env, config=PartialObsConfig(), seed=0):
        self.env, self.cfg = env, config
        self.rng = np.random.default_rng(seed)
        self.prev_action = np.zeros((env.num_robots, 3))
        self._tree_key = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.prev_action = np.zeros((self.env.num_robots, 3))

    def record_action(self, local_action):
        self.prev_action = np.asarray(local_action, np.float64).copy()

    def _graph(self):
        tree = self.env.tree
        if self._tree_key != id(tree):
            scale = float(tree.physical_mm_per_unit)
            self._adj = [[(j, w*scale) for j, w in nb] for nb in tree.station_graph]
            self._points = self.env.transport.points.astype(np.float64)
            self._tree_key = id(tree)
        return self._adj, self._points

    def _noisy(self, x):
        if self.cfg.noise <= 0:
            return x
        return x*self.rng.normal(1., self.cfg.noise, np.shape(x))

    def _local_paths(self, start, pos):
        """Centreline paths leaving `start` inside the view radius (Dijkstra tree, frontier/leaf ends)."""
        adj, pts = self._graph()
        R = self.cfg.view_radius_mm
        dist, parent = {start: 0.}, {start: -1}
        heap = [(0., start)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d >= R:
                continue
            for v, w in adj[u]:
                nd = d+w
                if nd < dist.get(v, np.inf):
                    dist[v], parent[v] = nd, u
                    heapq.heappush(heap, (nd, v))
        children = {}
        for v, p in parent.items():
            if p >= 0:
                children.setdefault(p, []).append(v)
        ends = [v for v in parent if v != start and v not in children]
        paths = []
        for e in ends:
            chain = [e]
            while parent[chain[-1]] != start:
                chain.append(parent[chain[-1]])
            chain.reverse()                      # start's neighbour ... end
            steer = next((s for s in chain if np.linalg.norm(pts[s]-pos) > self.cfg.steer_lookahead_mm), chain[-1])
            dead = len(adj[e]) == 1 and dist[e] < R
            paths.append((e, steer, min(dist[e], R)/R, dead))
        return paths

    def observe(self):
        env, cfg = self.env, self.cfg
        n = env.num_robots
        tree, t = env.tree, env.transport
        _, pts = self._graph()
        obs = np.zeros((n, OBS_DIM), np.float32)
        pos = env.positions_mm[:n].astype(np.float64)
        if cfg.position_noise_mm > 0:
            pos = pos+self.rng.normal(0., cfg.position_noise_mm, pos.shape)
        st = np.asarray(env.robot_stations)
        frame = np.stack((tree.tangents[st], tree.normals[st], tree.binormals[st]), axis=1).astype(np.float64)
        loc = lambda i, v: frame[i]@v
        axis, lumen, radial, _ = t.coordinates(env.positions_mm[:n], env.edges[:n], env.solution)
        active = env.active[:n]
        alive = env.masses > 0
        p_ids = np.flatnonzero(env.active[n:])+n
        p_pos = env.positions_mm[p_ids]
        p_vel = t.velocity_mm_s(p_pos, env.edges[p_ids], env.solution) if len(p_ids) else np.zeros((0, 3))
        for i in range(n):
            if not active[i]:
                continue
            o = obs[i]
            o[0:3] = self.prev_action[i]
            o[3:6] = self._noisy(loc(i, env.velocity_mm_s[i]))/env.config.robot_speed_mm_s
            o[6:9] = self._noisy(loc(i, (env.positions_mm[i]-axis[i])/max(lumen[i], 1e-9)))
            o[9] = self._noisy((lumen[i]-radial[i]-env.config.robot_radius_mm)/max(lumen[i], 1e-9))
            o[10] = max(0., 1-env.elapsed_s/env.config.episode_duration_s)
            # clots: straight-line vector only
            rel = env.clot_positions_mm-pos[i]
            d = np.linalg.norm(rel, axis=1)
            order = np.argsort(np.where(alive, d, np.inf))[:CLOT_SLOTS]
            for k, j in enumerate(order):
                if not alive[j]:
                    continue
                s = CLOT0+6*k
                o[s] = 1.; o[s+1] = self._noisy(d[j])/10.
                o[s+2:s+5] = loc(i, rel[j]/max(d[j], 1e-9)); o[s+5] = env.masses[j]/env.initial_mass[j]
            # local lumen geometry inside the view radius
            paths = self._local_paths(int(st[i]), pos[i])
            info = []
            for e, steer, far, dead in paths:
                sd = loc(i, pts[steer]-pos[i]); fd = loc(i, pts[e]-pos[i])
                info.append((sd/max(np.linalg.norm(sd), 1e-9), fd/max(np.linalg.norm(fd), 1e-9), far, dead,
                             np.log(max(float(env.solution['radius_mm'][e]), 1e-6))))
            info.sort(key=lambda x: -x[0][0])   # fixed geometric order: most forward (tangent) first
            for k, (sd, fd, far, dead, lr) in enumerate(info[:PATH_SLOTS]):
                s = PATH0+10*k
                o[s] = 1.; o[s+1:s+4] = sd; o[s+4:s+7] = fd; o[s+7] = far; o[s+8] = float(dead); o[s+9] = lr
            # particles inside the sensing radius
            if len(p_ids):
                rp = p_pos-pos[i]; dp = np.linalg.norm(rp, axis=1)
                near = np.flatnonzero(dp < cfg.sensing_radius_mm)
                near = near[np.argsort(dp[near])][:PARTICLE_SLOTS]
                for k, j in enumerate(near):
                    s = PART0+7*k
                    o[s] = 1.; o[s+1:s+4] = self._noisy(loc(i, rp[j]))/cfg.sensing_radius_mm
                    o[s+4:s+7] = self._noisy(loc(i, p_vel[j]-env.velocity_mm_s[i]))/env.config.robot_speed_mm_s
            # teammates inside the sensing radius
            rm = pos-pos[i]; dm = np.linalg.norm(rm, axis=1); dm[i] = np.inf; dm[~active] = np.inf
            mates = np.flatnonzero(dm < cfg.sensing_radius_mm)
            mates = mates[np.argsort(dm[mates])][:MATE_SLOTS]
            for k, j in enumerate(mates):
                s = MATE0+4*k
                o[s] = 1.; o[s+1:s+4] = self._noisy(loc(i, rm[j]))/cfg.sensing_radius_mm
        return obs
