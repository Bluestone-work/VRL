"""Weighted vessel graph G = (V, E, W) for the global layer.

Built only from preoperative information: centreline stations, healthy lumen
radius, tree topology and a nominal flow solution.  The realised flow regime,
patient-level actuation response and obstacle motion are unknown to it.

Two levels are kept:
* the station graph (fine): every simulator station is a vertex; it carries the
  exact geodesics, travel times and risk used by the planner;
* the key graph (coarse): root, bifurcations, leaves, targets and release
  candidates, with aggregated segment attributes; used for reporting and figures.

Travel time of a station edge a->b: L / max(v_min, eta*v_robot + s*u), with s = +1
downstream (towards larger depth from the root), -1 upstream, u the nominal
cross-sectional mean flow speed.  Risk is the time spent in edges whose lumen
clearance is below three body radii or that are traversed against flow faster
than 0.3 mm/s.  On a tree, the route between two stations is unique, so routing
reduces to choosing endpoints and visiting order.
"""
from __future__ import annotations

import numpy as np

from marl.v3 import assumptions as A


class VesselGraph:
    def __init__(self, env, body_mm=None):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path, breadth_first_order
        tree = env.tree
        self.env = env
        self.n = tree.n_stations
        self.scale = float(tree.physical_mm_per_unit)
        self.points = env.transport.points.astype(float)
        self.radius = np.asarray(env.flow_model.healthy_radius_mm, float)
        self.body = float(body_mm if body_mm is not None else env.config.robot_radius_mm)
        rows, cols, w = zip(*[(i, j, d * self.scale) for i, nb in enumerate(tree.station_graph) for j, d in nb])
        self.adj = csr_matrix((w, (rows, cols)), shape=(self.n, self.n))
        self.dist, self._pred = shortest_path(self.adj, directed=False, return_predecessors=True)
        self.root = int(env.flow_model.root)
        order, parent = breadth_first_order(self.adj, self.root, directed=False, return_predecessors=True)
        self.parent = parent
        self.depth = self.dist[self.root]
        self.degree = np.asarray([len(nb) for nb in tree.station_graph])
        self.flow = self._nominal_flow_speed(env)
        mm = A.load()['motion_model_for_planning']
        self.v_robot = float(mm['robot_speed_mm_s']) * float(mm['speed_efficiency'])
        self.v_min = float(mm['min_effective_speed_mm_s'])
        self._cache = {}

    @staticmethod
    def _nominal_flow_speed(env):
        """Mean speed per station of the healthy tree at the nominal 0.05 mm/s inlet (LysisEpisode scaling, multiplier 2)."""
        fm = env.flow_model
        healthy = fm.solve(fm.healthy_radius_mm)
        v0 = float(healthy['station_inflow_mm3_s'][fm.root]) / (np.pi * float(fm.healthy_radius_mm[fm.root]) ** 2)
        return np.asarray(healthy['mean_speed_mm_s'], float) * 0.05 / max(v0, 1e-12)

    # ---------------------------------------------------------------- routes
    def path(self, a, b):
        """Station list from a to b (inclusive)."""
        a, b = int(a), int(b)
        out = [b]
        while out[-1] != a:
            p = int(self._pred[a, out[-1]])
            if p < 0:
                raise ValueError(f'stations {a} and {b} are not connected')
            out.append(p)
        return out[::-1]

    def edge_profile(self, a, b):
        """(length_mm, time_s, risk_s, min_radius_mm) of the station edge a->b."""
        key = (a, b)
        if key in self._cache:
            return self._cache[key]
        L = float(np.linalg.norm(self.points[b] - self.points[a]))
        down = self.depth[b] > self.depth[a]
        u = 0.5 * (self.flow[a] + self.flow[b])
        v = max(self.v_min, self.v_robot + (u if down else -u))
        t = L / v
        r = float(min(self.radius[a], self.radius[b]))
        risky = (r - self.body < 3 * self.body) or ((not down) and u > 0.3)
        out = (L, t, t if risky else 0., r)
        self._cache[key] = out
        return out

    def route_profile(self, stations):
        """Cumulative arclength (mm) and time (s) at every station of a route, plus total risk (s) and min radius."""
        s = [0.]; t = [0.]; risk = 0.; rmin = np.inf
        for a, b in zip(stations, stations[1:]):
            L, dt, rk, r = self.edge_profile(a, b)
            s.append(s[-1] + L); t.append(t[-1] + dt); risk += rk; rmin = min(rmin, r)
        return np.asarray(s), np.asarray(t), risk, rmin

    def passable(self, stations):
        return bool(np.all(self.radius[stations] > self.body + 1e-7))

    def ancestors(self, s):
        """Stations on the root path of s, root first."""
        return self.path(self.root, s)

    # ------------------------------------------------------------- deployment
    def release_candidates(self, root_station):
        dep = A.load()['deployment']
        rmin = float(dep['catheter_min_lumen_radius_mm'])
        bif = np.flatnonzero(self.degree >= 3)
        out = [int(root_station)]
        for s in bif:
            route = self.path(self.root, int(s))
            if np.all(self.radius[route] >= rmin):
                out.append(int(s))
        return sorted(set(out), key=lambda s: self.depth[s])

    def catheter_time(self, a, b):
        return float(self.dist[a, b]) / float(A.get('deployment', 'catheter_speed_mm_s'))

    # -------------------------------------------------------------- key graph
    def key_graph(self, targets, release):
        """Coarse graph for reporting: nodes = root, bifurcations, leaves, targets, release candidates."""
        import networkx as nx
        key = set(np.flatnonzero(self.degree != 2).tolist()) | {self.root} | set(map(int, targets)) | set(map(int, release))
        G = nx.Graph()
        for k in key:
            kind = ('target' if k in set(map(int, targets)) else 'root' if k == self.root else
                    'bifurcation' if self.degree[k] >= 3 else 'leaf' if self.degree[k] == 1 else 'release')
            G.add_node(int(k), kind=kind, release=k in set(map(int, release)), pos=self.points[k].tolist(),
                       radius_mm=float(self.radius[k]))
        for k in key:
            if k == self.root:
                continue
            # walk towards the root until the next key station
            cur, seg = int(k), [int(k)]
            while True:
                p = int(self.parent[cur])
                if p < 0:
                    break
                seg.append(p); cur = p
                if p in key:
                    break
            if cur in key and cur != k:
                route = seg[::-1]
                s, t_down, risk, rmin = self.route_profile(route)
                _, t_up, risk_up, _ = self.route_profile(route[::-1])
                G.add_edge(int(cur), int(k), length_mm=float(s[-1]), time_down_s=float(t_down[-1]),
                           time_up_s=float(t_up[-1]), risk_down_s=float(risk), risk_up_s=float(risk_up),
                           min_radius_mm=float(rmin), passable=self.passable(route),
                           mean_flow_mm_s=float(np.mean(self.flow[route])), stations=len(route))
        return G
