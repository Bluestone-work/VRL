"""Path-pursuit navigation controllers for the unified benchmark (2026-10-05).

Motivation (measured): the step-wise route bearing of the old teacher oscillates at shallow
bifurcations. The nearest centreline station flips between the parent end and the first station
of the wrong daughter branch, the next-hop goal flips with it, and a single cluster can circle at
one junction for tens of seconds (MCA scene 2600000000: geodesic distance to the target frozen
at 10.5 mm for 100 s while moving at 0.9 mm/s). Five robots spread over the branches rarely cross
a junction backwards, so this never showed up before; one cluster visiting four clots must.

Both controllers follow a polyline with a monotone progress index (pure pursuit with a lookahead
carrot), so a flip of the nearest-station projection cannot reverse the commitment.

  RoutePursuit  [privileged] polyline = shortest path on the known vessel map to the planned clot
  LocalPursuit  [fair]       polyline = path inside the local imaging view (same view radius as
                              marl.partial_obs) to the visible end that best points at the clot's
                              straight-line direction; dead ends the cluster has already explored
                              are remembered (own odometry) and avoided; choice is held until the
                              chosen end is reached or leaves the view

Both add the same safety layer: particle repulsion + wait (forecast clearance < 0.3 margins within
0.5 s), wall keeping (remove outward component and push back near the wall) and the 0.35 stop
deadzone. RoutePursuit reads particle/wall features from the routed observation; LocalPursuit only
from the fair observation vector.
"""
from __future__ import annotations

import heapq

import numpy as np

from marl.fair_reactive import _closest, _particles
from marl.partial_obs import PartialObsConfig

LOOKAHEAD_MM = .4
JUNCTION_MM = .15
JUNCTION_ZONE_MM = 1.5
NEAR_LOOKAHEAD_MM = .08
DIRECT_MM = .7
AVOID_GAIN, WAIT_CLEAR, WAIT_HORIZON, DEADZONE = 6., .3, .5, .35


def _frames(env):
    s = env.robot_stations
    return np.stack((env.tree.tangents[s], env.tree.normals[s], env.tree.binormals[s]), axis=1).astype(np.float64)


def _finish(local, push, wait, wall_out, wall_near, wall_gain):
    local = np.asarray(local, np.float64)+AVOID_GAIN*push
    if wall_near > 0:
        outward = max(float(local@wall_out), 0.)
        local = local-wall_near*outward*wall_out-wall_gain*wall_near*wall_out
    local = np.clip(local, -1, 1); local /= max(np.linalg.norm(local), 1.)
    if wait or np.linalg.norm(local) < DEADZONE:
        return np.zeros(3)
    return local


def _dedupe(points, branches, stations):
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]
    keep_next = np.r_[keep[1:], True]
    # of a coincident pair keep the later point, which carries the daughter branch id
    sel = np.flatnonzero(keep_next | (np.arange(len(points)) == len(points)-1))
    return points[sel], np.asarray(branches)[sel], np.asarray(stations)[sel]


class _Pursuit:
    TAKEOFF_ZONE_MM = .6
    OVERSHOOT_MM = .3

    def _takeoff(self, i, pos):
        """Direction for entering a daughter branch, or None.

        The transport switches edges only when a body crosses an edge END axially and then picks
        the incident edge best aligned with its motion. A daughter branch whose first segment points
        backwards relative to the current edge therefore cannot be entered by turning on the spot
        (acute take-off). Overshoot past the take-off station onto the parent's other edge, then
        command the daughter direction so that the crossing happens through the take-off node.
        """
        env, pts, br, st = self.env, self.poly[i], self.poly_branch[i], self.poly_st[i]
        k = self.progress[i]
        e = int(env.edges[i]); ends = env.transport.ends[e]
        # The pending change is the first one (looking a few points back, since the carrot index
        # may already have moved on) whose daughter the body has not physically entered yet.
        change = [j for j in range(max(k-6, 0), len(pts)-1) if br[j+1] != br[j]
                  and not (env.tree.branch_ids[ends] == br[j+1]).all() and int(st[j+1]) not in ends]
        if not change:
            return None
        j = change[0]
        groups = env.transport.groups
        # Junction node: a daughter that starts at a merged (zero-length) junction belongs to that
        # junction's group; a mid-segment side branch hangs off its parent's station st[j].
        gT = groups[int(st[j+1])] if groups[int(st[j+1])] != int(st[j+1]) else groups[int(st[j])]
        Tp = env.transport.points[gT]
        if np.linalg.norm(Tp-pos) > self.TAKEOFF_ZONE_MM:
            return None
        far = next((q for q in pts[j+1:] if np.linalg.norm(q-Tp) > .3), pts[-1])
        d_b = far-Tp; d_b /= max(np.linalg.norm(d_b), 1e-9)
        if groups[ends[0]] != gT and groups[ends[1]] != gT:
            return Tp-pos                                      # not yet on an edge incident to the junction
        k_end = 0 if groups[ends[0]] == gT else 1
        axial = float(d_b@env.transport.direction[e])
        if (k_end == 1 and axial > 0) or (k_end == 0 and axial < 0):
            if np.linalg.norm(Tp-pos) > JUNCTION_MM:
                return Tp-pos                                  # reach the junction, then turn
            return d_b                                         # crossing the junction end selects the daughter
        # wrong side: overshoot along the parent beyond the junction, away from this edge's other end
        other = ends[1-k_end]
        beyond = Tp-env.transport.points[other]
        beyond /= max(np.linalg.norm(beyond), 1e-9)
        return Tp+self.OVERSHOOT_MM*beyond-pos

    def _remaining(self, i):
        pts = self.poly[i][self.progress[i]:]
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.

    def _carrot(self, i, pos):
        pts, k = self.poly[i], self.progress[i]
        window = pts[k:k+40]
        k = k+int(np.argmin(np.linalg.norm(window-pos, axis=1)))
        self.progress[i] = k
        acc, j = 0., k
        br = self.poly_branch[i]
        # Near a take-off the daughter centreline is the only lumen-safe line: track it tightly.
        seg = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        change = np.flatnonzero(np.diff(br) != 0)
        near_junction = len(change) and np.min(np.abs(seg[change]-seg[k])) < JUNCTION_ZONE_MM
        look = NEAR_LOOKAHEAD_MM if near_junction else LOOKAHEAD_MM
        while j+1 < len(pts) and acc < look:
            # Side branches leave the parent's centreline mid-segment: never cut the corner.
            # Reach the take-off station before aiming into the new branch.
            if br[j+1] != br[j]:
                if np.linalg.norm(pts[j]-pos) > JUNCTION_MM:
                    break
                self.progress[i] = max(self.progress[i], j+1)   # junction reached: commit to the new branch
            acc += float(np.linalg.norm(pts[j+1]-pts[j])); j += 1
        dev = float(np.min(np.linalg.norm(window-pos, axis=1)))
        self.near_junction[i] = bool(near_junction)
        return pts[j], dev, j == len(pts)-1


class RoutePursuit(_Pursuit):
    """Privileged: known-map shortest path to the planned clot."""
    def __init__(self, env, wall_gain=3.):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        tree = env.tree; n = tree.n_stations; s = float(tree.physical_mm_per_unit)
        r, c, w = zip(*[(i, j, d*s) for i, nb in enumerate(tree.station_graph) for j, d in nb])
        _, self.pred = shortest_path(csr_matrix((w, (r, c)), shape=(n, n)), directed=False, return_predecessors=True)
        self.env, self.wall_gain = env, wall_gain
        k = env.num_robots
        self.goal = np.full(k, -1); self.poly = [None]*k; self.poly_branch = [None]*k; self.progress = np.zeros(k, int)
        self.near_junction = np.zeros(k, bool); self.poly_st = [None]*k

    def _plan(self, i, target):
        env = self.env
        a, b = int(env.robot_stations[i]), int(env.clot_stations[target])
        path = [b]
        while path[-1] != a and self.pred[a, path[-1]] >= 0:
            path.append(int(self.pred[a, path[-1]]))
        path = path[::-1]
        self.poly[i], self.poly_branch[i], self.poly_st[i] = _dedupe(env.transport.points[path].astype(np.float64), env.tree.branch_ids[path], path)
        self.progress[i] = 0; self.goal[i] = target

    def act(self, targets):
        env = self.env; n = env.num_robots
        nodes = env._observation()['nodes'].astype(np.float64)
        frames = _frames(env); out = np.zeros((n, 3))
        for i in range(n):
            t = int(targets[i])
            if t < 0 or not env.active[i]:
                continue
            pos = env.positions_mm[i].astype(np.float64)
            if self.goal[i] != t or self.poly[i] is None:
                self._plan(i, t)
            carrot, dev, _ = self._carrot(i, pos)
            if dev > 1.5:
                self._plan(i, t); carrot, _, _ = self._carrot(i, pos)
            if self._remaining(i) < DIRECT_MM:
                carrot = env.clot_positions_mm[t]
            d = carrot-pos
            man = self._takeoff(i, pos)
            if man is not None:
                d = man
            local = frames[i]@(d/max(np.linalg.norm(d), 1e-9))
            push = np.zeros(3); wait = False
            for k in range(4):
                s = 36+10*k
                if nodes[i, s+9] <= 0:
                    continue
                rel = nodes[i, s:s+3]*1.5; relv = nodes[i, s+3:s+6]*env.config.robot_speed_mm_s
                tc, closest, clear = _closest(rel, relv)
                push += np.clip(1-clear, 0, 1)**2*(-closest/max(np.linalg.norm(closest), 1e-9))
                wait |= clear < WAIT_CLEAR and tc < WAIT_HORIZON
            near = float(np.clip(1-nodes[i, 12], 0, 1))      # clearance in robot radii, margin 1
            out[i] = _finish(local, push, wait, nodes[i, 9:12], near, self.wall_gain)
        return out


class LocalPursuit(_Pursuit):
    """Fair: only the lumen visible within the view radius, the clot's straight-line vector and own memory."""
    def __init__(self, env, cfg=PartialObsConfig(), wall_gain=3., commit_s=2.):
        self.env, self.cfg, self.wall_gain = env, cfg, wall_gain
        tree = env.tree; s = float(tree.physical_mm_per_unit)
        self.adj = [[(j, d*s) for j, d in nb] for nb in tree.station_graph]
        self.pts = env.transport.points.astype(np.float64)
        k = env.num_robots
        self.poly = [None]*k; self.poly_branch = [None]*k; self.progress = np.zeros(k, int); self.end = np.full(k, -1)
        self.near_junction = np.zeros(k, bool); self.poly_st = [None]*k
        self.goal = np.full(k, -1); self.commit_until = np.zeros(k)
        self.dead = [set() for _ in range(k)]      # explored dead-end stations (own odometry)

    def _view(self, start):
        R = self.cfg.view_radius_mm
        dist, parent = {start: 0.}, {start: -1}; heap = [(0., start)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u] or d >= R:
                continue
            for v, w in self.adj[u]:
                if d+w < dist.get(v, np.inf):
                    dist[v], parent[v] = d+w, u; heapq.heappush(heap, (d+w, v))
        children = {p for p in parent.values() if p >= 0}
        ends = [v for v in parent if v != start and v not in children]
        return dist, parent, ends

    def act(self, targets, obs):
        env = self.env; n = env.num_robots
        frames = _frames(env); out = np.zeros((n, 3))
        for i in range(n):
            t = int(targets[i])
            if t < 0 or not env.active[i]:
                continue
            if self.goal[i] != t:
                self.goal[i] = t; self.poly[i] = None; self.commit_until[i] = 0.
            pos = env.positions_mm[i].astype(np.float64)
            goal_vec = env.clot_positions_mm[t]-pos                 # straight-line vector (preoperative clot position)
            gdist = float(np.linalg.norm(goal_vec)); gdir = goal_vec/max(gdist, 1e-9)
            start = int(env.robot_stations[i])
            if len(self.adj[start]) == 1 and gdist > self.cfg.view_radius_mm:
                self.dead[i].add(start)                            # standing in an explored dead end
            dist, parent, ends = self._view(start)
            replan = self.poly[i] is None or env.elapsed_s >= self.commit_until[i] or self.end[i] not in dist
            if not replan:
                carrot, dev, done = self._carrot(i, pos)
                replan = done or dev > 1.
            if replan:
                best, score = None, -np.inf
                for e in ends:
                    v = self.pts[e]-pos; v /= max(np.linalg.norm(v), 1e-9)
                    leaf = len(self.adj[e]) == 1
                    sc = float(v@gdir) - (2. if leaf and dist[e] < self.cfg.view_radius_mm and gdist > dist[e]+1. else 0.) \
                         - (3. if e in self.dead[i] else 0.)
                    if sc > score:
                        best, score = e, sc
                if best is None:
                    continue
                chain = [best]
                while parent[chain[-1]] != -1:
                    chain.append(parent[chain[-1]])
                chain = chain[::-1]
                self.poly[i], self.poly_branch[i], self.poly_st[i] = _dedupe(self.pts[chain], env.tree.branch_ids[chain], chain)
                self.progress[i] = 0; self.end[i] = best
                self.commit_until[i] = env.elapsed_s+2.
                carrot, _, _ = self._carrot(i, pos)
            # fair: only when the clot lies on the visible path and little path remains to it
            if gdist < DIRECT_MM and np.min(np.linalg.norm(self.poly[i]-env.clot_positions_mm[t], axis=1)) < .3 \
                    and self._remaining(i) < DIRECT_MM+.3:
                carrot = env.clot_positions_mm[t]
            d = carrot-pos
            man = self._takeoff(i, pos)
            if man is not None:
                d = man
            local = frames[i]@(d/max(np.linalg.norm(d), 1e-9))
            o = obs[i]; push = np.zeros(3); wait = False
            for rel, relv in _particles(o, self.cfg):
                tc, closest, clear = _closest(rel, relv)
                push += np.clip(1-clear, 0, 1)**2*(-closest/max(np.linalg.norm(closest), 1e-9))
                wait |= clear < WAIT_CLEAR and tc < WAIT_HORIZON
            off = o[6:9].astype(np.float64); r = float(np.linalg.norm(off))
            near = 0. if r < 1e-6 else float(np.clip(1-o[9]/.25, 0, 1))
            out[i] = _finish(local, push, wait, off/max(r, 1e-9), near, self.wall_gain)
        return out
