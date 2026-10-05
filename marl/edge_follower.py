"""Edge-level path following that respects the transport's junction rule (2026-10-05).

Motivation (measured): the simulator moves a body along one centreline edge and switches edge only
when the body crosses an edge END; it then picks the incident edge whose outward direction best
aligns with the step. Station-level pursuit therefore fails in two ways: it oscillates across a
shallow bifurcation (nearest station flips between branches) and it can never enter an ACUTE
side branch, whose first segment points back along the parent (turning on the spot moves the body
away from the junction, so no crossing happens). MCA perforators are acute side branches; a
single cluster visiting four clots must enter them. Both failures froze one cluster for >100 s.

The follower plans in the transport's junction-group graph and commands, per step:
  far from the next junction        -> along the current edge towards it (+ centring)
  at a junction, obtuse turn        -> along the next edge: the crossing then selects it
  at a junction, acute turn         -> straight through (overshoot onto the continuation edge),
                                       then back along the next edge (the return crossing selects it)
Used by the privileged route follower (global group path) and by the fair local follower (group path
inside the local view only); the path SOURCE is the only difference.
"""
from __future__ import annotations

import heapq

import numpy as np

AT_JUNCTION_MM = .25
CENTRING = 1.5


class GroupGraph:
    def __init__(self, env):
        t = env.transport
        self.t = t
        self.g = t.groups
        self.eg = t.edge_groups                      # [edge, 2] group ids of the two ends
        self.adj = {}
        for e, (a, b) in enumerate(self.eg):
            self.adj.setdefault(int(a), []).append((int(b), float(t.length[e]), e))
            self.adj.setdefault(int(b), []).append((int(a), float(t.length[e]), e))
        self.point = {int(g): t.points[int(g)] for g in self.adj}

    def shortest(self, sources, goal):
        """sources: {group: initial cost}. Returns list of groups from a source to goal."""
        dist = dict(sources); prev = {}
        heap = [(c, g) for g, c in sources.items()]; heapq.heapify(heap)
        while heap:
            d, u = heapq.heappop(heap)
            if u == goal:
                break
            if d > dist.get(u, np.inf):
                continue
            for v, w, _ in self.adj.get(u, []):
                if d+w < dist.get(v, np.inf):
                    dist[v] = d+w; prev[v] = u; heapq.heappush(heap, (d+w, v))
        if goal not in dist:
            return None
        path = [goal]
        while path[-1] in prev:
            path.append(prev[path[-1]])
        return path[::-1]

    def edge_between(self, a, b):
        for v, _, e in self.adj.get(a, []):
            if v == b:
                return e
        return None

    def away(self, e, g):
        """Unit direction of edge e pointing away from its end in group g."""
        d = self.t.direction[e]
        return d if int(self.eg[e, 0]) == g else -d


def follow_command(graph, env, i, path, goal_point):
    """World-frame unit command for body i following group `path` (ends at the goal's group)."""
    t = graph.t
    pos = env.positions_mm[i].astype(np.float64)
    e = int(env.edges[i]); a, b = int(graph.eg[e, 0]), int(graph.eg[e, 1])
    axis, lumen, radial, _ = t.coordinates(pos[None], np.array([e]), env.solution)
    centre = (axis[0]-pos)/max(float(lumen[0]), 1e-9)
    if path is None or len(path) < 2:
        d = goal_point-pos
        return d/max(np.linalg.norm(d), 1e-9)
    # locate the current edge on the path
    k = None
    for j in range(len(path)-1):
        if {path[j], path[j+1]} == {a, b}:
            k = j; break
    if k is None:
        # off-path edge incident to a path junction (after an overshoot): head back through it
        for j, g in enumerate(path[:-1]):
            if g in (a, b):
                nxt = graph.edge_between(path[j], path[j+1])
                d_next = graph.away(nxt, g)
                toward_g = -graph.away(e, g)
                if float(np.linalg.norm(graph.point[g]-pos)) > AT_JUNCTION_MM:
                    c = toward_g+CENTRING*centre                    # come back along the axis first
                    return c/max(np.linalg.norm(c), 1e-9)
                if float(d_next@toward_g) > 0:
                    return d_next                                   # return crossing selects the next edge
                return toward_g
        d = graph.point[path[0]]-pos
        return d/max(np.linalg.norm(d), 1e-9)
    G = path[k+1]                                                  # junction we are heading to
    toward = -graph.away(e, G)
    dist_G = float(np.linalg.norm(graph.point[G]-pos))
    if k+1 == len(path)-1:                                         # last edge: go for the goal point
        d = goal_point-pos
        return d/max(np.linalg.norm(d), 1e-9)
    nxt = graph.edge_between(G, path[k+2])
    d_next = graph.away(nxt, G)
    if dist_G > AT_JUNCTION_MM:
        c = toward+CENTRING*centre
        return c/max(np.linalg.norm(c), 1e-9)
    if float(d_next@toward) > .05:
        return d_next                                              # obtuse turn: cross along the next edge
    return toward                                                  # acute turn: overshoot straight through


AVOID_GAIN, WAIT_CLEAR, WAIT_HORIZON, DEADZONE = 6., .3, .5, .35


def _safety(local, particles):
    """Particle repulsion from the closest-approach point + wait rule + stop deadzone (Frenet frame)."""
    from marl.fair_reactive import _closest
    push = np.zeros(3); wait = False
    for rel, relv in particles:
        tc, closest, clear = _closest(rel, relv)
        push += np.clip(1-clear, 0, 1)**2*(-closest/max(np.linalg.norm(closest), 1e-9))
        wait |= clear < WAIT_CLEAR and tc < WAIT_HORIZON
    local = np.clip(local+AVOID_GAIN*push, -1, 1); local /= max(np.linalg.norm(local), 1.)
    return np.zeros(3) if wait or np.linalg.norm(local) < DEADZONE else local


def _frames(env):
    s = env.robot_stations
    return np.stack((env.tree.tangents[s], env.tree.normals[s], env.tree.binormals[s]), axis=1).astype(np.float64)


def _sources(graph, env, i):
    pos = env.positions_mm[i]; e = int(env.edges[i]); a, b = int(graph.eg[e, 0]), int(graph.eg[e, 1])
    return {a: float(np.linalg.norm(graph.point[a]-pos)), b: float(np.linalg.norm(graph.point[b]-pos))}


class RouteFollower:
    """[privileged] global junction-group shortest path on the known vessel map to the planned clot."""
    def __init__(self, env):
        self.env, self.graph = env, GroupGraph(env)

    def act(self, targets):
        env = self.env; n = env.num_robots
        nodes = env._observation()['nodes'].astype(np.float64)
        frames = _frames(env); out = np.zeros((n, 3))
        for i in range(n):
            t = int(targets[i])
            if t < 0 or not env.active[i]:
                continue
            goal_g = int(self.graph.g[int(env.clot_stations[t])])
            path = self.graph.shortest(_sources(self.graph, env, i), goal_g)
            w = follow_command(self.graph, env, i, path, env.clot_positions_mm[t].astype(np.float64))
            parts = [(nodes[i, 36+10*k:39+10*k]*1.5, nodes[i, 39+10*k:42+10*k]*env.config.robot_speed_mm_s)
                     for k in range(4) if nodes[i, 45+10*k] > 0]
            out[i] = _safety(frames[i]@w, parts)
        return out


class LocalFollower:
    """[fair] junction-group path inside the local view only, towards the end that best points at the
    clot's straight-line direction; explored dead ends remembered (own odometry); choice held 2 s."""
    FEATURES = ('cos_goal', 'view_dist', 'euclid_progress', 'leaf', 'dead', 'tabu', 'radius_ratio', 'log_goal_mm',
                'n_frontier', 'turn_cos', 'is_current')

    def __init__(self, env, view_mm=4., commit_s=2., memory=False, tabu_s=0., scorer=None, recorder=None):
        self.scorer, self.recorder = scorer, recorder
        self.env, self.graph, self.view, self.commit, self.memory = env, GroupGraph(env), view_mm, commit_s, memory
        self.tabu_s = tabu_s
        self.tabu = [dict() for _ in range(env.num_robots)]   # abandoned frontier group -> time it was abandoned
        n = env.num_robots
        self.end = [None]*n; self.until = np.zeros(n); self.goal = np.full(n, -1)
        self.dead = [set() for _ in range(n)]
        self.visits = [dict() for _ in range(n)]   # junction groups passed (own odometry), per target

    def _local(self, sources):
        g = self.graph; dist = dict(sources); prev = {}
        heap = [(c, u) for u, c in sources.items()]; heapq.heapify(heap)
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d >= self.view:
                continue
            for v, w, _ in g.adj.get(u, []):
                if d+w < dist.get(v, np.inf):
                    dist[v] = d+w; prev[v] = u; heapq.heappush(heap, (d+w, v))
        return dist, prev

    def act(self, targets, obs):
        from marl.fair_reactive import _particles
        from marl.partial_obs import PartialObsConfig
        env, g = self.env, self.graph; n = env.num_robots
        frames = _frames(env); out = np.zeros((n, 3))
        for i in range(n):
            t = int(targets[i])
            if t < 0 or not env.active[i]:
                continue
            if self.goal[i] != t:
                self.goal[i] = t; self.end[i] = None; self.until[i] = 0.; self.visits[i] = {}; self.tabu[i] = {}
            pos = env.positions_mm[i].astype(np.float64)
            clot = env.clot_positions_mm[t].astype(np.float64)
            gdir = (clot-pos)/max(np.linalg.norm(clot-pos), 1e-9)
            src = _sources(g, env, i)
            dist, prev = self._local(src)
            goal_g = int(g.g[int(env.clot_stations[t])])
            for u in src:                                   # standing in an explored dead end
                if len(g.adj.get(u, [])) == 1 and src[u] < .5 and u != goal_g:
                    self.dead[i].add(u)
                if self.memory and src[u] < .3 and len(g.adj.get(u, [])) >= 3:
                    self.visits[i][u] = self.visits[i].get(u, 0)+(1 if self.visits[i].get('_last') != u else 0)
                    self.visits[i]['_last'] = u
            if goal_g in dist:                               # clot visible on the local lumen
                end, point = goal_g, clot
            else:
                if self.end[i] is None or self.end[i] not in dist or env.elapsed_s >= self.until[i] or dist[self.end[i]] < .3:
                    frontier = [u for u in dist if (dist[u] >= self.view-1e-9 or len(g.adj.get(u, [])) == 1) and u not in src]
                    frontier = frontier or [u for u in dist if u not in src]
                    def feats(u):
                        v = g.point[u]-pos; nv = float(np.linalg.norm(v)); v = v/max(nv, 1e-9)
                        leaf = len(g.adj.get(u, [])) == 1 and dist[u] < self.view
                        tabu = any(np.linalg.norm(g.point[u]-g.point[q]) < 1.5 and env.elapsed_s-tq < max(self.tabu_s, 30.)
                                   for q, tq in self.tabu[i].items())
                        r_here = float(env.solution['radius_mm'][int(env.robot_stations[i])])
                        heading = env.velocity_mm_s[i]; hn = float(np.linalg.norm(heading))
                        return np.array([float(v@gdir), dist[u]/self.view,
                                         (np.linalg.norm(clot-pos)-np.linalg.norm(clot-g.point[u]))/max(dist[u], 1e-3),
                                         float(leaf), float(u in self.dead[i]), float(tabu),
                                         float(env.solution['radius_mm'][u])/max(r_here, 1e-6), np.log(max(np.linalg.norm(clot-pos), 1e-3)),
                                         len(frontier)/4., float(v@heading)/hn if hn > 1e-6 else 0., float(u == self.end[i])])

                    def score(u):
                        v = g.point[u]-pos; v /= max(np.linalg.norm(v), 1e-9)
                        leaf = len(g.adj.get(u, [])) == 1 and dist[u] < self.view
                        sc = float(v@gdir) - 2.*leaf - 3.*(u in self.dead[i])
                        if self.tabu_s > 0:
                            # a frontier abandoned recently lies behind us: re-choosing it closes a limit cycle
                            near_tabu = any(np.linalg.norm(g.point[u]-g.point[q]) < 1.5 and env.elapsed_s-tq < self.tabu_s
                                            for q, tq in self.tabu[i].items())
                            sc -= 3.*near_tabu
                        if self.memory:
                            # penalise frontiers whose local path re-enters junctions already passed for this target
                            q, rep = u, 0
                            while q in prev:
                                rep += self.visits[i].get(q, 0); q = prev[q]
                            sc -= .75*rep
                        return sc
                    if frontier and (self.scorer is not None or self.recorder is not None):
                        F = np.stack([feats(u) for u in frontier])
                        if self.recorder is not None:
                            self.recorder(env, i, t, frontier, F)
                    if frontier and self.scorer is not None:
                        new_end = frontier[int(np.argmax(self.scorer(F)))]
                    else:
                        new_end = max(frontier, key=score) if frontier else None
                    if self.tabu_s > 0 and self.end[i] is not None and new_end != self.end[i] and dist.get(self.end[i], 0.) >= .3:
                        self.tabu[i][self.end[i]] = env.elapsed_s       # abandoned before reaching it
                    self.end[i] = new_end
                    self.until[i] = env.elapsed_s+self.commit
                end = self.end[i]
                point = g.point[end] if end is not None else clot
            path = None
            if end is not None:
                path = [end]
                while path[-1] in prev:
                    path.append(prev[path[-1]])
                path = path[::-1]
            w = follow_command(g, env, i, path, point)
            out[i] = _safety(frames[i]@w, _particles(obs[i], PartialObsConfig()))
        return out
