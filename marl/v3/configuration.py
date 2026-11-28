"""Joint cluster configuration and task planning on the vessel graph.

Decision variables (all chosen by the program, none supplied by the user):
  N            number of clusters, 1 <= N <= n_max
  P            release station of every cluster (single catheter entry at the root)
  A            ordered targets of every cluster (allocation + visiting order)
  R            routes (unique geodesics on the tree once P and A are fixed)
  S            catheter release order and times, and passage order at shared segments

Planning model
--------------
* catheter: starts at the root release station at t = 0, moves at v_c between
  release stations along the tree; consecutive releases are at least
  release_interval_s apart:  t_next = t_prev + max(release_interval, d(prev, next)/v_c);
* a released cluster follows its route with the edge travel times of VesselGraph;
* passage order: prioritised simulation (dt = timeline_dt_s).  A cluster waits in
  place when its next move would bring it within d_safe of another released
  cluster; a cluster is released only when every released cluster is at least
  release_clearance_mm from the tip.  No progress of any unfinished cluster for
  60 s is a deadlock.  Several priority orders are simulated and the best kept.

Objective (lexicographic, registered in configs/v3/assumptions.json)
  (1) feasible: every target covered, routes passable, N <= n_max, no deadlock
  (2) J = T_pred + w_wait * W_pred + w_risk * R_pred
      T_pred deadline-scored predicted last target arrival (from catheter insertion)
      W_pred predicted waiting cluster-seconds (cost of resolving interactions)
      R_pred predicted seconds in narrow / adverse-flow segments
  (3) N* = smallest N whose best J is within epsilon of the best J over all N
  (4) total predicted cluster-time

Search
  'exact'    best-first enumeration of every (N, ordered allocation, release
             station per cluster, catheter order) with the admissible bound
             LB = T_free + w_risk * R (waiting can only delay).  Exact within the
             planning model for the candidate release set; used for m <= 5.
  'local'    the scalable optimiser: seeded construction + relocate / swap /
             release-station / release-order moves.  Its optimality gap is
             measured against 'exact' on the small-scale instances.
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from marl.v3 import assumptions as A


@dataclass
class Configuration:
    method: str
    n: int
    tours: list
    release_stations: list
    release_order: list
    release_times: list
    routes: list
    predicted: dict = field(default_factory=dict)
    certificate: str = ''
    planning_time_s: float = 0.
    search: dict = field(default_factory=dict)

    def hash(self, base_task_hash):
        from scripts.multicluster_protocol import digest
        return digest(dict(base=base_task_hash, n=self.n, tours=self.tours, release_stations=self.release_stations,
                           release_order=self.release_order, release_times=[round(float(t), 6) for t in self.release_times]))

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), default=jsonable))


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.generic,)):
        return x.item()
    raise TypeError(type(x))


def set_partitions(items, k):
    """All partitions of items into exactly k non-empty unlabeled blocks."""
    items = list(items)
    if k == 0:
        if not items:
            yield []
        return
    if len(items) < k:
        return
    first, rest = items[0], items[1:]
    for p in set_partitions(rest, k - 1):
        yield [[first]] + p
    for p in set_partitions(rest, k):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]


def ordered_tour_sets(m, k):
    for part in set_partitions(range(m), k):
        for perms in itertools.product(*[itertools.permutations(b) for b in part]):
            yield [tuple(p) for p in perms]


class Planner:
    def __init__(self, graph, target_stations, root_station, *, objective='full', topology='geodesic',
                 n_max=None, deadline=None, seed=0):
        cfg = A.load()
        self.g = graph
        self.targets = [int(t) for t in target_stations]
        self.m = len(self.targets)
        self.root_station = int(root_station)
        self.objective, self.topology = objective, topology
        self.n_max = int(n_max or cfg['resources']['n_max'])
        self.deadline = float(deadline or cfg['mission']['deadline_s'])
        self.d_safe = float(cfg['safety']['d_safe_mm'])
        dep = cfg['deployment']
        self.v_c = float(dep['catheter_speed_mm_s'])
        self.tau_rel = float(dep['release_interval_s'])
        self.release_clear = float(dep['release_clearance_mm'])
        ob = cfg['objective']
        self.w_wait, self.w_risk, self.eps = float(ob['w_wait']), float(ob['w_risk']), float(ob['epsilon_relative'])
        self.dt = float(cfg['motion_model_for_planning']['timeline_dt_s'])
        self.release_all = graph.release_candidates(root_station)
        self.rng = np.random.default_rng(seed)
        self._tour = {}
        self.sim_calls = 0

    # ------------------------------------------------------------ primitives
    def candidates_for(self, first_target, mode='ancestor'):
        if mode == 'root':
            return [self.root_station]
        if mode == 'all':
            return list(self.release_all)
        anc = set(self.g.ancestors(self.targets[first_target]))
        return [s for s in self.release_all if s in anc] or [self.root_station]

    def tour(self, release, tour):
        """Route, per-station cumulative time, arrival offsets, total time, risk, passability."""
        key = (int(release), tuple(tour))
        if key in self._tour:
            return self._tour[key]
        stations = [int(release)]
        for t in tour:
            seg = self.g.path(stations[-1], self.targets[t])
            stations += seg[1:]
        s, ts, risk, rmin = self.g.route_profile(stations)
        idx, pos, arr = 0, {}, []
        for t in tour:                                  # arrival = first index of the target station after the last one
            st = self.targets[t]
            k = stations.index(st, idx)
            arr.append(float(ts[k])); idx = k
        if self.topology == 'euclidean':                # ablation: allocation ignores topology and flow
            pts = [self.g.points[release]] + [self.g.points[self.targets[t]] for t in tour]
            d = np.cumsum([0.] + [float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:])])
            plan_time, plan_arr = float(d[-1]) / self.g.v_robot, [float(x) / self.g.v_robot for x in d[1:]]
        else:
            plan_time, plan_arr = float(ts[-1]), arr
        out = dict(stations=stations, cum_t=ts, cum_s=s, arrivals=arr, time=float(ts[-1]), risk=float(risk),
                   passable=self.g.passable(stations), plan_time=plan_time, plan_arrivals=plan_arr)
        self._tour[key] = out
        return out

    def release_times(self, nodes_in_order):
        t, prev, out = 0., self.root_station, []
        for k, nd in enumerate(nodes_in_order):
            travel = self.g.catheter_time(prev, nd)
            t = travel if k == 0 else t + max(self.tau_rel, travel)
            out.append(t); prev = nd
        return out

    def free_completion(self, tours, nodes, order, planning=True):
        """Conflict-free completion (lower bound) given catheter order (cluster indices)."""
        rel = self.release_times([nodes[c] for c in order])
        r = np.empty(len(tours)); r[list(order)] = rel
        key = 'plan_time' if planning else 'time'
        T = max(r[c] + self.tour(nodes[c], tours[c])[key] for c in range(len(tours)))
        R = sum(self.tour(nodes[c], tours[c])['risk'] for c in range(len(tours)))
        return float(T), float(R), r

    # ------------------------------------------------------------- scheduler
    def _dense(self, release, tour):
        info = self.tour(release, tour)
        key = ('dense', int(release), tuple(tour))
        if key in self._tour:
            return self._tour[key]
        T = info['time']
        grid = np.arange(0., T + self.dt, self.dt)
        P = self.g.points[info['stations']]
        pos = np.stack([np.interp(grid, info['cum_t'], P[:, a]) for a in range(3)], 1)
        arr_idx = [int(np.ceil(a / self.dt - 1e-9)) for a in info['arrivals']]
        out = (pos, arr_idx)
        self._tour[key] = out
        return out

    def simulate(self, tours, nodes, order, priority, coordinated=True):
        """Prioritised execution of a plan on the planning model. Returns predicted outcome."""
        self.sim_calls += 1
        n = len(tours)
        dense = [self._dense(nodes[c], tours[c]) for c in range(n)]
        node_pts = [self.g.points[nodes[c]] for c in range(n)]
        k = np.zeros(n, int); released = np.zeros(n, bool); done = np.zeros(n, bool)
        arrivals = {}
        rel_actual = np.full(n, np.nan)
        wait = 0.; viol = 0.; dmin = np.inf; events = 0
        in_viol = np.zeros((n, n), bool)
        next_rel = 0
        gaps = [self.g.catheter_time(self.root_station, nodes[order[0]])]
        for a, b in zip(order, order[1:]):
            gaps.append(max(self.tau_rel, self.g.catheter_time(nodes[a], nodes[b])))
        ready = gaps[0]
        t = 0.; last_progress = 0.
        steps = int(np.ceil(self.deadline / self.dt))
        for step in range(steps + 1):
            t = step * self.dt
            # catheter releases in order
            while next_rel < n and t >= ready - 1e-9:
                c = order[next_rel]
                clear = all(np.linalg.norm(dense[o][0][k[o]] - node_pts[c]) >= self.release_clear
                            for o in range(n) if released[o]) if coordinated else True
                if not clear:
                    break
                released[c] = True; rel_actual[c] = t; last_progress = t
                next_rel += 1
                if next_rel < n:
                    ready = t + gaps[next_rel]
            cur = np.array([dense[c][0][k[c]] for c in range(n)])
            for c in priority:
                if not released[c] or done[c]:
                    continue
                pos, arr_idx = dense[c]
                if k[c] + 1 >= len(pos):
                    done[c] = True
                    continue
                p = pos[k[c] + 1]
                blocked = False
                if coordinated:
                    for o in range(n):
                        if o == c or not released[o]:
                            continue
                        dn = np.linalg.norm(p - cur[o]); dc = np.linalg.norm(cur[c] - cur[o])
                        if dn < self.d_safe and dn < dc - 1e-9:
                            blocked = True; break
                if blocked:
                    wait += self.dt
                    continue
                k[c] += 1; cur[c] = p; last_progress = t
                for j, ai in enumerate(arr_idx):
                    if k[c] >= ai and (c, j) not in arrivals:
                        arrivals[(c, j)] = t
                if k[c] + 1 >= len(pos):
                    done[c] = True
            # spacing among released clusters
            for a in range(n):
                for b in range(a + 1, n):
                    if released[a] and released[b]:
                        d = float(np.linalg.norm(cur[a] - cur[b])); dmin = min(dmin, d)
                        v = d < self.d_safe
                        if v:
                            viol += self.dt
                            if not in_viol[a, b]:
                                events += 1
                        in_viol[a, b] = v
            if len(arrivals) == self.m:
                break
            if t - last_progress > 60. and (next_rel < n or not done.all()):
                break
        complete = len(arrivals) == self.m
        deadlock = (not complete) and (t < self.deadline - self.dt)
        arr = [arrivals.get((c, j)) for c in range(n) for j in range(len(tours[c]))]
        T = max(arr) if complete else self.deadline
        cluster_time = float(sum((max([arrivals.get((c, j), self.deadline) for j in range(len(tours[c]))]) -
                                  (rel_actual[c] if np.isfinite(rel_actual[c]) else self.deadline)) for c in range(n)))
        return dict(T=float(T), complete=bool(complete), deadlock=bool(deadlock), wait_s=float(wait),
                    violation_pair_s=float(viol), conflict_events=int(events),
                    min_pair_distance_mm=float(dmin) if np.isfinite(dmin) else None,
                    release_times=[float(x) if np.isfinite(x) else None for x in rel_actual],
                    arrivals=[None if a is None else float(a) for a in arr], cluster_time_s=cluster_time,
                    priority=list(map(int, priority)))

    def priorities(self, tours, nodes, order):
        n = len(tours)
        if n <= 3:
            return [list(p) for p in itertools.permutations(range(n))]
        times = [self.tour(nodes[c], tours[c])['time'] for c in range(n)]
        out = [list(order), list(order)[::-1], list(np.argsort(times)[::-1]), list(np.argsort(times))]
        uniq = []
        for p in out:
            if p not in uniq:
                uniq.append([int(x) for x in p])
        return uniq

    def evaluate(self, tours, nodes, order, objective=None):
        objective = objective or self.objective
        tours = [tuple(t) for t in tours]
        passable = all(self.tour(nodes[c], tours[c])['passable'] for c in range(len(tours)))
        T_free, R, rel = self.free_completion(tours, nodes, order)
        if objective == 'time':
            return dict(J=T_free if passable else np.inf, LB=T_free, T_free=T_free, risk_s=R, feasible=passable,
                        scheduled=None)
        best = None
        for pr in self.priorities(tours, nodes, order):
            sim = self.simulate(tours, nodes, order, pr)
            J = sim['T'] + self.w_wait * sim['wait_s'] + self.w_risk * R
            feasible = passable and sim['complete'] and not sim['deadlock']
            if not feasible:
                J = 2 * self.deadline + J
            if best is None or J < best['J']:
                best = dict(J=float(J), LB=float(T_free + self.w_risk * R), T_free=T_free, risk_s=R,
                            feasible=feasible, scheduled=sim)
        return best

    def predicted(self, tours, nodes, order):
        """Full predicted record of a chosen plan (always with the geodesic model and coordination)."""
        topo, self.topology = self.topology, 'geodesic'
        self._tour = {k: v for k, v in self._tour.items() if False}
        try:
            ev = self.evaluate(tours, nodes, order, objective='full')
            free = self.simulate([tuple(t) for t in tours], nodes, order, list(order), coordinated=False)
            routes = [self.tour(nodes[c], tuple(tours[c]))['stations'] for c in range(len(tours))]
            shared = 0.
            for a in range(len(routes)):
                for b in range(a + 1, len(routes)):
                    common = set(zip(routes[a], routes[a][1:])) | set(zip(routes[a][1:], routes[a]))
                    for e in zip(routes[b], routes[b][1:]):
                        if e in common:
                            shared += float(np.linalg.norm(self.g.points[e[1]] - self.g.points[e[0]]))
        finally:
            self.topology = topo
            self._tour = {}
        sch = ev['scheduled']
        return dict(J=ev['J'], feasible=ev['feasible'], T_pred=sch['T'], T_free=ev['T_free'], wait_s=sch['wait_s'],
                    risk_s=ev['risk_s'], cluster_time_s=sch['cluster_time_s'], deadlock=sch['deadlock'],
                    release_times_pred=sch['release_times'], arrivals_pred=sch['arrivals'],
                    priority=sch['priority'], deployment_time_s=max([x for x in sch['release_times'] if x is not None] or [0.]),
                    potential_conflict_pair_s=free['violation_pair_s'], potential_conflict_events=free['conflict_events'],
                    potential_min_pair_distance_mm=free['min_pair_distance_mm'], shared_segment_mm=shared,
                    residual_conflict_pair_s=sch['violation_pair_s']), routes

    # ---------------------------------------------------------------- search
    def _node_options(self, tours, deployment, slots=None):
        if slots is not None:
            return [[s] for s in slots]
        return [self.candidates_for(t[0], 'root' if deployment == 'root' else 'ancestor') for t in tours]

    def exact(self, k, deployment='optimized', objective=None, cap=3000):
        """Best-first exact search for exactly k clusters (exact within the planning model)."""
        objective = objective or self.objective
        cands = []
        for tours in ordered_tour_sets(self.m, k):
            opts = self._node_options(tours, deployment)
            for nodes in itertools.product(*opts):
                for order in itertools.permutations(range(k)):
                    T, R, _ = self.free_completion(tours, list(nodes), list(order))
                    lb = T if objective == 'time' else T + self.w_risk * R
                    cands.append((lb, tours, list(nodes), list(order)))
        cands.sort(key=lambda x: x[0])
        best, evals, truncated = None, 0, False
        for lb, tours, nodes, order in cands:
            if best is not None and lb >= best[0] - 1e-9:
                break
            if evals >= cap:
                truncated = True
                break
            ev = self.evaluate(tours, nodes, order, objective)
            evals += 1
            if best is None or ev['J'] < best[0] - 1e-9:
                best = (ev['J'], tours, nodes, order)
        cert = 'exact_within_model' if not truncated else f'best_first_truncated_at_{cap}'
        return best, dict(candidates=len(cands), evaluated=evals, certificate=cert)

    def _best_order(self, group, node):
        if len(group) <= 6:
            perms = itertools.permutations(group)
        else:
            perms = [tuple(sorted(group, key=lambda t: self.g.dist[node, self.targets[t]]))]
        return min(perms, key=lambda p: self.tour(node, p)['plan_time'])

    def _best_node(self, group, deployment, slot=None):
        if slot is not None:
            return slot, self._best_order(group, slot)
        best = None
        for nd in self.candidates_for(group[0] if len(group) == 1 else
                                      min(group, key=lambda t: self.g.depth[self.targets[t]]),
                                      'root' if deployment == 'root' else 'ancestor'):
            tour = self._best_order(group, nd)
            # the node must be an ancestor of the first visited target, re-check after ordering
            if deployment != 'root' and nd not in set(self.g.ancestors(self.targets[tour[0]])):
                continue
            cost = self.g.catheter_time(self.root_station, nd) + self.tour(nd, tour)['plan_time']
            if best is None or cost < best[0]:
                best = (cost, nd, tour)
        if best is None:
            return self.root_station, self._best_order(group, self.root_station)
        return best[1], best[2]

    def _orders(self, nodes):
        k = len(nodes)
        if k <= 4:
            return [list(p) for p in itertools.permutations(range(k))]
        by = list(np.argsort([self.g.depth[s] for s in nodes]))
        return [by, by[::-1]]

    def _group_options(self, group, deployment, slot=None):
        """(release node, best visiting order from it) pairs for one group of targets."""
        if slot is not None:
            return [(slot, tuple(self._best_order(group, slot)))]
        mode = 'root' if deployment == 'root' else 'ancestor'
        nodes = []
        for t in group:
            for nd in self.candidates_for(t, mode):
                if nd not in nodes:
                    nodes.append(nd)
        out = []
        for nd in nodes:
            anc = None if mode == 'root' else nd
            perms = [p for p in itertools.permutations(group)] if len(group) <= 6 else [tuple(group)]
            if anc is not None:
                perms = [p for p in perms if nd in set(self.g.ancestors(self.targets[p[0]]))]
            if not perms:
                continue
            out.append((nd, min(perms, key=lambda p: self.tour(nd, p)['plan_time'])))
        return out or [(self.root_station, tuple(self._best_order(group, self.root_station)))]

    def _complete(self, groups, deployment, slots, objective, top=3):
        """Best release nodes and catheter order for a fixed partition: bound-ranked, top few simulated."""
        opts = [self._group_options(gr, deployment, None if slots is None else slots[i]) for i, gr in enumerate(groups)]
        ranked = []
        for combo in itertools.product(*opts):
            nodes = [c[0] for c in combo]; tours = [c[1] for c in combo]
            for order in self._orders(nodes):
                T, R, _ = self.free_completion(tours, nodes, order)
                ranked.append((T + (0. if objective == 'time' else self.w_risk * R), tours, nodes, order))
        ranked.sort(key=lambda x: x[0])
        best = None
        for lb, tours, nodes, order in ranked[:top]:
            if best is not None and lb >= best[0]:
                break
            ev = self.evaluate(tours, nodes, order, objective)
            if best is None or ev['J'] < best[0]:
                best = (ev['J'], tours, nodes, list(order))
        return best

    def local(self, k, deployment='optimized', objective=None, slots=None, restarts=3, budget=250):
        """Scalable optimiser for exactly k clusters: construction + relocate / swap moves."""
        objective = objective or self.objective
        D = self.g.dist[np.ix_(self.targets, self.targets)]
        evals, best = 0, None
        for r in range(restarts):
            # farthest-point seeds on geodesic distance, then nearest-seed assignment
            first = int(np.argmax(self.g.depth[self.targets])) if r == 0 else int(self.rng.integers(self.m))
            seeds = [first]
            while len(seeds) < k:
                seeds.append(int(np.argmax(np.min(D[:, seeds], axis=1))))
            groups = [[s] for s in seeds]
            for t in range(self.m):
                if t not in seeds:
                    groups[int(np.argmin(D[t, seeds]))].append(t)
            cur = self._complete(groups, deployment, slots, objective); evals += 1
            improved = True
            while improved and evals < budget:
                improved = False
                cur_groups = [list(t) for t in cur[1]]
                moves = []
                for a in range(k):
                    for t in cur_groups[a]:
                        for b in range(k):
                            if a != b and len(cur_groups[a]) > 1:
                                moves.append(('relocate', a, t, b))
                for a in range(k):
                    for b in range(a + 1, k):
                        for t in cur_groups[a]:
                            for u in cur_groups[b]:
                                moves.append(('swap', a, t, b, u))
                self.rng.shuffle(moves)
                for mv in moves:
                    g2 = [list(x) for x in cur_groups]
                    if mv[0] == 'relocate':
                        _, a, t, b = mv; g2[a].remove(t); g2[b].append(t)
                    else:
                        _, a, t, b, u = mv; g2[a].remove(t); g2[b].remove(u); g2[a].append(u); g2[b].append(t)
                    cand = self._complete(g2, deployment, slots, objective); evals += 1
                    if cand[0] < cur[0] - 1e-9:
                        cur, improved = cand, True
                        break
                    if evals >= budget:
                        break
            if best is None or cur[0] < best[0]:
                best = cur
        return best, dict(evaluated=evals, certificate='local_search', restarts=restarts, budget=budget)

    def solve(self, *, method='proposed', n_fixed=None, deployment='optimized', search='local', objective=None,
              random_slots_seed=None):
        """Return a Configuration.  n_fixed=None selects N by the epsilon rule."""
        t0 = time.monotonic()
        objective = objective or self.objective
        ks = [n_fixed] if n_fixed else list(range(1, min(self.n_max, self.m) + 1))
        per_k, logs = {}, {}
        for k in ks:
            slots = None
            if deployment == 'random':
                rr = np.random.default_rng([int(random_slots_seed or 0), k])
                slots = [int(rr.choice(self.release_all)) for _ in range(k)]
            if search == 'exact' and deployment != 'random':
                best, log = self.exact(k, deployment, objective)
            else:
                best, log = self.local(k, deployment, objective, slots=slots)
            per_k[k], logs[k] = best, log
        Js = {k: per_k[k][0] for k in per_k}
        Jstar = min(Js.values())
        if n_fixed:
            k_sel = n_fixed
        else:
            ok = [k for k in sorted(Js) if Js[k] <= (1 + self.eps) * Jstar + 1e-9]
            k_sel = ok[0]
        J, tours, nodes, order = per_k[k_sel]
        pred, routes = self.predicted(tours, nodes, order)
        cfg = Configuration(method=method, n=int(k_sel), tours=[list(map(int, t)) for t in tours],
                            release_stations=[int(x) for x in nodes], release_order=[int(x) for x in order],
                            release_times=[float(x) if x is not None else None for x in pred['release_times_pred']],
                            routes=[[int(s) for s in r] for r in routes], predicted=pred,
                            certificate=logs[k_sel]['certificate'], planning_time_s=float(time.monotonic() - t0),
                            search=dict(objective=objective, deployment=deployment, search=search,
                                        topology=self.topology, J_by_n={str(k): float(v) for k, v in Js.items()},
                                        logs={str(k): v for k, v in logs.items()}, sim_calls=self.sim_calls,
                                        epsilon=self.eps, n_candidates=ks))
        return cfg
