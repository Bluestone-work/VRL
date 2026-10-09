"""Unified single-/multi-cluster benchmark (2026-10-05): every method, same scenes, same metrics.

Task: N clusters (N=1 sequential, N=2/3 parallel) clear all clots of one anatomy. Scenes come from
`paired_environment` (identical geometry, clots, particles and nested starts across N and methods;
fixed total catalytic rate). High-level allocation A: a pre-operative plan computed ONCE at t=0 from
the vessel map (geodesic distances) - each cluster gets an ordered clot list minimising the makespan
(ties: total length). Navigation must then follow from the method's own information.

Methods (information class in brackets)
  plan_route        [privileged] plan A + known-map route following + particle avoid/wait (teacher rule)
  plan_reactive     [fair]       plan A + fair local reactive path controller (marl.fair_reactive)
  nearest_reactive  [fair]       no allocation: nearest clot, fair reactive path controller
For N>=2 every method passes through the same spacing filter / right-of-way rule (MultiClusterController).

Metrics: cluster-safe success (all cleared, wall < 1 cluster-s, no particle/pair contact, no cluster
lost, spacing compliant), raw success, removal, time to 50/90/100 % removal (censored at the horizon),
normalised removal AUC, total path length, wall contact (total, ratio, longest run), particle contact,
spacing violation time and minimum spacing, yield / prolonged-yield (deadlock proxy) events, timeout.
usage: benchmark_multicluster.py --method M --clusters N --anatomy A --seeds S0:S1 --out JSONL
"""
from __future__ import annotations
import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from environments.mca_physical_env import DynamicsConfig
from marl.fair_reactive import fair_reactive_action
from marl.multicluster import MultiClusterConfig, MultiClusterController
from marl.multicluster_observation import ClusterSensorAdapter
from marl.partial_obs import PartialObsConfig
from marl.teacher import TEACHER_CONFIG, teacher_config, teacher_label
from scripts.multicluster_protocol import SpacingTracker, attach_spacing_monitor, paired_environment
from scripts.safe_metrics import WallTracker, episode_metrics

INFORMATION = {'plan_route': 'privileged', 'plan_reactive': 'fair', 'nearest_reactive': 'fair',
               'route_pursuit': 'privileged', 'local_pursuit': 'fair',
               'route_follow': 'privileged', 'local_follow': 'fair',
               'bc_graph': 'privileged', 'bc_local': 'fair', 'local_memory': 'fair', 'local_tabu': 'fair', 'local_learned': 'fair', 'rl_local': 'fair',
               'route_tpg': 'privileged', 'route_tpg_noshield': 'privileged'}
STUDENT = {}   # checkpoint path per learned method, set from --checkpoint


def station_geodesic(env):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    tree = env.tree; n = tree.n_stations; scale = float(tree.physical_mm_per_unit)
    rows, cols, w = zip(*[(i, j, d*scale) for i, nb in enumerate(tree.station_graph) for j, d in nb])
    return shortest_path(csr_matrix((w, (rows, cols)), shape=(n, n)), directed=False)


ALLOCATION = {'mode': 'makespan', 'lambda': 1., 'site_mm': 2.5}


def _station_paths(env):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    tree = env.tree; n = tree.n_stations; scale = float(tree.physical_mm_per_unit)
    rows, cols, w = zip(*[(i, j, d*scale) for i, nb in enumerate(tree.station_graph) for j, d in nb])
    G, pred = shortest_path(csr_matrix((w, (rows, cols)), shape=(n, n)), directed=False, return_predecessors=True)
    def path(a, b):
        out = [b]
        while out[-1] != a and pred[a, out[-1]] >= 0:
            out.append(int(pred[a, out[-1]]))
        return out
    return G, path


def conflict_penalty(env, tours, G, path):
    """Vessel-occupancy conflicts of a plan (mm): length of cluster i's route inside the work site
    (within site_mm geodesic) of a clot assigned to another cluster, plus half the corridor length
    two clusters' routes share. Both measured on the pre-operative map."""
    starts = np.asarray(env.robot_stations); clots = np.asarray(env.clot_stations)
    spacing = float(np.median([d for nb in env.tree.station_graph for _, d in nb]))*float(env.tree.physical_mm_per_unit)
    routes = []
    for i, seq in enumerate(tours):
        st, cur = set(), int(starts[i])
        for c in seq:
            st |= set(path(cur, int(clots[c]))); cur = int(clots[c])
        routes.append(st)
    pen = 0.
    for i in range(len(tours)):
        for j in range(len(tours)):
            if i == j:
                continue
            for c in tours[j]:
                site = np.flatnonzero(G[int(clots[c])] < ALLOCATION['site_mm'])
                pen += spacing*len(routes[i] & set(site.tolist()))
            if j > i:
                pen += .5*spacing*len(routes[i] & routes[j])
    return pen


def preoperative_plan(env):
    """Allocation A: ordered clot list per cluster minimising (makespan, total) of geodesic travel.
    mode 'conflict': minimise makespan + lambda * conflict_penalty instead (N>1)."""
    G = station_geodesic(env)
    starts = np.asarray(env.robot_stations); clots = np.asarray(env.clot_stations)
    n, m = len(starts), len(clots)
    def tour(i, subset):
        if not subset:
            return 0., ()
        best = (np.inf, ())
        for order in itertools.permutations(subset):
            length = G[starts[i], clots[order[0]]] + sum(G[clots[a], clots[b]] for a, b in zip(order, order[1:]))
            best = min(best, (length, order))
        return best
    best = None
    conflict = ALLOCATION['mode'] == 'conflict' and n > 1
    if conflict:
        Gs, spath = _station_paths(env)
    for labels in itertools.product(range(n), repeat=m):
        tours = [tour(i, [j for j in range(m) if labels[j] == i]) for i in range(n)]
        key = (max(t[0] for t in tours), sum(t[0] for t in tours))
        if conflict:
            key = (key[0]+ALLOCATION['lambda']*conflict_penalty(env, [list(t[1]) for t in tours], Gs, spath), key[1])
        if best is None or key < best[0]:
            best = (key, [list(t[1]) for t in tours])
    return best[1], dict(makespan_mm=float(best[0][0]), total_mm=float(best[0][1]))


FALLBACK = {'mode': 'help'}   # cluster whose own plan is done: 'help' -> nearest alive clot, 'hold' -> stay


class PlanTargets:
    """Current target of each cluster: first alive clot of its plan; afterwards FALLBACK."""
    def __init__(self, plan):
        self.plan = plan

    def targets(self, env, positions, clot_alive=None):
        out = np.full(len(self.plan), -1, np.int64)
        alive = np.asarray(env.masses > 0 if clot_alive is None else clot_alive, bool)
        for i, seq in enumerate(self.plan):
            nxt = [j for j in seq if alive[j]]
            if nxt:
                out[i] = nxt[0]
            elif seq and FALLBACK['mode'] == 'park' and alive.any():
                out[i] = seq[-1]              # station-keep at its last (cleared) clot: the schedule parks it there
            elif alive.any() and FALLBACK['mode'] == 'help':
                d = np.linalg.norm(env.clot_positions_mm-positions[i], axis=1)
                out[i] = int(np.argmin(np.where(alive, d, np.inf)))
        return out


def slots_for(packet, targets):
    slots = np.full(len(targets), -1, np.int32)
    for i, t in enumerate(targets):
        hit = np.flatnonzero(packet.clot_ids[i] == t)
        slots[i] = hit[0] if len(hit) else -1
    return slots


SHIELD_MODE = {'mode': 'project'}
BACKOFF_SPEED = .5
JUNCTION = {'model': 'graph'}
TAG = {'tag': None}   # 'project' (baseline) | 'tube' (longitudinal-only conflict resolution)


class Shield(MultiClusterController):
    """Spacing filter + right-of-way bookkeeping of MultiClusterController, applied to any nominal command.

    mode 'project' (baseline): lower-priority cluster stops on a predicted conflict and every command is
    projected onto the linear clearance constraints, which deflects clusters sideways.
    mode 'tube' (proposed): a vessel is a quasi-1-D tube, so sideways deflection mostly ends at the wall.
    Keep the filter's along-vessel (Frenet tangent) correction only, i.e. resolve conflicts by slowing,
    waiting or backing along the lumen; the lateral part of the nominal command is restored.
    """
    def filtered(self, local, packet):
        n = self.config.clusters
        yielding = np.zeros(n, bool)
        if n > 1 and self.config.min_spacing_mm > 0:
            nominal = local.copy()
            local, yielding = self._spacing_filter(local, packet)
            if SHIELD_MODE['mode'] in ('tube', 'backoff'):
                lateral = nominal.copy(); lateral[:, 0] = 0.               # Frenet frame: column 0 = tangent
                local = np.column_stack((local[:, 0], lateral[:, 1:]))
                local[yielding] = 0.
                if SHIELD_MODE['mode'] == 'backoff':
                    # Yielding cluster retreats along its lumen axis, away from the nearest visible peer that
                    # it conflicts with, so a head-on pair in one tube can pass at the next junction.
                    for i in np.flatnonzero(yielding):
                        vis = np.flatnonzero(packet.peer_visible[i])
                        if len(vis):
                            j = vis[np.argmin(np.linalg.norm(packet.peer_relative_mm[i, vis], axis=1))]
                            along = packet.peer_relative_mm[i, j, 0]
                            local[i] = np.array([-np.sign(along) if abs(along) > 1e-6 else 0., 0., 0.])*BACKOFF_SPEED
                local /= np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1.)
        local[~packet.active] = 0.
        self.yield_events += int(np.sum(yielding & ~self.previous_yielding))
        self.yield_agent_steps += int(yielding.sum())
        stalled = yielding & (np.linalg.norm(packet.navigation[:, 3:6], axis=1)*self.speed < .05)
        self.wait_steps = np.where(stalled, self.wait_steps+1, 0)
        prolonged = self.wait_steps == self.config.deadlock_window_steps
        self.persistent_yield_events += int(prolonged.sum())
        if np.any(prolonged):
            self.ranks = (self.ranks+1) % n
        self.previous_yielding = yielding
        return local


def run_episode(method, clusters, anatomy, seed, horizon_s, d_min, control_seed=42):
    t0 = time.monotonic()
    base = replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy=anatomy, episode_duration_s=horizon_s,
                   junction_model=JUNCTION['model'])
    env, manifest = paired_environment(base, clusters, seed)
    mc = MultiClusterConfig(method='multi_parallel' if clusters > 1 else 'single_sequential', clusters=clusters,
                            min_spacing_mm=d_min if clusters > 1 else 0., control_seed=control_seed)
    sensor = ClusterSensorAdapter(env, mc)
    sensor.reset(int(np.random.SeedSequence([seed, control_seed]).generate_state(1)[0]))
    shield = Shield(mc, robot_speed_mm_s=env.config.robot_speed_mm_s, control_dt_s=env.config.control_dt_s)
    nearest = MultiClusterController(mc, robot_speed_mm_s=env.config.robot_speed_mm_s, control_dt_s=env.config.control_dt_s)
    plan, plan_info = preoperative_plan(env)
    from marl.pursuit_controllers import LocalPursuit, RoutePursuit
    from marl.edge_follower import LocalFollower, RouteFollower
    pursuit = {'route_pursuit': RoutePursuit, 'local_pursuit': LocalPursuit, 'route_follow': RouteFollower,
               'local_follow': LocalFollower, 'local_memory': lambda e: LocalFollower(e, memory=True),
               'local_tabu': lambda e: LocalFollower(e, tabu_s=30.),
               'local_learned': lambda e: LocalFollower(e, tabu_s=30., scorer=__import__('marl.frontier_selector', fromlist=['x']).make_scorer(STUDENT['local_learned']))}.get(method, lambda e: None)(env)
    if method == 'bc_graph':
        from marl.graph_transformer_student import load_student, student_local_action
        from marl.scene_graph import extract_scene
        gmodel = load_student(STUDENT[method])
    coord = None
    if method in ('route_tpg', 'route_tpg_noshield'):
        from marl.edge_follower import RouteFollower
        from marl.tpg_coordinator import TPGCoordinator
        pursuit = RouteFollower(env)
        _, spath = _station_paths(env)
        coord = TPGCoordinator(env, plan, spath, d_min_mm=d_min) if clusters > 1 else None
        FALLBACK['mode'] = 'park'      # the schedule parks finished clusters; re-targeting would break it
    if method == 'rl_local':
        import torch
        from scripts.train_cluster_ppo import Policy, execute as rl_execute
        rl = Policy(); rl.load_state_dict(torch.load(STUDENT[method], map_location='cpu', weights_only=False)['state']); rl.eval()
    if method == 'bc_local':
        from marl.local_student import load_local, local_student_action
        lmodel = load_local(STUDENT[method])
    targets = PlanTargets(plan)
    tcfg = teacher_config(env.config)
    walls = WallTracker(clusters)
    spacing = SpacingTracker(env.positions_mm[:clusters], env.active[:clusters], mc.min_spacing_mm if clusters > 1 else 0.)
    attach_spacing_monitor(env, spacing)
    initial = float(env.initial_mass.sum())
    milestone = {50: None, 90: None, 100: None}; auc = 0.; pair = 0.
    while True:
        packet = sensor.observe()
        if method == 'nearest_reactive':
            local, _ = nearest.act(packet)
        else:
            tgt = targets.targets(env, env.positions_mm[:clusters])
            if method == 'plan_route':
                env._assignment = tgt.copy(); env._assignment_alive = np.flatnonzero(env.masses > 0).tobytes()
                local, stop, _ = teacher_label(env, tcfg)
                local = np.where(stop[:, None], 0., local).astype(np.float64)
            elif method == 'bc_graph':
                scene = extract_scene(env); scene['robot_goal'] = np.where(env.active[:clusters], tgt, -1)
                local = student_local_action(gmodel, [scene])[0].astype(np.float64)
            elif method == 'rl_local':
                sl = slots_for(packet, tgt)
                with torch.no_grad():
                    d, _ = rl.dist(torch.as_tensor(packet.navigation, dtype=torch.float32), torch.as_tensor(sl, dtype=torch.long))
                local = rl_execute(d.mean.numpy().astype(np.float64)); local[sl < 0] = 0.
            elif method == 'bc_local':
                local = local_student_action(lmodel, packet.navigation, slots_for(packet, tgt))
            elif method in ('route_pursuit', 'route_follow', 'route_tpg', 'route_tpg_noshield'):
                local = pursuit.act(tgt)
                if coord is not None:
                    local[coord.gate()] = 0.
            elif method in ('local_pursuit', 'local_follow', 'local_memory', 'local_tabu', 'local_learned'):
                local = pursuit.act(tgt, packet.navigation)
            elif method == 'plan_reactive':
                local = fair_reactive_action(packet.navigation, 'path', PartialObsConfig(), target_slots=slots_for(packet, tgt))
            else:
                raise ValueError(method)
            if method != 'route_tpg_noshield':
                local = shield.filtered(local, packet)
        before = 1-float(env.masses.sum())/initial
        active = env.active[:clusters].copy()
        spacing.begin_step()
        _, _, term, trunc, info = env.step(sensor.execute(local))
        spacing.end_step()
        dt = float(info['step_duration_s'])
        walls.update(info, active, dt); pair += float(info['robot_pair_contact_s'])
        removal = 1-float(info['remaining_mass'])/initial
        auc += .5*(before+removal)*dt
        for k in milestone:
            if milestone[k] is None and removal >= k/100-1e-12:
                milestone[k] = float(info['elapsed_s'])
        if term or trunc:
            break
    m = episode_metrics(info, walls, initial)
    sp = spacing.summary()
    ctl = nearest if method == 'nearest_reactive' else shield
    safe = bool(m['safe_collision_free'] and sp['spacing_compliant'] and pair <= 1e-12)
    tag = method + ('' if SHIELD_MODE['mode'] == 'project' else '+'+SHIELD_MODE['mode']) + ('' if FALLBACK['mode'] == 'help' else '+hold') \
          + ('' if ALLOCATION['mode'] == 'makespan' else f"+conflict{ALLOCATION['lambda']:g}")
    row = dict(method=TAG['tag'] or tag, junction_model=JUNCTION['model'], shield=SHIELD_MODE['mode'], fallback=FALLBACK['mode'], information=INFORMATION[method], checkpoint=STUDENT.get(method), clusters=clusters, anatomy=anatomy, seed=seed,
               horizon_s=horizon_s, d_min_mm=d_min, scenario_hash=manifest['scenario_hash'],
               plan=plan, plan_makespan_mm=plan_info['makespan_mm'], plan_total_mm=plan_info['total_mm'],
               cluster_safe_success=safe, **m, robot_pair_contact_s=pair,
               particle_contact_s=float(info['episode_particle_contact_s']),
               particle_events=int(info['episode_particle_collision_events']), lost=int(info['lost_robots']),
               **{f't{k}_s': milestone[k] for k in milestone}, removal_auc=auc/horizon_s,
               path_mm=float(np.sum(info['robot_path_mm'])), spacing=sp,
               yield_events=ctl.yield_events, prolonged_yield_events=ctl.persistent_yield_events,
               walltime_s=time.monotonic()-t0)
    env.close()
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', required=True, choices=sorted(INFORMATION))
    p.add_argument('--clusters', type=int, required=True); p.add_argument('--anatomy', required=True)
    p.add_argument('--seeds', required=True, help='first:last (inclusive)')
    p.add_argument('--horizon-s', type=float, default=300.); p.add_argument('--d-min-mm', type=float, default=2.)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--checkpoint', help='learned methods')
    p.add_argument('--tag', help='method label in the output rows')
    p.add_argument('--junction', default='graph', choices=('graph', 'union'))
    p.add_argument('--shield', default='project', choices=('project', 'tube', 'backoff'))
    p.add_argument('--fallback', default='help', choices=('help', 'hold', 'park'))
    p.add_argument('--allocation', default='makespan', choices=('makespan', 'conflict'))
    p.add_argument('--conflict-lambda', type=float, default=1.)
    a = p.parse_args()
    if a.checkpoint:
        STUDENT[a.method] = a.checkpoint
    SHIELD_MODE['mode'] = a.shield; FALLBACK['mode'] = a.fallback
    ALLOCATION['mode'] = a.allocation; ALLOCATION['lambda'] = a.conflict_lambda
    TAG['tag'] = a.tag; JUNCTION['model'] = a.junction
    s0, s1 = map(int, a.seeds.split(':'))
    with a.out.open('a') as f:
        for seed in range(s0, s1+1):
            f.write(json.dumps(run_episode(a.method, a.clusters, a.anatomy, seed, a.horizon_s, a.d_min_mm))+'\n'); f.flush()


if __name__ == '__main__':
    main()
