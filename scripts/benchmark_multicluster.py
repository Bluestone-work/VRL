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
               'route_follow': 'privileged', 'local_follow': 'fair'}


def station_geodesic(env):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    tree = env.tree; n = tree.n_stations; scale = float(tree.physical_mm_per_unit)
    rows, cols, w = zip(*[(i, j, d*scale) for i, nb in enumerate(tree.station_graph) for j, d in nb])
    return shortest_path(csr_matrix((w, (rows, cols)), shape=(n, n)), directed=False)


def preoperative_plan(env):
    """Allocation A: ordered clot list per cluster minimising (makespan, total) of geodesic travel."""
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
    for labels in itertools.product(range(n), repeat=m):
        tours = [tour(i, [j for j in range(m) if labels[j] == i]) for i in range(n)]
        key = (max(t[0] for t in tours), sum(t[0] for t in tours))
        if best is None or key < best[0]:
            best = (key, [list(t[1]) for t in tours])
    return best[1], dict(makespan_mm=float(best[0][0]), total_mm=float(best[0][1]))


class PlanTargets:
    """Current target of each cluster: first alive clot of its plan, else the nearest alive clot."""
    def __init__(self, plan):
        self.plan = plan

    def targets(self, env, positions):
        out = np.full(len(self.plan), -1, np.int64)
        alive = env.masses > 0
        for i, seq in enumerate(self.plan):
            nxt = [j for j in seq if alive[j]]
            if nxt:
                out[i] = nxt[0]
            elif alive.any():
                d = np.linalg.norm(env.clot_positions_mm-positions[i], axis=1)
                out[i] = int(np.argmin(np.where(alive, d, np.inf)))
        return out


def slots_for(packet, targets):
    slots = np.full(len(targets), -1, np.int32)
    for i, t in enumerate(targets):
        hit = np.flatnonzero(packet.clot_ids[i] == t)
        slots[i] = hit[0] if len(hit) else -1
    return slots


class Shield(MultiClusterController):
    """Spacing filter + right-of-way bookkeeping of MultiClusterController, applied to any nominal command."""
    def filtered(self, local, packet):
        n = self.config.clusters
        yielding = np.zeros(n, bool)
        if n > 1 and self.config.min_spacing_mm > 0:
            local, yielding = self._spacing_filter(local, packet)
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
    base = replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy=anatomy, episode_duration_s=horizon_s)
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
               'local_follow': LocalFollower}.get(method, lambda e: None)(env)
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
            elif method in ('route_pursuit', 'route_follow'):
                local = pursuit.act(tgt)
            elif method in ('local_pursuit', 'local_follow'):
                local = pursuit.act(tgt, packet.navigation)
            elif method == 'plan_reactive':
                local = fair_reactive_action(packet.navigation, 'path', PartialObsConfig(), target_slots=slots_for(packet, tgt))
            else:
                raise ValueError(method)
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
    row = dict(method=method, information=INFORMATION[method], clusters=clusters, anatomy=anatomy, seed=seed,
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
    a = p.parse_args()
    s0, s1 = map(int, a.seeds.split(':'))
    with a.out.open('a') as f:
        for seed in range(s0, s1+1):
            f.write(json.dumps(run_episode(a.method, a.clusters, a.anatomy, seed, a.horizon_s, a.d_min_mm))+'\n'); f.flush()


if __name__ == '__main__':
    main()
