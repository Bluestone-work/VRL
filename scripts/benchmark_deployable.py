"""Benchmark episodes under the deployable information model (marl.deployable_sensing).

Same scenes, physics, allocation A, metrics and safety definition as scripts/benchmark_multicluster.py.
Controllers read only DeployableSensor estimates and the pre-operative map.
  route_follow_dep   I0 edge-level junction-aware follower on estimates
  route_tpg_dep      + spacing-safe TPG coordinator (zones precomputed on the map; progress from estimates)
usage: benchmark_deployable.py --method M --clusters N --anatomy A --seeds S0:S1 --out JSONL [--sigma 0.05]
"""
from __future__ import annotations
import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from environments.mca_physical_env import DynamicsConfig
from marl.deployable_sensing import DeployableConfig, DeployableSensor
from marl.edge_follower import DeployableRouteFollower
from marl.multicluster import ClusterObservation, MultiClusterConfig
from marl.teacher import TEACHER_CONFIG
import scripts.benchmark_multicluster as bm
from scripts.multicluster_protocol import SpacingTracker, attach_spacing_monitor, paired_environment
from scripts.safe_metrics import WallTracker, episode_metrics


def packet_from(est, F, prev_local):
    """Spacing-filter input built from estimates only (ego frames from the map-matched stations)."""
    n = len(est.pos)
    nav = np.zeros((n, 111), np.float32)
    nav[:, :3] = prev_local
    nav[:, 3:6] = np.einsum('nij,nj->ni', F, est.vel)
    rel = np.einsum('iab,ijb->ija', F, est.peers_rel)
    relv = np.einsum('iab,ijb->ija', F, est.vel[None, :, :]-est.vel[:, None, :])*est.peers_vis[..., None]
    return ClusterObservation(nav, np.full((n, 4), -1, np.int32), rel, relv, est.peers_vis, est.active)


def run(method, n, anatomy, seed, horizon, d_min, cfg):
    t0 = time.monotonic()
    base = replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy=anatomy, episode_duration_s=horizon)
    env, manifest = paired_environment(base, n, seed)
    sensor = DeployableSensor(env, cfg, seed=seed)
    mc = MultiClusterConfig(method='multi_parallel' if n > 1 else 'single_sequential', clusters=n,
                            min_spacing_mm=d_min if n > 1 else 0.)
    shield = bm.Shield(mc, robot_speed_mm_s=env.config.robot_speed_mm_s, control_dt_s=env.config.control_dt_s)
    plan, info_plan = bm.preoperative_plan(env)
    bm.FALLBACK['mode'] = 'park' if method == 'route_tpg_dep' else 'help'
    targets = bm.PlanTargets(plan)
    ctl = DeployableRouteFollower(env, sensor)
    coord = None
    if method == 'route_tpg_dep' and n > 1:
        from marl.tpg_coordinator import TPGCoordinator
        _, sp = bm._station_paths(env)
        coord = TPGCoordinator(env, plan, sp, d_min_mm=d_min)
    walls = WallTracker(n)
    spacing = SpacingTracker(env.positions_mm[:n], env.active[:n], mc.min_spacing_mm if n > 1 else 0.)
    attach_spacing_monitor(env, spacing)
    initial = float(env.initial_mass.sum()); ms = {50: None, 90: None, 100: None}; auc = 0.; pair = 0.
    prev_local = np.zeros((n, 3))
    while True:
        est = sensor.observe()
        tgt = targets.targets(env, est.pos)
        local = ctl.act(tgt, est)
        if coord is not None:
            local[coord.gate(est.pos, est.active)] = 0.
        F = ctl.frames(est)
        if n > 1:
            local = shield.filtered(local, packet_from(est, F, prev_local))
        prev_local = local.copy()
        before = 1-float(env.masses.sum())/initial; active = env.active[:n].copy()
        spacing.begin_step()
        _, _, term, trunc, info = env.step(ctl.to_world(local, est))
        spacing.end_step()
        walls.update(info, active, float(info['step_duration_s'])); pair += float(info['robot_pair_contact_s'])
        removal = 1-float(info['remaining_mass'])/initial
        auc += .5*(before+removal)*float(info['step_duration_s'])
        for k in ms:
            if ms[k] is None and removal >= k/100-1e-12:
                ms[k] = float(info['elapsed_s'])
        if term or trunc:
            break
    m = episode_metrics(info, walls, initial); sp = spacing.summary()
    row = dict(method=method, information='deployable', clusters=n, anatomy=anatomy, seed=seed, horizon_s=horizon,
               d_min_mm=d_min, sensing=asdict(cfg), scenario_hash=manifest['scenario_hash'], plan=plan,
               plan_makespan_mm=info_plan['makespan_mm'],
               cluster_safe_success=bool(m['safe_collision_free'] and sp['spacing_compliant'] and pair <= 1e-12), **m,
               robot_pair_contact_s=pair, particle_contact_s=float(info['episode_particle_contact_s']),
               particle_events=int(info['episode_particle_collision_events']), lost=int(info['lost_robots']),
               t50_s=ms[50], t90_s=ms[90], t100_s=ms[100], removal_auc=auc/horizon,
               path_mm=float(np.sum(info['robot_path_mm'])), spacing=sp, yield_events=shield.yield_events,
               prolonged_yield_events=shield.persistent_yield_events, walltime_s=time.monotonic()-t0)
    env.close()
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', required=True, choices=('route_follow_dep', 'route_tpg_dep'))
    p.add_argument('--clusters', type=int, required=True); p.add_argument('--anatomy', required=True)
    p.add_argument('--seeds', required=True); p.add_argument('--horizon-s', type=float, default=300.)
    p.add_argument('--d-min-mm', type=float, default=2.); p.add_argument('--sigma', type=float, default=.05)
    p.add_argument('--latency', type=int, default=1); p.add_argument('--tag')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    cfg = DeployableConfig(position_sigma_mm=a.sigma, latency_steps=a.latency)
    s0, s1 = map(int, a.seeds.split(':'))
    with a.out.open('a') as f:
        for seed in range(s0, s1+1):
            r = run(a.method, a.clusters, a.anatomy, seed, a.horizon_s, a.d_min_mm, cfg)
            if a.tag:
                r['method'] = a.tag
            f.write(json.dumps(r, default=str)+'\n'); f.flush()


if __name__ == '__main__':
    main()
