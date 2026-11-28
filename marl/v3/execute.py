"""Execute a global configuration in the simulator with a fixed local controller.

Pipeline per 0.1 s control step (identical for every configuration arm):
  catheter release gate -> image observation (1-step latency) -> pure-pursuit rule toward the
  current target -> release-aware TPG hold -> SettleGuard -> wall SDF projection -> spacing shield
  -> patient-level actuation variation -> physics.
Only the configuration (N, release stations, allocation, release schedule) and the coordination
switch differ between arms.  A held cluster receives a zero command; in flow this is NOT station
keeping (it drifts), which the executed metrics include.
"""
from __future__ import annotations

import time

import numpy as np

from marl.v3 import assumptions as A
from marl.v3.metrics import FourMetricRecorder


def release_aware_tpg(env, tours, station_path, d_min, release_s):
    from marl.tpg_coordinator import TPGCoordinator

    class ReleaseAwareTPG(TPGCoordinator):
        """TPG whose offline schedule starts every cluster at its planned release time."""
        def __init__(self, *a, release_offsets=None, **kw):
            self._offsets = np.asarray(release_offsets, float)
            super().__init__(*a, **kw)

        def _timeline(self, i, waits):
            return super()._timeline(i, waits) + self._offsets[i]

    return ReleaseAwareTPG(env, [list(t) for t in tours], station_path, d_min_mm=d_min, dwell_s=0.,
                           release_offsets=release_s)


def run_configuration(scene, config, *, tpg=True, shield=True, record_trace=False, local='pursuit'):
    from scripts.benchmark_lysis import LysisEpisode, SettleGuard
    from marl.deployable_sensing import DeployableConfig
    from marl.vessel_sdf import VesselSDF, project_local_commands
    from marl.v3.scene import FLOW_MODES, attach_variation
    cfg = A.load()
    task = scene.task
    n = int(config.n)
    deadline = float(cfg['mission']['deadline_s'])
    d_safe = float(cfg['safety']['d_safe_mm']); tol = float(cfg['safety']['arrival_tolerance_mm'])
    clear = float(cfg['deployment']['release_clearance_mm'])
    flow, pulsatile = FLOW_MODES[task.flow_mode]
    started = time.monotonic()
    ep = LysisEpisode(n, task.anatomy, task.seed, horizon=deadline, d_min=d_safe, sensing='image', shield=shield,
                      tpg=False, navigation=True, flow_inlet_mm_s=flow, dynamic_obstacle_count=task.obstacles,
                      sense_cfg=DeployableConfig(latency_steps=1), env_factory=scene.env_factory(config.release_stations))
    if task.variation > 0:
        attach_variation(ep, task.variation, pulsatile, int(cfg['resources']['n_max']))
    env = ep.env
    tours = [list(t) for t in config.tours]
    owner = [None] * len(scene.targets)
    for i, tour in enumerate(tours):
        for t in tour:
            owner[t] = i
    if any(o is None for o in owner):
        raise ValueError('configuration does not cover every target')
    coord = None
    if tpg and n > 1:
        _, station_path = ep.bm._station_paths(env)
        rel = [x if x is not None else 0. for x in config.release_times]
        coord = release_aware_tpg(env, tours, station_path, d_safe, rel)
        ep.coord = coord
    guard = SettleGuard(ep)
    sdf = VesselSDF(ep.sensor)
    rec = FourMetricRecorder(n, len(scene.targets), d_safe=d_safe, body_radius=float(env.config.robot_radius_mm),
                             deadline=deadline, tol=tol, target_owner=owner)
    next_index = np.zeros(n, int)
    current = np.asarray([t[0] for t in tours], int)
    order = list(config.release_order); rel_ptr = 0
    planned = [x if x is not None else 0. for x in config.release_times]
    release_pts = env.transport.points[np.asarray(config.release_stations, int)]
    trace = [] if record_trace else None
    done_env = False
    while True:
        t_now = float(env.elapsed_s)
        while rel_ptr < n:                                    # catheter: releases strictly in planned order
            c = order[rel_ptr]
            if t_now + 1e-9 < planned[c]:
                break
            others = [o for o in range(n) if env.active[o]]
            if any(np.linalg.norm(env.positions_mm[o] - release_pts[c]) < clear for o in others):
                break
            env.release(c); rec.mark_release(c, t_now); rel_ptr += 1
        est = ep.observe()
        targets = current.copy()
        finished = next_index >= np.asarray([len(t) for t in tours])
        targets[finished] = -1                               # delivered clusters stop (zero command)
        ep.navigation_targets = np.where(targets >= 0, targets, current)
        safe_targets = np.where(targets >= 0, targets, current)
        rule = ep.ctl.act(safe_targets, est)
        hold = coord.gate(est.pos, est.active) if coord is not None else np.zeros(n, bool)
        hold = np.asarray(hold, bool).copy()
        guarded = guard(ep, est, safe_targets, rule, hold)
        frames = ep.ctl.frames(est)
        local, _ = project_local_commands(guarded, est, targets, hold, frames, sdf, ep.ctl.body)
        # per-cluster wall rewrite flags (project_local_commands only returns a count)
        pre = np.clip(np.asarray(guarded, float), -1., 1.)
        wall_rw = np.array([bool(est.active[i] and not hold[i] and targets[i] >= 0 and
                                 np.linalg.norm(pre[i] - local[i]) > 1e-6) for i in range(n)])
        pre_shield = local.copy(); pre_shield[hold] = 0.
        active_before = env.active[:n].copy()
        done_env, out = ep.step(est, local, hold)
        shield_changed = np.linalg.norm(ep.prev_local - pre_shield, axis=1) > 1e-9
        info = ep.info
        dt = float(info['step_duration_s']); t_now = float(env.elapsed_s)
        _, lumen, radial, _ = env.transport.coordinates(env.positions_mm[:n], env.edges[:n], env.solution)
        clearance = lumen - radial - env.body_radius[:n]
        dist = env._target_distances()[np.arange(n), current]
        for i in range(n):
            if next_index[i] >= len(tours[i]) or not env.active[i]:
                continue
            if np.isfinite(dist[i]) and dist[i] <= tol:
                rec.mark_arrival(int(tours[i][next_index[i]]), t_now)
                next_index[i] += 1
                if next_index[i] < len(tours[i]):
                    current[i] = tours[i][next_index[i]]
        delivered_mask = (next_index >= np.asarray([len(t) for t in tours])).astype(float)
        rec.update(t_now, dt, positions=env.positions_mm[:n], active=env.active[:n] & active_before,
                   wall_s=out['wall'], obstacle_contact_s=np.asarray(info.get('particle_contact_s', np.zeros(n)), float),
                   obstacle_events=np.asarray(info.get('particle_collision_events', np.zeros(n)), int),
                   lost_now=out['lost'], clearance=clearance, hold=hold, shield_changed=shield_changed,
                   wall_rewritten=wall_rw, cmd_world=ep.sent_world, actual_velocity=env.velocity_mm_s,
                   delivered_mask=delivered_mask)
        if trace is not None:
            trace.append(dict(t=t_now, pos=env.positions_mm[:n].round(4).tolist(), active=env.active[:n].tolist(),
                              parked=env.parked.tolist(), hold=hold.tolist(), shield=shield_changed.tolist(),
                              wall_rw=wall_rw.tolist(), wall=np.asarray(out['wall']).round(4).tolist(),
                              clearance=np.asarray(clearance).round(4).tolist(),
                              cmd=np.asarray(ep.sent_world).round(4).tolist(),
                              vel=np.asarray(env.velocity_mm_s).round(4).tolist(), target=current.tolist()))
        all_done = bool(np.all(np.isfinite(rec.arrival)))
        remaining = [j for j in range(len(scene.targets)) if not np.isfinite(rec.arrival[j])]
        hopeless = bool(remaining) and all(np.isfinite(rec.lost_time[owner[j]]) for j in remaining)
        if all_done or hopeless or done_env or t_now >= deadline - 1e-9:
            break
    sp = ep.spacing.summary()
    summary = rec.summary(substep_violation_pair_s=float(sp['spacing_violation_pair_s']))
    summary.update(seconds=float(time.monotonic() - started), elapsed_s=float(env.elapsed_s),
                   tpg=bool(coord is not None), shield=bool(shield), local=local,
                   tpg_zones=len(coord.zones) if coord is not None else 0)
    ep.close()
    return summary, trace
