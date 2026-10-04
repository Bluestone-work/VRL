"""Shared EXP0046 scenes, resource conventions, and evaluation-only metrics."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np

from environments.mca_physical_env import MCAPhysicalEnv

REVISION = 'EXP0046_v2_observation_paired'


def reserved_seed(seed):
    splits=json.loads((Path(__file__).resolve().parents[1]/'configs/evaluation_splits.json').read_text())
    return any(v['seed_base'] <= seed < v['seed_base']+v['count']
               for a in splits['anatomies'].values() for v in a.values()
               if isinstance(v,dict) and 'seed_base' in v)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def paired_environment(base, clusters, seed, *, budget='fixed_total', pool_size=3, spawn_clearance_mm=4.):
    """Generate one immutable scene at max N, reuse nested robot starts.

    Clots, particles, geometry, flow and candidate starts are identical across
    N, d_min, policies and budget arms. The single-cluster entry point rotates
    across the same start pool by scene seed. Excluded initial states and their
    seeds are logged. This is a pre-deployed-cluster comparison, not insertion.
    """
    if not 1 <= clusters <= pool_size or budget not in ('fixed_total', 'per_cluster'):
        raise ValueError('Invalid cluster count or catalytic budget')
    if reserved_seed(seed):
        raise ValueError('This diagnostic runner rejects registered evaluation seeds')
    base = replace(base, num_robots=pool_size, robot_initialization='distributed_branches',
                   action_prior='none', action_shield_horizon_s=0., command_speed='bounded',
                   target_observation='routed_assigned_own', progress_reward_scale=0.)
    rejected = []
    for attempt in range(100):
        accepted = seed if attempt == 0 else int(np.random.SeedSequence([seed, 4600, attempt]).generate_state(1)[0])
        if reserved_seed(accepted):
            rejected.append(dict(seed=accepted, reason='reserved_seed_excluded_before_reset'))
            continue
        scene = MCAPhysicalEnv(base)
        try:
            scene.reset(seed=accepted)
        except ValueError as exc:
            scene.close()
            if str(exc) != 'Initial body does not fit the obstructed lumen':
                raise
            rejected.append(dict(seed=accepted, reason='invalid_body_placement'))
            continue
        starts = scene.positions_mm[:pool_size].copy()
        distance = np.linalg.norm(starts[:, None]-starts[None, :], axis=2)
        if pool_size > 1 and float(distance[np.triu_indices(pool_size, 1)].min()) < spawn_clearance_mm:
            rejected.append(dict(seed=accepted, reason='initial_spacing_below_registered_maximum'))
            scene.close()
            continue
        break
    else:
        raise RuntimeError('No valid paired scene in 100 registered attempts')
    permutation = np.roll(np.arange(pool_size), -(seed % pool_size))
    starts = starts[permutation]
    particles = scene.positions_mm[pool_size:].copy()
    shared = dict(accepted_seed=accepted, anatomy=base.anatomy,
                  geometry_mm=scene.transport.points.tolist(),
                  healthy_radius_mm=scene.flow_model.healthy_radius_mm.tolist(),
                  clot_stations=scene.clot_stations.tolist(), initial_mass=scene.initial_mass.tolist(),
                  particles_mm=particles.tolist(), start_pool_mm=starts.tolist(),
                  flow_multiplier=float(scene.episode_flow_multiplier),
                  driving_pressure=float(scene.flow_model.driving_pressure))
    initial_hash = digest(shared)
    initial_rate = base.lysis_mass_per_s
    actual_rate = initial_rate/clusters if budget == 'fixed_total' else initial_rate
    cfg = replace(base, num_robots=clusters, lysis_mass_per_s=actual_rate)
    env = MCAPhysicalEnv(cfg)
    env.reset(seed=accepted, options=dict(robot_positions_mm=starts[:clusters], particle_positions_mm=particles))
    assert np.array_equal(env.clot_stations, scene.clot_stations)
    assert np.array_equal(env.transport.points, scene.transport.points)
    assert np.array_equal(env.positions_mm[clusters:], particles)
    assert env.flow_model.driving_pressure == scene.flow_model.driving_pressure
    scene.close()
    manifest = dict(requested_seed=seed, accepted_seed=accepted, rejected_attempts=rejected,
                    scenario_hash=initial_hash, pool_size=pool_size, spawn_clearance_mm=spawn_clearance_mm,
                    shared_scene=shared, selected_start_indices=permutation[:clusters].tolist(),
                    resource_budget=budget, per_cluster_lysis_mass_per_s=actual_rate,
                    aggregate_lysis_capacity_mass_per_s=clusters*actual_rate,
                    resource_matching_scope='catalytic_rate_proxy_only; body_size_and_total_magnetic_power_not_matched',
                    actual_config=asdict(cfg), revision=REVISION)
    return env, manifest


class SpacingTracker:
    """Ground truth for reporting, never visible to a controller.

    Integrates violation duration along the piecewise-linear accepted substep
    paths. Exit timing is respected. This is resolved-substep interpolation,
    not a claim about unobserved continuous trajectories or magnetic fields.
    """
    def __init__(self, positions, active, d_min_mm):
        self.n = len(positions)
        self.threshold = float(d_min_mm)
        self.minimum = np.inf
        self.pair_violation_s = 0.
        self.active_cluster_s = 0.
        self.control_violation_steps = 0
        self._step_violation = False
        self.initial_violation = False
        for a in range(self.n):
            for b in range(a+1, self.n):
                if active[a] and active[b]:
                    d = float(np.linalg.norm(positions[b]-positions[a]))
                    self.minimum = min(self.minimum, d)
                    self.initial_violation |= d < self.threshold

    def begin_step(self):
        self._step_violation = False

    def update(self, before, after, active, body_dt, dt):
        self.active_cluster_s += float(np.sum(np.asarray(body_dt)[:self.n]*np.asarray(active)[:self.n]))
        for a in range(self.n):
            for b in range(a+1, self.n):
                if not (active[a] and active[b]):
                    continue
                duration = min(float(body_dt[a]), float(body_dt[b]))
                if duration <= 0:
                    continue
                rel = before[b]-before[a]
                va = (after[a]-before[a])/max(float(body_dt[a]), 1e-30)
                vb = (after[b]-before[b])/max(float(body_dt[b]), 1e-30)
                v = vb-va
                aa, bb = float(v@v), float(2*rel@v)
                closest = float(np.clip(-bb/(2*aa), 0., duration)) if aa > 1e-24 else 0.
                min_d = float(np.linalg.norm(rel+closest*v))
                self.minimum = min(self.minimum, min_d, float(np.linalg.norm(rel+duration*v)))
                cc = float(rel@rel-self.threshold**2)
                violation_time = 0.
                if aa <= 1e-24:
                    violation_time = duration if cc < 0 else 0.
                else:
                    disc = bb*bb-4*aa*cc
                    if disc > 0:
                        left, right = (-bb-np.sqrt(disc))/(2*aa), (-bb+np.sqrt(disc))/(2*aa)
                        violation_time = max(0., min(duration, right)-max(0., left))
                self.pair_violation_s += violation_time
                self._step_violation |= violation_time > 0.

    def end_step(self):
        self.control_violation_steps += int(self._step_violation)

    def summary(self):
        return dict(minimum_spacing_mm=float(self.minimum) if np.isfinite(self.minimum) else None,
                    spacing_violation_pair_s=self.pair_violation_s,
                    spacing_violation_steps=self.control_violation_steps,
                    initial_spacing_violation=bool(self.initial_violation),
                    spacing_compliant=not self.initial_violation and self.control_violation_steps == 0,
                    spacing_measurement='accepted_substep_piecewise_linear')


def attach_spacing_monitor(env, tracker):
    """Tap accepted substeps while leaving the physics callback unchanged."""
    original = env.transport.advance
    def monitored(*args, after_substep=None, **kwargs):
        def callback(before, after, edges, active, body_dt, dt):
            tracker.update(before, after, active, body_dt, dt)
            return after_substep(before, after, edges, active, body_dt, dt) if after_substep is not None else None
        return original(*args, after_substep=callback, **kwargs)
    env.transport.advance = monitored
