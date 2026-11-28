"""Base tasks that do not depend on the number of clusters.

A base task fixes the vessel geometry, the lesion (target) stations, the dynamic
obstacles, the flow regime and the patient-level dynamics variation.  Every
cluster configuration (any N, any release points) is executed on the same base
task, identified by ``base_task_hash``.  The configuration itself is hashed
separately (``configuration_hash`` in marl.v3.configuration).

Construction
------------
1. probe A (no targets chosen yet, no obstacles) gives the tree for the seed;
2. targets are sampled from the tree with an RNG keyed on (seed, m, pattern);
3. probe B with those targets fixed samples the dynamic obstacles;
4. ``env_factory`` rebuilds the scene for any N with clusters placed at their
   release stations, parked inside the catheter, and asserts that geometry,
   targets and obstacles are bit-identical to probe B.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

FLOW_MODES = {'weak': (0.025, False), 'strong': (0.10, False), 'pulsatile': (0.05, True)}
PATTERNS = ('dispersed', 'clustered')
PATTERN_ID = {p: k for k, p in enumerate(PATTERNS)}
MIN_TARGET_SEPARATION_MM = 3.0


@dataclass(frozen=True)
class BaseTask:
    anatomy: str
    seed: int
    n_targets: int
    pattern: str
    flow_mode: str
    obstacles: int = 8
    variation: float = 0.75
    split: str = 'dev'

    def key(self):
        return asdict(self)


def _base_config(task, horizon):
    from environments.mca_physical_env import DynamicsConfig
    from marl.teacher import TEACHER_CONFIG
    base = replace(DynamicsConfig.from_json(TEACHER_CONFIG), anatomy=task.anatomy, episode_duration_s=horizon,
                   junction_model='union', particle_count=int(task.obstacles))
    return replace(base, action_prior='none', action_shield_horizon_s=0., command_speed='bounded',
                   target_observation='routed_assigned_own', progress_reward_scale=0.)


def root_release_station(tree, body_mm=0.08, min_offset_mm=0.5):
    """First trunk station at least min_offset_mm downstream of the inlet (a body on the inlet node exits)."""
    b = tree.branches[0]
    scale = float(tree.physical_mm_per_unit)
    for s in range(b.start, b.stop + 1):
        if (tree.arclength[s] - tree.arclength[b.start]) * scale >= max(min_offset_mm, 3 * body_mm):
            return int(s)
    return int(b.start + 1)


def target_candidates(env):
    """Branch-interior stations a body fits in, >1 mm from both branch ends (same rule as random_branches)."""
    tree = env.tree
    scale = float(tree.physical_mm_per_unit)
    out = {}
    r = np.asarray(env.flow_model.healthy_radius_mm)
    for bid, br in enumerate(tree.branches):
        ids = np.arange(br.start + 1, br.stop)
        if not len(ids):
            continue
        head = (tree.arclength[ids] - tree.arclength[br.start]) * scale
        tail = (tree.arclength[br.stop] - tree.arclength[ids]) * scale
        ids = ids[(head > 1.) & (tail > 1.) & (r[ids] * env.config.initial_radius_fraction > env.config.robot_radius_mm + 1e-7)]
        if bid == 0:
            # keep targets off the first 5 mm of the trunk where every cluster is released
            ids = ids[(tree.arclength[ids] - tree.arclength[br.start]) * scale > 5.]
        if len(ids):
            out[bid] = ids
    return out


def _subtree(tree, bid):
    kids = {b.branch_id: [] for b in tree.branches}
    for b in tree.branches:
        if b.parent is not None and b.parent >= 0:
            kids[b.parent].append(b.branch_id)
    out, stack = [], [bid]
    while stack:
        k = stack.pop(); out.append(k); stack.extend(kids.get(k, []))
    return out


def sample_targets(env, m, pattern, seed):
    rng = np.random.default_rng([int(seed), 7301, int(m), PATTERN_ID[pattern]])
    cand = target_candidates(env)
    pts = env.transport.points
    def pick(pool_by_branch, order):
        chosen = []
        for _ in range(4000):
            if len(chosen) == m:
                break
            bid = order[len(chosen) % len(order)] if pattern == 'dispersed' else rng.choice(order)
            pool = pool_by_branch[int(bid)]
            s = int(rng.choice(pool))
            if all(np.linalg.norm(pts[s] - pts[c]) >= MIN_TARGET_SEPARATION_MM for c in chosen):
                chosen.append(s)
            elif pattern == 'dispersed':
                order = list(rng.permutation(order))
        return chosen if len(chosen) == m else None
    branches = list(cand)
    if pattern == 'dispersed':
        for _ in range(50):
            got = pick(cand, list(rng.permutation(branches)))
            if got:
                return np.asarray(got, np.int32), 'dispersed'
    else:
        tree = env.tree
        non_root = [b for b in branches if b != 0] or branches
        for bid in rng.permutation(non_root):
            sub = [b for b in _subtree(tree, int(bid)) if b in cand]
            if sum(len(cand[b]) for b in sub) < 3 * m:
                continue
            got = pick({b: cand[b] for b in sub}, sub)
            if got:
                return np.asarray(got, np.int32), 'clustered'
        got = pick(cand, list(rng.permutation(branches)))
        if got:
            return np.asarray(got, np.int32), 'clustered_fallback_dispersed'
    raise ValueError(f'cannot place {m} separated targets with pattern {pattern}')


class SceneBuilder:
    """Builds the base task once; returns env factories for any N and release stations."""

    def __init__(self, task: BaseTask, horizon=300.):
        from environments.mca_physical_env import MCAPhysicalEnv
        from scripts.multicluster_protocol import digest
        self.task, self.horizon = task, float(horizon)
        cfg = _base_config(task, horizon)
        self._cfg = cfg
        # probe A: tree and flow for this seed (both drawn before targets and obstacles)
        tmp = MCAPhysicalEnv(replace(cfg, num_robots=1, particle_count=0, clot_initialization='historical_sites',
                                     robot_initialization='distributed_branches'))
        tmp.reset(seed=task.seed)
        self.root_station = root_release_station(tmp.tree)
        root_point = tmp.transport.points[[self.root_station]].astype(np.float64)
        targets, realised = sample_targets(tmp, task.n_targets, task.pattern, task.seed)
        self.pattern_realised = realised
        tmp.close()
        self.targets = targets
        probe = MCAPhysicalEnv(replace(cfg, num_robots=1), clot_stations=targets)
        probe.reset(seed=task.seed, options=dict(robot_positions_mm=root_point))
        self.particles = probe.positions_mm[1:].copy()
        self.geometry = probe.transport.points.copy()
        self.healthy_radius = np.asarray(probe.flow_model.healthy_radius_mm).copy()
        shared = dict(version='v3_base_task_1', task=task.key(), horizon_s=self.horizon,
                      geometry_mm=self.geometry.round(9).tolist(), healthy_radius_mm=self.healthy_radius.round(9).tolist(),
                      target_stations=targets.tolist(), particles_mm=self.particles.round(9).tolist(),
                      flow_multiplier=float(probe.episode_flow_multiplier),
                      driving_pressure=float(probe.flow_model.driving_pressure),
                      flow_inlet_mm_s=FLOW_MODES[task.flow_mode][0], pulsatile=FLOW_MODES[task.flow_mode][1])
        self.base_task_hash = digest(shared)
        self.probe = probe                       # preoperative model used by the planner (geometry + targets)

    def env_factory(self, release_stations):
        """Factory for LysisEpisode(env_factory=...): clusters start parked at their release stations."""
        from environments.mca_physical_env import MCAPhysicalEnv
        task, targets, particles = self.task, self.targets, self.particles
        geometry, base_hash = self.geometry, self.base_task_hash
        release_stations = [int(s) for s in release_stations]

        def factory(base, n, seed):
            if n != len(release_stations):
                raise ValueError('cluster count does not match the release plan')
            cfg = replace(base, num_robots=n, action_prior='none', action_shield_horizon_s=0., command_speed='bounded',
                          target_observation='routed_assigned_own', progress_reward_scale=0.)
            env = MCAPhysicalEnv(cfg, clot_stations=targets)
            start = geometry[release_stations].astype(np.float64)
            env.reset(seed=task.seed, options=dict(robot_positions_mm=start, particle_positions_mm=particles))
            if not (np.array_equal(env.transport.points, geometry) and np.array_equal(env.clot_stations, targets)
                    and np.array_equal(env.positions_mm[n:], particles)):
                raise AssertionError('base task is not reproduced for this configuration')
            env.park(np.arange(n))
            manifest = dict(base_task_hash=base_hash, scenario_hash=base_hash, accepted_seed=int(task.seed),
                            release_stations=release_stations, deployment='catheter_release_v3',
                            selected_start_indices=list(range(n)))
            return env, manifest
        return factory

    def close(self):
        self.probe.close()


def attach_variation(ep, strength, pulsatile, n_max):
    """Patient-level dynamics variation drawn for n_max clusters and sliced to ep.n.

    marl.physio_variation draws per-cluster parameters with size ep.n, which makes
    every later draw depend on N.  Drawing with n_max and slicing keeps the patient
    (flow mean, pulsatility, heart rate, lag, adhesion ...) and cluster k's gain and
    bias identical across configurations of the same base task.
    """
    from marl.physio_variation import Variation
    n = ep.n
    ep.n = int(n_max)
    try:
        var = Variation(ep, float(strength), np.random.default_rng([ep.seed, 5150, int(round(1000 * strength))]),
                        pulsatile=bool(pulsatile))
    finally:
        ep.n = n
    var.gain0 = var.gain0[:n]; var.log_drift = var.log_drift[:n]; var.rot = var.rot[:n]
    var.u_eff = var.u_eff[:n]; var.touch = var.touch[:n]
    ep.var = var
    return var
