"""EXP0046 v2 paired, observation-only development evaluation. No sealed data."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np

from environments.mca_physical_env import DynamicsConfig
from marl.multicluster import MultiClusterConfig, MultiClusterController
from marl.multicluster_observation import ClusterSensorAdapter
from marl.teacher import teacher_config, teacher_label
from scripts.multicluster_protocol import paired_environment, SpacingTracker, attach_spacing_monitor, REVISION, digest
from scripts.safe_metrics import WallTracker, aggregate, episode_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT/'configs/experiments/EXP_0029_MCA_ALL_RANDOM_DYNAMICS.json'
METHODS = ('single_sequential', 'single_route', 'multi_parallel', 'multi_unshielded')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--clusters', type=int, default=2)
    p.add_argument('--d-min-mm', type=float, default=2.)
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--seed-base', type=int, default=1302000000)
    p.add_argument('--control-seed', type=int, default=42)
    p.add_argument('--anatomy', default='mca_m1_lvo')
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--noise', type=float, default=.025)
    p.add_argument('--position-noise-mm', type=float, default=.02)
    p.add_argument('--peer-sensing-mm', type=float, default=6.)
    p.add_argument('--budget', choices=('fixed_total', 'per_cluster'), default='fixed_total')
    p.add_argument('--duration-s', type=float, default=180.)
    p.add_argument('--max-steps', type=int)
    args = p.parse_args()
    if args.episodes < 1 or (args.max_steps is not None and args.max_steps < 1):
        p.error('episodes and max-steps must be positive')
    return args


def source_hashes():
    files = ['marl/multicluster.py', 'marl/multicluster_observation.py', 'marl/partial_obs.py',
             'marl/fair_reactive.py', 'marl/teacher.py', 'scripts/evaluate_multicluster.py',
             'scripts/multicluster_protocol.py', 'environments/mca_physical_env.py',
             'environments/mca_physical_dynamics.py', 'environments/mca_physiology.py']
    return {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}


def run_episode(cfg, protocol, seed, max_steps=None, *, budget='fixed_total'):
    started = time.monotonic()
    env, manifest = paired_environment(cfg, protocol.clusters, seed, budget=budget)
    sensor = ClusterSensorAdapter(env, protocol)
    # Paired observation randomization. Policy/controller seed is independent
    # of the scene RNG, not an RL training seed.
    sensor.reset(int(np.random.SeedSequence([seed, protocol.control_seed]).generate_state(1)[0]))
    policy = None if protocol.method == 'single_route' else MultiClusterController(
        protocol, robot_speed_mm_s=cfg.robot_speed_mm_s, control_dt_s=cfg.control_dt_s)
    if policy is not None:
        policy.reset(protocol.control_seed)
    order = np.argsort(env.tree.arclength[env.clot_stations]) if policy is None else None
    teacher_cfg = teacher_config(env.config) if policy is None else None
    walls = WallTracker(env.num_robots)
    spacing = SpacingTracker(env.positions_mm[:env.num_robots], env.active[:env.num_robots], protocol.min_spacing_mm)
    attach_spacing_monitor(env, spacing)
    trace, reward = [], 0.
    pair_contact, active_cluster_s, command_sq_s = 0., 0., 0.
    milestone = {str(k): None for k in (50, 90, 100)}
    clearance_auc_s = 0.
    initial_mass = float(env.initial_mass.sum())
    guard_hit = False
    try:
        while True:
            packet = sensor.observe()
            if policy is None:
                live = np.flatnonzero(env.masses > 0)
                if len(live):
                    target = next(int(j) for j in order if env.masses[j] > 0)
                    env._assignment = np.full(env.num_robots, target, np.int64)
                    env._assignment_alive = live.tobytes()
                local, stop, _ = teacher_label(env, teacher_cfg)
                local[stop] = 0.
                control = dict(controller_input='privileged_route_baseline', safety_certificate=False)
            else:
                local, control = policy.act(packet)
            before_removal = 1-float(env.masses.sum())/initial_mass
            active_before = env.active[:env.num_robots].copy()
            action = sensor.execute(local)
            assert env.config.action_prior == 'none', 'Controller must be applied exactly once'
            spacing.begin_step()
            _, r, terminated, truncated, info = env.step(action)
            spacing.end_step()
            dt = float(info['step_duration_s'])
            walls.update(info, active_before, dt)
            reward += float(r)
            pair_contact += float(info['robot_pair_contact_s'])
            active_cluster_s = spacing.active_cluster_s
            command_sq_s += float(np.square(action).sum())*dt
            removal = 1-float(info['remaining_mass'])/initial_mass
            clearance_auc_s += .5*(before_removal+removal)*dt
            for target in (50, 90, 100):
                if milestone[str(target)] is None and removal >= target/100-1e-12:
                    milestone[str(target)] = float(info['elapsed_s'])
            if env.steps == 1 or env.steps % 100 == 0 or terminated or truncated:
                trace.append(dict(step=env.steps, elapsed_s=env.elapsed_s, removal=removal,
                                  control=control, spacing=spacing.summary()))
            if terminated or truncated:
                break
            if max_steps is not None and env.steps >= max_steps:
                guard_hit = True
                info = dict(info, termination_reason='development_step_guard')
                break
        metrics = episode_metrics(info, walls, initial_mass)
        measured = spacing.summary()
        cluster_safe = bool(metrics['safe_collision_free'] and measured['spacing_compliant']
                            and pair_contact <= 1e-12 and not guard_hit)
        removed = initial_mass-float(info['remaining_mass'])
        return dict(revision=REVISION, seed=seed, control_seed=protocol.control_seed,
                    method=protocol.method, clusters=protocol.clusters,
                    min_spacing_threshold_mm=protocol.min_spacing_mm,
                    reset_info=manifest, actual_config=manifest['actual_config'],
                    final_state_hash=digest(dict(positions=env.positions_mm.tolist(), masses=env.masses.tolist(),
                                                 active=env.active.tolist(), elapsed_s=env.elapsed_s)),
                    controller_config=asdict(protocol), source_hashes=source_hashes(),
                    runtime=dict(python=platform.python_version(), numpy=np.__version__),
                    controller=control, trace=trace, reward=reward,
                    development_guard_hit=guard_hit, status='completed',
                    local_observation=policy is not None, sealed_test_used=False,
                    resource_budget=budget, **metrics, **measured,
                    cluster_safe_success=cluster_safe,
                    particle_contact_s=float(info['episode_particle_contact_s']),
                    particle_collision_events=int(info['episode_particle_collision_events']),
                    robot_pair_contact_s=pair_contact, lost_clusters=int(info['lost_robots']),
                    time_to_removal_s=milestone, removal_auc_s=clearance_auc_s,
                    removed_mass=removed, active_cluster_s=active_cluster_s,
                    mass_per_cluster_s=removed/max(active_cluster_s, 1e-12),
                    path_mm=float(np.sum(info['robot_path_mm'])),
                    command_squared_s=command_sq_s,
                    simulation_walltime_s=time.monotonic()-started)
    finally:
        env.close()


def main():
    from dataclasses import replace
    args = parse_args()
    cfg = replace(DynamicsConfig.from_json(args.config), num_robots=args.clusters,
                  anatomy=args.anatomy, episode_duration_s=args.duration_s)
    protocol = MultiClusterConfig(method=args.method, clusters=args.clusters,
                                  min_spacing_mm=args.d_min_mm, observation_noise=args.noise,
                                  control_seed=args.control_seed, position_noise_mm=args.position_noise_mm,
                                  peer_sensing_radius_mm=args.peer_sensing_mm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # Exclusive create: preserve all historical/raw experiments on a rerun.
    with args.out.open('x', encoding='utf-8') as stream:
        for i in range(args.episodes):
            row = run_episode(cfg, protocol, args.seed_base+i, args.max_steps, budget=args.budget)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
            print(json.dumps({k: row[k] for k in ('seed', 'method', 'clusters', 'safe_success',
                  'cluster_safe_success', 'removal', 'elapsed_s', 'spacing_violation_pair_s')}), flush=True)
    summary = dict(revision=REVISION, episodes=len(rows), aggregate=aggregate(rows),
                   cluster_safe_success=float(np.mean([r['cluster_safe_success'] for r in rows])),
                   budget=args.budget, duration_s=args.duration_s, requested_config=str(args.config),
                   sealed_test_used=False, development_diagnostic_only=True)
    args.out.with_name(args.out.stem+'_summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
