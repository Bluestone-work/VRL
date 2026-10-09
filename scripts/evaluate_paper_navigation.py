"""Deployable CBF-QP and MPPI comparison under image sensing."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np


def build_configurations(methods, values=None):
    """Build independent typed configurations for each requested method."""
    from marl.paper_navigation import CBFConfig, MPPIConfig

    values = {} if values is None else values
    configurations = {}
    for method in methods:
        method_values = values.get(method, values) if isinstance(values, dict) else {}
        config_type = CBFConfig if method == 'cbf_qp' else MPPIConfig
        configurations[method] = asdict(config_type(**method_values))
    return configurations


def _observation_hash(estimate):
    """Hash only the deployable first-frame estimate for audit reproducibility."""
    obstacles = []
    for detections in estimate.obstacles:
        obstacles.append([
            [np.asarray(relative, float).round(6).tolist(),
             np.asarray(relative_velocity, float).round(6).tolist(), float(radius)]
            for relative, relative_velocity, radius in detections
        ])
    observation = dict(
        pos=np.asarray(estimate.pos, float).round(6).tolist(),
        vel=np.asarray(estimate.vel, float).round(6).tolist(),
        edge=np.asarray(estimate.edge, int).tolist(),
        station=np.asarray(estimate.station, int).tolist(),
        active=np.asarray(estimate.active, bool).tolist(),
        obstacles=obstacles,
        clot_alive=np.asarray(estimate.clot_alive, bool).tolist(),
    )
    return hashlib.sha256(json.dumps(observation, sort_keys=True).encode()).hexdigest()


def evaluate(job):
    anatomy, seed, method, horizon, configuration, audit_directory = job
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    from marl.hierarchical_navigation import EventReplanner, NavigationConfig, relaxed_success
    from marl.lookahead_teacher import option_local
    from marl.paper_navigation import CBFConfig, CBFQPFilter, MPPIConfig, MPPIController
    from scripts.benchmark_obstacles import Episode
    result = dict(anatomy=anatomy, seed=seed, method=method, horizon_s=horizon,
                  config=configuration, error=None)
    episode = None
    audit = None
    try:
        episode = Episode(1, anatomy, seed, horizon=horizon, sensing='image', topo_pursuit=True)
        if method.startswith('cbf'):
            navigator = CBFQPFilter(episode.sensor, 1, episode.ctl.body,
                                    CBFConfig(**dict(configuration,
                                                    speed_mm_s=episode.env.config.robot_speed_mm_s)))
        else:
            navigator = MPPIController(episode.sensor, 1, episode.ctl.body,
                                       MPPIConfig(**dict(configuration,
                                                        speed_mm_s=episode.env.config.robot_speed_mm_s)))
        replanner = EventReplanner(1, NavigationConfig())
        counts = {}
        first_observation_hash = None
        if audit_directory:
            path = Path(audit_directory)/f'{method}_{anatomy}_{seed}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            audit = path.open('w')
        while True:
            estimate, targets, rule, hold = episode.observe()
            if first_observation_hash is None:
                first_observation_hash = _observation_hash(estimate)
            time_s = episode.env.elapsed_s
            previous = episode.ctl.to_world(episode.prev_local, estimate)
            changed = replanner.update(time_s, estimate, targets, previous, hold, episode.ctl)
            if changed:
                rule = episode.ctl.act(targets, estimate)
                navigator.previous[:] = 0.
            nominal = option_local(episode, estimate, rule, 8)
            local = navigator.act(time_s, estimate, episode.ctl, nominal, hold)
            done, outcome = episode.step(estimate, local, hold)
            for record in navigator.last_audit:
                record['time_s'] = float(time_s)
                audit.write(json.dumps(record, allow_nan=False)+'\n') if audit else None
            mode = method
            counts[mode] = counts.get(mode, 0)+1
            if done:
                break
        row = episode.row(method)
        row['relaxed_safe_success'] = relaxed_success(row)
        row['removal_auc_observed'] = row['removal_auc']
        row['removal_auc'] += row['removal']*max(horizon-row['elapsed_s'], 0.)/horizon
        row.update(initial_observation_hash=first_observation_hash, sensing_model='image', primitive_counts=counts,
                   replan_count=len(replanner.events), replan_events=replanner.events)
        result['row'] = row
    except Exception as error:
        result.update(error=repr(error), row=None)
    finally:
        if audit:
            audit.close()
        if episode is not None:
            episode.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', choices=('cbf_qp', 'mppi'), required=True)
    parser.add_argument('--anatomies', nargs='+')
    parser.add_argument('--seeds', type=int, default=6)
    parser.add_argument('--first-seed', type=int, default=2600000000)
    parser.add_argument('--horizon', type=float, default=300.)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--result-timeout-s', type=float, default=180.)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--audit-dir', type=Path)
    args = parser.parse_args()
    from marl.evaluation_watchdog import evaluation_results
    from scripts.run_obstacle_gate import ANATOMIES
    anatomies = args.anatomies or ANATOMIES
    values = json.loads(args.config.read_text()) if args.config else {}
    configurations = build_configurations(args.methods, values)
    sources = ['marl/paper_navigation.py', 'marl/hierarchical_navigation.py', 'marl/vessel_sdf.py',
               'marl/deployable_sensing.py', 'marl/lookahead_teacher.py', 'scripts/benchmark_obstacles.py',
               'scripts/evaluate_paper_navigation.py', 'marl/evaluation_watchdog.py']
    manifest = dict(source_hashes={source: hashlib.sha256(Path(source).read_bytes()).hexdigest() for source in sources},
                    configurations=configurations, horizon_s=args.horizon, sensing='image',
                    privileged_inputs=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(str(args.out)+'.manifest.json')
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error('protocol changed; use a new output path')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    existing = [json.loads(line) for line in args.out.read_text().splitlines()] if args.out.exists() else []
    completed = {(record['anatomy'], record['seed'], record['method']) for record in existing if not record.get('error')}
    jobs = [(anatomy, seed, method, args.horizon, configurations[method],
             str(args.audit_dir) if args.audit_dir else None)
            for anatomy in anatomies for seed in range(args.first_seed, args.first_seed+args.seeds)
            for method in args.methods if (anatomy, seed, method) not in completed]
    with mp.get_context('spawn').Pool(args.workers, maxtasksperchild=4) as pool, args.out.open('a') as output:
        for result in evaluation_results(pool, evaluate, jobs, args.result_timeout_s,
                                         str(args.out)+'.runtime_timeout.json'):
            output.write(json.dumps(result, allow_nan=False)+'\n'); output.flush(); print(json.dumps({
                'anatomy': result['anatomy'], 'seed': result['seed'], 'method': result['method'],
                'error': result['error']}), flush=True)


if __name__ == '__main__':
    main()
