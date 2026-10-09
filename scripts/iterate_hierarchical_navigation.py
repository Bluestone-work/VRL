"""Paired development evaluation of local navigation, without modifying active training runs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np


def evaluate(job):
    anatomy, seed, method, horizon, configuration, audit_dir = job
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    from marl.hierarchical_navigation import EventReplanner, LocalNavigator, NavigationConfig, SemanticAvoidance, WallRecoveryExecutor, relaxed_success
    from marl.lookahead_teacher import option_local, primitive_local
    from scripts.benchmark_obstacles import Episode

    result = dict(anatomy=anatomy, seed=seed, method=method, horizon_s=horizon, config=configuration, error=None)
    episode = None
    audit = None
    try:
        episode = Episode(1, anatomy, seed, horizon=horizon, sensing='image', topo_pursuit=True,
                          avoid=method != 'pursuit')
        initial = dict(positions=episode.env.positions_mm[:1].tolist(), masses=episode.env.masses.tolist(),
                       edges=episode.env.edges[:1].tolist(), obstacles=episode.field.positions()[0].tolist())
        initial_hash = hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()
        cfg = NavigationConfig(**configuration)
        navigator = LocalNavigator(episode.sensor, 1, episode.env.config.robot_speed_mm_s, episode.ctl.body, cfg)
        executor = WallRecoveryExecutor(episode.sensor, 1, episode.ctl.body, cfg)
        semantic = SemanticAvoidance(1, episode.ctl.body, cfg)
        replanner = EventReplanner(1, cfg)
        if audit_dir:
            path = Path(audit_dir)/f'{method}_{anatomy}_{seed}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            audit = path.open('w')
        primitive_counts = {}
        while True:
            estimate, targets, rule, hold = episode.observe()
            time_s = episode.env.elapsed_s
            if method.endswith('_replan'):
                previous_world = episode.ctl.to_world(episode.prev_local, estimate)
                changed = replanner.update(time_s, estimate, targets, previous_world, hold, episode.ctl)
                if changed:
                    rule = episode.ctl.act(targets, estimate)
                    executor.trigger(time_s, changed)
            if method.startswith('local'):
                reference = option_local(episode, estimate, rule, 8)
                local = navigator.act(time_s, estimate, episode.ctl, reference, hold)
            elif method.startswith('switch'):
                local = option_local(episode, estimate, rule, 8)
            elif method.startswith('guard'):
                reference = option_local(episode, estimate, rule, 8)
                local = executor.act(time_s, estimate, episode.ctl, reference, hold)
            elif method.startswith('semantic'):
                reference = semantic.act(time_s, estimate, episode.ctl)
                local = executor.act(time_s, estimate, episode.ctl, reference, hold)
                for record in executor.last_audit:
                    record['decision_primitive'] = semantic.modes[record['robot']]
            elif method == 'away':
                local = primitive_local(episode, estimate, rule, 3)
            else:
                local = rule
            done, outcome = episode.step(estimate, local, hold)
            audit_records = navigator.last_audit if method.startswith('local') else executor.last_audit
            for record in audit_records:
                primitive_counts[record['primitive']] = primitive_counts.get(record['primitive'], 0)+1
                if audit:
                    record.update(time_s=float(time_s), executed_local=episode.prev_local[record['robot']].tolist(),
                                  wall_contact_s=float(outcome['wall'][record['robot']]),
                                  obstacle_events=int(outcome['obs_events'][record['robot']]))
                    audit.write(json.dumps(record, allow_nan=False)+'\n')
            if done:
                break
        row = episode.row(method)
        row['relaxed_safe_success'] = relaxed_success(row)
        row['removal_auc_observed'] = row['removal_auc']
        row['removal_auc'] += row['removal']*max(horizon-row['elapsed_s'], 0.)/horizon
        row.update(initial_state_hash=initial_hash, sensing_model='image', primitive_counts=primitive_counts,
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


def summarize(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    summary = {}
    for method in sorted({record['method'] for record in records}):
        group = [record for record in records if record['method'] == method]
        rows = [record['row'] for record in group if record.get('row') and not record.get('error')]
        summary[method] = dict(episodes=len(rows), errors=len(group)-len(rows))
        for metric in ('task_success', 'cluster_safe_success', 'relaxed_safe_success', 'wall_contact_s',
                       'max_continuous_wall_contact_s', 'obstacle_events_static', 'obstacle_events_dynamic',
                       'removal_auc', 'replan_count'):
            summary[method][metric] = float(np.mean([row[metric] for row in rows])) if rows else None
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=['switch', 'guard', 'guard_replan'],
                        choices=['pursuit', 'apf', 'switch', 'switch_replan', 'away', 'local', 'local_replan',
                                 'guard', 'guard_replan', 'semantic', 'semantic_replan'])
    parser.add_argument('--anatomies', nargs='+')
    parser.add_argument('--first-seed', type=int, default=2600000000)
    parser.add_argument('--seeds', type=int, default=6)
    parser.add_argument('--horizon', type=float, default=300.)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--result-timeout-s', type=float, default=120.)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--audit-dir', type=Path)
    arguments = parser.parse_args()
    from marl.hierarchical_navigation import NavigationConfig
    from marl.evaluation_watchdog import evaluation_results
    from scripts.run_obstacle_gate import ANATOMIES
    anatomies = arguments.anatomies or ANATOMIES
    if any(anatomy not in ANATOMIES for anatomy in anatomies):
        parser.error('Unknown development anatomy')
    seeds = list(range(arguments.first_seed, arguments.first_seed+arguments.seeds))
    if not seeds or any(not 2600000000 <= seed < 2700000000 for seed in seeds):
        parser.error('Only the explicitly non-sealed 2600000000:2700000000 development range is supported')
    configuration = asdict(NavigationConfig(**(json.loads(arguments.config.read_text()) if arguments.config else {})))
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    sources = ['marl/hierarchical_navigation.py', 'marl/deployable_sensing.py', 'marl/lookahead_teacher.py',
               'marl/obstacle_control.py', 'marl/vessel_sdf.py', 'scripts/benchmark_obstacles.py',
               'scripts/iterate_hierarchical_navigation.py', 'marl/evaluation_watchdog.py']
    hashes = {source: hashlib.sha256(Path(source).read_bytes()).hexdigest() for source in sources}
    manifest = dict(source_hashes=hashes, configuration=configuration, horizon_s=arguments.horizon)
    manifest_path = Path(str(arguments.out)+'.manifest.json')
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error('Output protocol or source changed; use a new output path')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    existing = [json.loads(line) for line in arguments.out.read_text().splitlines()] if arguments.out.exists() else []
    if any(record.get('config') != configuration for record in existing):
        parser.error('Output contains a different configuration; use a new output path')
    if any(record.get('horizon_s') != arguments.horizon for record in existing):
        parser.error('Output contains a different horizon; use a new output path')
    completed = {(record['anatomy'], record['seed'], record['method']) for record in existing if not record.get('error')}
    jobs = [(anatomy, seed, method, arguments.horizon, configuration,
             str(arguments.audit_dir) if arguments.audit_dir else None)
            for anatomy in anatomies for seed in seeds for method in arguments.methods
            if (anatomy, seed, method) not in completed]
    with mp.get_context('spawn').Pool(arguments.workers, maxtasksperchild=4) as pool, arguments.out.open('a') as output:
        for result in evaluation_results(pool, evaluate, jobs, arguments.result_timeout_s,
                                         str(arguments.out)+'.runtime_timeout.json'):
            output.write(json.dumps(result, allow_nan=False)+'\n')
            output.flush()
            print(json.dumps(dict(anatomy=result['anatomy'], seed=result['seed'], method=result['method'],
                                  error=result['error'])), flush=True)
    summary = summarize(arguments.out)
    Path(str(arguments.out)+'.summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
