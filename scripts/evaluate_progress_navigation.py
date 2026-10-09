"""EXP0064: isolated paired development tests of measured wall-risk recovery."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np


def _observation_hash(estimate):
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
    from marl.hierarchical_navigation import EventReplanner, relaxed_success
    from marl.lookahead_teacher import option_local
    from marl.progress_navigation import AllObstacleNavigator, ProgressConfig, ProgressNavigator, RouteProgressMonitor
    from marl.risk_navigation import (ProgressBoundRiskConfig, ProgressBoundRiskNavigator,
                                      RiskConfig, RiskHysteresisNavigator)
    from scripts.benchmark_obstacles import Episode
    result = dict(anatomy=anatomy, seed=seed, method=method, horizon_s=horizon, config=configuration, error=None)
    episode = None
    audit = None
    try:
        episode = Episode(1, anatomy, seed, horizon=horizon, sensing='image', topo_pursuit=True)
        cfg = ProgressBoundRiskConfig(**configuration) if 'progress_stall_s' in configuration else RiskConfig(**configuration)
        if method in ('all_semantic', 'semantic_no_gate', 'semantic_reference_guard'):
            navigator = AllObstacleNavigator(episode.sensor, 1, episode.ctl.body, cfg)
        elif method == 'risk_hysteresis':
            navigator = RiskHysteresisNavigator(episode.sensor, 1, episode.ctl.body, cfg)
        elif method == 'progress_bound_risk':
            navigator = ProgressBoundRiskNavigator(episode.sensor, 1, episode.ctl.body, cfg)
        else:
            navigator = ProgressNavigator(episode.sensor, 1, episode.ctl.body, cfg)
        physical = EventReplanner(1, cfg)
        progress = RouteProgressMonitor(1, cfg)
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
            changed = physical.update(time_s, estimate, targets, previous, hold, episode.ctl)
            if changed:
                rule = episode.ctl.act(targets, estimate)
                navigator.trigger(time_s, changed)
            if method == 'all_semantic':
                local = navigator.act_all(time_s, estimate, episode.ctl, option_local(episode, estimate, rule, 8), hold)
            elif method == 'semantic_no_gate':
                local = navigator.act_all_no_gate(time_s, estimate, episode.ctl,
                                                  option_local(episode, estimate, rule, 8), hold)
            elif method == 'semantic_reference_guard':
                local = navigator.act_all_reference_guard(time_s, estimate, episode.ctl,
                                                          option_local(episode, estimate, rule, 8), hold)
            elif method == 'risk_hysteresis':
                pass
            elif method == 'progress_bound_risk':
                pass
            elif method != 'legacy_replan':
                loops = progress.update(time_s, estimate, targets, hold, episode.ctl, navigator.active(time_s))
                if loops:
                    if method == 'route_replan':
                        for robot in loops:
                            episode.ctl._plan(robot, estimate.pos[robot], int(targets[robot]))
                        rule = episode.ctl.act(targets, estimate)
                        navigator.trigger(time_s, loops)
                    else:
                        navigator.start(time_s, loops, estimate, targets, episode.ctl)
            reference = option_local(episode, estimate, rule, 8)
            if method not in ('all_semantic', 'semantic_no_gate', 'semantic_reference_guard', 'risk_hysteresis', 'progress_bound_risk'):
                local = navigator.act_with_targets(time_s, estimate, targets, episode.ctl, reference, hold)
            elif method == 'all_semantic':
                local = navigator.act_all(time_s, estimate, episode.ctl, reference, hold)
            elif method == 'semantic_no_gate':
                local = navigator.act_all_no_gate(time_s, estimate, episode.ctl, reference, hold)
            elif method == 'semantic_reference_guard':
                local = navigator.act_all_reference_guard(time_s, estimate, episode.ctl, reference, hold)
            elif method == 'risk_hysteresis':
                local = navigator.act_risk(time_s, estimate, episode.ctl, reference, hold)
            else:
                local = navigator.act_progress_bound(time_s, estimate, episode.ctl, reference, hold, targets)
            done, outcome = episode.step(estimate, local, hold)
            for mode in navigator.last_modes:
                counts[mode] = counts.get(mode, 0)+1
            if audit:
                record = dict(time_s=float(time_s), mode=navigator.last_modes[0], target=int(targets[0]),
                              route_station=int(episode.ctl.station[0]), route_progress=int(episode.ctl.prog[0]),
                              position_est_mm=estimate.pos[0].tolist(), reference_local=reference[0].tolist(),
                              proposed_local=local[0].tolist(), executed_local=episode.prev_local[0].tolist(),
                              obstacle_events=int(outcome['obs_events'][0]), wall_contact_s=float(outcome['wall'][0]),
                              removed=float(outcome['removed']), plans=navigator.last_plan,
                              risk=getattr(navigator, 'last_risk', []), risk_events=getattr(navigator, 'risk_events', []))
                audit.write(json.dumps(record, allow_nan=False)+'\n')
            if done:
                break
        row = episode.row(method)
        row['relaxed_safe_success'] = relaxed_success(row)
        row['removal_auc_observed'] = row['removal_auc']
        row['removal_auc'] += row['removal']*max(horizon-row['elapsed_s'], 0.)/horizon
        row.update(initial_observation_hash=first_observation_hash, sensing_model='image', primitive_counts=counts,
                   replan_count=len(physical.events)+len(progress.events), global_replan_count=len(physical.events),
                   route_event_count=len(progress.events), replan_events=physical.events+progress.events,
                   risk_event_count=len(getattr(navigator, 'risk_events', [])),
                   risk_guard_steps=counts.get('risk_guard', 0)+counts.get('progress_risk_guard', 0))
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
    parser.add_argument('--methods', nargs='+', choices=['legacy_replan', 'route_replan', 'progress_bypass', 'all_semantic', 'semantic_no_gate', 'semantic_reference_guard', 'risk_hysteresis', 'progress_bound_risk'],
                        default=['legacy_replan', 'route_replan', 'progress_bypass', 'all_semantic'])
    parser.add_argument('--anatomies', nargs='+')
    parser.add_argument('--seeds', type=int, default=6)
    parser.add_argument('--first-seed', type=int, default=2600000000)
    parser.add_argument('--horizon', type=float, default=300.)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--result-timeout-s', type=float, default=120.)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--audit-dir', type=Path)
    arguments = parser.parse_args()
    from marl.evaluation_watchdog import evaluation_results
    from marl.risk_navigation import ProgressBoundRiskConfig, RiskConfig
    from scripts.iterate_hierarchical_navigation import summarize
    from scripts.run_obstacle_gate import ANATOMIES
    anatomies = arguments.anatomies or ANATOMIES
    seeds = list(range(arguments.first_seed, arguments.first_seed+arguments.seeds))
    if any(anatomy not in ANATOMIES for anatomy in anatomies) or not seeds or any(not 2600000000 <= seed < 2700000000 for seed in seeds):
        parser.error('Only registered anatomies and non-sealed development seeds are allowed')
    config_values = json.loads(arguments.config.read_text()) if arguments.config else {}
    cfg = ProgressBoundRiskConfig(**config_values) if 'progress_stall_s' in config_values else RiskConfig(**config_values)
    configuration = asdict(cfg)
    sources = ['marl/progress_navigation.py', 'marl/risk_navigation.py', 'marl/hybrid_navigation.py', 'marl/hierarchical_navigation.py',
               'marl/evaluation_watchdog.py', 'marl/deployable_sensing.py', 'marl/lookahead_teacher.py',
               'marl/obstacle_control.py', 'marl/vessel_sdf.py', 'scripts/benchmark_obstacles.py',
               'scripts/evaluate_progress_navigation.py']
    manifest = dict(source_hashes={source: hashlib.sha256(Path(source).read_bytes()).hexdigest() for source in sources},
                    configuration=configuration, horizon_s=arguments.horizon)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    path = Path(str(arguments.out)+'.manifest.json')
    if path.exists() and json.loads(path.read_text()) != manifest:
        parser.error('Sources/protocol changed; use a new output path')
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    existing = [json.loads(line) for line in arguments.out.read_text().splitlines()] if arguments.out.exists() else []
    if any(record.get('config') != configuration or record.get('horizon_s') != arguments.horizon for record in existing):
        parser.error('Existing rows use a different protocol')
    completed = {(record['anatomy'], record['seed'], record['method']) for record in existing if not record.get('error')}
    jobs = [(anatomy, seed, method, arguments.horizon, configuration,
             str(arguments.audit_dir) if arguments.audit_dir else None)
            for anatomy in anatomies for seed in seeds for method in arguments.methods
            if (anatomy, seed, method) not in completed]
    with mp.get_context('spawn').Pool(arguments.workers, maxtasksperchild=4) as pool, arguments.out.open('a') as output:
        for result in evaluation_results(pool, evaluate, jobs, arguments.result_timeout_s, str(arguments.out)+'.runtime_timeout.json'):
            output.write(json.dumps(result, allow_nan=False)+'\n')
            output.flush()
            print(json.dumps(dict(anatomy=result['anatomy'], seed=result['seed'], method=result['method'], error=result['error'])), flush=True)
    summary = summarize(arguments.out)
    Path(str(arguments.out)+'.summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
