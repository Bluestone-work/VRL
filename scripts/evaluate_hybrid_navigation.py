"""Frozen development evaluation of measured-size-gated local navigation (EXP0062 v6)."""
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
    anatomy, seed, method, configuration = job[:4]
    audit_directory = job[4] if len(job) > 4 else None
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    from marl.hierarchical_navigation import EventReplanner, relaxed_success
    from marl.hybrid_navigation import HybridConfig, HybridNavigator
    from marl.lookahead_teacher import option_local
    from scripts.benchmark_obstacles import Episode

    result = dict(anatomy=anatomy, seed=seed, method=method, config=configuration, error=None)
    episode = None
    audit = None
    try:
        episode = Episode(1, anatomy, seed, horizon=300., sensing='image', topo_pursuit=True)
        initial = dict(positions=episode.env.positions_mm[:1].tolist(), masses=episode.env.masses.tolist(),
                       edges=episode.env.edges[:1].tolist(), obstacles=episode.field.positions()[0].tolist())
        initial_hash = hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()
        cfg = HybridConfig(**configuration)
        navigator = HybridNavigator(episode.sensor, 1, episode.ctl.body, cfg)
        replanner = EventReplanner(1, cfg)
        if audit_directory:
            audit_path = Path(audit_directory)/f'{method}_{anatomy}_{seed}.jsonl'
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit = audit_path.open('w')
        counts = {}
        while True:
            estimate, targets, rule, hold = episode.observe()
            time_s = episode.env.elapsed_s
            if method == 'hybrid_replan':
                previous = episode.ctl.to_world(episode.prev_local, estimate)
                changed = replanner.update(time_s, estimate, targets, previous, hold, episode.ctl)
                if changed:
                    rule = episode.ctl.act(targets, estimate)
                    navigator.trigger(time_s, changed)
            reference = option_local(episode, estimate, rule, 8)
            local = navigator.act(time_s, estimate, episode.ctl, reference, hold)
            done, outcome = episode.step(estimate, local, hold)
            if audit:
                record = dict(time_s=float(time_s), mode=navigator.last_modes[0], target=int(targets[0]),
                              route_station=int(episode.ctl.station[0]), route_progress=int(episode.ctl.prog[0]),
                              position_est_mm=estimate.pos[0].tolist(), velocity_est_mm_s=estimate.vel[0].tolist(),
                              nominal_local=episode.ctl.nominal[0].tolist(), reference_local=reference[0].tolist(),
                              proposed_local=local[0].tolist(), executed_local=episode.prev_local[0].tolist(),
                              executed_world=episode.ctl.to_world(episode.prev_local, estimate)[0].tolist(),
                              obstacle_events=int(outcome['obs_events'][0]), wall_contact_s=float(outcome['wall'][0]),
                              removed=float(outcome['removed']))
                audit.write(json.dumps(record, allow_nan=False)+'\n')
            for mode in navigator.last_modes:
                counts[mode] = counts.get(mode, 0)+1
            if done:
                break
        row = episode.row(method)
        row['relaxed_safe_success'] = relaxed_success(row)
        row['removal_auc_observed'] = row['removal_auc']
        row['removal_auc'] += row['removal']*max(300.-row['elapsed_s'], 0.)/300.
        row.update(initial_state_hash=initial_hash, sensing_model='image', primitive_counts=counts,
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
    parser.add_argument('--methods', nargs='+', choices=['hybrid', 'hybrid_replan'], default=['hybrid', 'hybrid_replan'])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--result-timeout-s', type=float, default=120.)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--audit-dir', type=Path)
    arguments = parser.parse_args()
    from marl.hybrid_navigation import HybridConfig
    from marl.evaluation_watchdog import evaluation_results
    from scripts.iterate_hierarchical_navigation import summarize
    from scripts.run_obstacle_gate import ANATOMIES
    configuration = asdict(HybridConfig(**(json.loads(arguments.config.read_text()) if arguments.config else {})))
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    sources = ['marl/hybrid_navigation.py', 'marl/hierarchical_navigation.py', 'marl/deployable_sensing.py',
               'marl/lookahead_teacher.py', 'marl/obstacle_control.py', 'marl/vessel_sdf.py',
               'scripts/benchmark_obstacles.py', 'scripts/evaluate_hybrid_navigation.py', 'marl/evaluation_watchdog.py']
    manifest = dict(source_hashes={source: hashlib.sha256(Path(source).read_bytes()).hexdigest() for source in sources},
                    configuration=configuration, horizon_s=300.)
    manifest_path = Path(str(arguments.out)+'.manifest.json')
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error('Protocol changed; use a new output path')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    records = [json.loads(line) for line in arguments.out.read_text().splitlines()] if arguments.out.exists() else []
    if any(record.get('config') != configuration for record in records):
        parser.error('Configuration changed; use a new output path')
    completed = {(record['anatomy'], record['seed'], record['method']) for record in records if not record.get('error')}
    jobs = [(anatomy, 2600000000+index, method, configuration,
             str(arguments.audit_dir) if arguments.audit_dir else None)
            for anatomy in ANATOMIES for index in range(6) for method in arguments.methods
            if (anatomy, 2600000000+index, method) not in completed]
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
