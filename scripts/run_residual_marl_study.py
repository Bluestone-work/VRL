"""Run requested EXP0056 attempts once, retaining failures and exact scene pairing."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.run_residual_marl import ROOT, PROTOCOL, source_hashes, protocol
from scripts.run_measured_marl import enforce_runtime
from scripts.run_option_learning import atomic_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--preflight-only', action='store_true')
    args = ap.parse_args()
    args.out = args.out.resolve()
    runtime = enforce_runtime()
    p = protocol(); source = source_hashes()
    args.out.mkdir(parents=True, exist_ok=False)
    for name in source:
        destination = args.out/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    atomic_json(args.out/'manifest.json', dict(protocol=p, source_hashes=source,
        runtime=runtime, confirmation_accessed=False, created_at=datetime.now().astimezone().isoformat()))
    attempts = []
    environment = dict(os.environ, PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONFAULTHANDLER='1')

    def job(name, command):
        return dict(name=name, command=[sys.executable, '-m']+command)

    def run(j):
        folder = args.out/'jobs'/j['name']; folder.mkdir(parents=True, exist_ok=False)
        rec = dict(**j, requested_at=datetime.now().astimezone().isoformat())
        atomic_json(folder/'requested.json', rec); start = time.monotonic()
        with (folder/'stdout.log').open('x') as stream:
            process = subprocess.run(j['command'], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        rec.update(returncode=process.returncode, wall_s=time.monotonic()-start, source_unchanged=source==source_hashes())
        atomic_json(folder/'completion.json', rec)
        return rec

    def batch(jobs, phase):
        atomic_json(args.out/f'{phase}_requested.json', jobs)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run, j) for j in jobs]
            for future in as_completed(futures):
                record = future.result(); attempts.append(record)
                with (args.out/'attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                atomic_json(args.out/'status.json', dict(phase=phase, completed_attempts=len(attempts),
                    failures=sum(r['returncode'] != 0 or not r['source_unchanged'] for r in attempts)))
                print(json.dumps(record), flush=True)
        if any(r['returncode'] != 0 or not r['source_unchanged'] for r in attempts):
            atomic_json(args.out/'status.json', dict(phase='failed_gate', attempts=len(attempts), ranking_admitted=False))
            raise SystemExit(2)

    def evaljob(label, scene, variant='r_mappo', checkpoint=None, rule=False, phase='results'):
        command = ['scripts.run_residual_marl', 'evaluate', '--variant', variant, '--scene', str(scene),
            '--duration', str(p['duration_s']), '--out', str(args.out/phase/f'{label}.jsonl')]
        if checkpoint:
            command += ['--checkpoint', str(checkpoint)]
        if rule:
            command += ['--rule']
        return job(label, command)

    # Full-duration deterministic initialization checks, not just a few steps.
    preflight = []
    for scene in range(p['preflight_scene_base'], p['preflight_scene_base']+p['preflight_scenes']):
        preflight.append(evaljob(f'pre_tpg_{scene}', scene, rule=True, phase='preflight'))
        for variant in p['variants']:
            preflight.append(evaljob(f'pre_{variant}_{scene}', scene, variant, phase='preflight'))
    batch(preflight, 'preflight')
    checks = []
    for scene in range(p['preflight_scene_base'], p['preflight_scene_base']+p['preflight_scenes']):
        ref = json.loads((args.out/'preflight'/f'pre_tpg_{scene}.jsonl').read_text())
        for variant in p['variants']:
            other = json.loads((args.out/'preflight'/f'pre_{variant}_{scene}.jsonl').read_text())
            checks.append(dict(scene=scene, variant=variant,
                initial=ref['actual_initial_snapshot_hash']==other['actual_initial_snapshot_hash'],
                scenario=ref['scenario_hash']==other['scenario_hash'],
                final=ref['final_state_hash']==other['final_state_hash']))
    passed = all(c['initial'] and c['scenario'] and c['final'] for c in checks)
    atomic_json(args.out/'preflight_audit.json', dict(passed=passed, checks=checks))
    if not passed:
        atomic_json(args.out/'status.json', dict(phase='failed_identity', ranking_admitted=False))
        raise SystemExit(2)
    if args.preflight_only:
        atomic_json(args.out/'status.json', dict(phase='preflight_completed', ranking_admitted=False))
        return
    # Short learning checks are admission, not candidates for result selection.
    regressions = []
    for variant in p['variants']:
        regressions.append(job('regression_'+variant, ['scripts.run_residual_marl', 'train',
            '--variant', variant, '--seed', '42', '--steps', '2048',
            '--scene-base', str(p['preflight_scene_base']+1000), '--out', str(args.out/'regressions'/variant)]))
    batch(regressions, 'regressions')
    regression_checks = []
    for variant in p['variants']:
        stat = json.loads((args.out/'regressions'/variant/'status.json').read_text())
        regression_checks.append(dict(variant=variant, completed=stat['phase']=='completed',
            high_changed=stat['model_l2_change']['high_actor'] > 0,
            low_changed=stat['model_l2_change']['low_actor'] > 0, high_samples=stat['high_samples']))
    passed = all(r['completed'] and r['high_changed'] and r['low_changed'] for r in regression_checks)
    atomic_json(args.out/'regression_audit.json', dict(passed=passed, checks=regression_checks))
    if not passed:
        atomic_json(args.out/'status.json', dict(phase='failed_regression', ranking_admitted=False))
        raise SystemExit(2)
    training = []
    for variant in p['variants']:
        for i, seed in enumerate(p['training_seeds']):
            name = f'train_{variant}_{seed}'
            training.append(job(name, ['scripts.run_residual_marl', 'train', '--variant', variant,
                '--seed', str(seed), '--steps', str(p['training_steps']),
                '--scene-base', str(p['training_scene_base']+i*p['training_scene_stride']),
                '--out', str(args.out/'training'/name)]))
    batch(training, 'training')
    evaluations = []
    for scene in range(p['validation_scene_base'], p['validation_scene_base']+p['validation_scenes']):
        for variant in p['variants']:
            for seed in p['training_seeds']:
                for step in p['checkpoints']:
                    checkpoint = args.out/'training'/f'train_{variant}_{seed}'/f'policy_{step}.pt'
                    label = f'{variant}_{seed}_{step}_{scene}'
                    evaluations.append(evaljob(label, scene, variant, checkpoint))
        evaluations.append(evaljob(f'tpg_{scene}', scene, rule=True))
        for label, policy, n in (('balanced_1s','balanced',3), ('memory_n1','memory',1)):
            name = f'{label}_{scene}'
            evaluations.append(job(name, ['scripts.run_aligned_options', 'evaluate', '--policy', policy,
                '--variant', 'hierarchical', '--decision-steps', '10', '--clusters', str(n),
                '--scene-base', str(scene), '--episodes', '1', '--duration', str(p['duration_s']),
                '--out', str(args.out/'results'/f'{name}.jsonl')]))
    batch(evaluations, 'evaluation')
    atomic_json(args.out/'status.json', dict(phase='completed', training_runs=len(training),
        evaluations=len(evaluations), attempts=len(attempts), source_unchanged=source==source_hashes(),
        confirmation_accessed=False))
    analysis = subprocess.run([sys.executable, '-m', 'scripts.analyze_residual_marl', str(args.out)], cwd=ROOT, env=environment)
    if analysis.returncode:
        raise SystemExit(analysis.returncode)


if __name__ == '__main__':
    main()
