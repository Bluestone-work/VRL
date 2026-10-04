"""Complete the registered EXP0049 32k/64k study without selecting test scenes."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT/'research/training/EXP0049_TEMPORAL_CANDIDATES_20261004'
VALIDATION = ROOT/'research/validation'
CODE = 'EXP0049'
CONTRACT = 'temporal'
ENTRYPOINT = 'scripts/run_temporal_learning.py'
CONTINUATION_BASE = 1510010000


def status(phase, **extra):
    data = dict(phase=phase, **extra)
    temporary = TRAIN/'study_status.tmp'
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(TRAIN/'study_status.json')
    print(json.dumps(data), flush=True)


def run(label, args):
    started = time.monotonic()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONFAULTHANDLER='1')
    with (TRAIN/(label+'.console.txt')).open('x') as f:
        child = subprocess.run([sys.executable, '-X', 'faulthandler']+args,
                               cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    result = dict(label=label, returncode=child.returncode, wall_s=time.monotonic()-started)
    (TRAIN/(label+'.process.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result


def evaluation(budget, steps):
    args = ['scripts/evaluate_local_learning_matrix.py', '--contract', CONTRACT,
            '--stage', 'validation', '--workers', '4', '--out',
            str(VALIDATION/f'{CODE}_LEARNED{budget}_VAL_20261004')]
    for seed in (42,43,44):
        args += ['--checkpoint', str(TRAIN/f'seed{seed}_{budget}k/policy_{steps}.pt')]
    return run(f'learned{budget}_validation', args)


def main():
    status('awaiting_registered_pilot')
    deadline = time.monotonic()+900
    paths = [TRAIN/f'seed{s}_32k.process.json' for s in (42,43,44)]
    while not all(p.exists() for p in paths):
        if time.monotonic() > deadline:
            status('pilot_status_timeout'); return 2
        time.sleep(1.)
    pilots = [json.loads(p.read_text()) for p in paths]
    if any(p['returncode'] != 0 for p in pilots):
        status('pilot_failure_retained', attempts=pilots); return 2
    status('evaluate_32k_and_train_64k')
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(evaluation, 32, 32768)]
        for seed in (42,43,44):
            args = [ENTRYPOINT, 'train', '--seed', str(seed),
                    '--scene-base', str(CONTINUATION_BASE+(seed-42)*100000), '--steps', '65536',
                    '--initialize', str(TRAIN/f'seed{seed}_32k/policy_32768.pt'),
                    '--out', str(TRAIN/f'seed{seed}_64k')]
            futures.append(pool.submit(run, f'seed{seed}_64k', args))
        results = [f.result() for f in as_completed(futures)]
    if any(r['returncode'] != 0 and r['label'].startswith('seed') for r in results):
        status('continuation_failure_retained', attempts=results); return 2
    status('evaluate_registered_64k')
    results.append(evaluation(64, 65536))
    status('audit_paired_results')
    args = ['scripts/analyze_local_learning.py']
    folders = [f'{CODE}_BASELINES_VAL_20261004', f'{CODE}_LEARNED32_VAL_20261004',
               f'{CODE}_LEARNED64_VAL_20261004']
    if CODE == 'EXP0049': folders.append('EXP0049_SINGLE_MEMORY_VAL_20261004')
    for name in folders:
        args += ['--input', str(VALIDATION/name)]
    args += ['--out', str(VALIDATION/f'{CODE}_FINAL_DEVELOPMENT_20261004')]
    results.append(run('final_analysis', args))
    if CODE == 'EXP0049':
        results.append(run('architecture_transfer_analysis', ['scripts/analyze_temporal_transfer.py',
            '--feedforward', str(VALIDATION/'EXP0049_FEEDFORWARD64_TRANSFER_VAL_20261004'),
            '--temporal', str(VALIDATION/'EXP0049_LEARNED64_VAL_20261004'),
            '--out', str(VALIDATION/'EXP0049_ARCHITECTURE_TRANSFER_20261004')]))
    failed = any(r['returncode'] != 0 for r in results)
    status('completed_with_recorded_failures' if failed else 'completed_development',
           attempts=results, confirmation_accessed=False)
    return 2 if failed else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('temporal', 'memory_prior'), default='temporal')
    args = parser.parse_args()
    if args.study == 'memory_prior':
        CODE, CONTRACT = 'EXP0050', 'memory_prior'
        TRAIN = ROOT/'research/training/EXP0050_MEMORY_PRIOR_20261004'
        ENTRYPOINT = 'scripts/run_memory_prior_learning.py'
        CONTINUATION_BASE = 1610010000
    raise SystemExit(main())
