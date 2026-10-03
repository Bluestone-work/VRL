"""Cross-anatomy sealed comparison: pure RL vs traditional controllers vs residual RL, 14 anatomies.

Every (method, seed, anatomy) is one sealed-test evaluation of that anatomy's 500 registered test layouts.
Checkpoints and thresholds are fixed in advance (all chosen on MCA only). Study CROSS_ANATOMY_20261003.
Jobs already in the ledger for the same weights + protocol + anatomy are reused (cached), not re-run.
"""
import itertools, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
PY = '/home/wj/miniconda3/envs/v/bin/python'
INIT = 'research/initializations/CROSS_ANATOMY_20261003'
METHODS = {  # name: (protocol, checkpoint pattern, seeds)
    'pure_rl': ('CROSS_ANATOMY_PURE_RL', 'pure_rl_exp40_seed_{s}.pt', (42, 43, 44)),
    'route_prior': ('CROSS_ANATOMY_ROUTE_PRIOR', 'route_prior_prior_only.pt', (0,)),
    'route_avoid': ('CROSS_ANATOMY_ROUTE_AVOID', 'route_avoid_prior_only.pt', (0,)),
    'route_avoid_wait': ('CROSS_ANATOMY_ROUTE_AVOID_WAIT', 'route_avoid_wait_prior_only.pt', (0,)),
    'residual': ('CROSS_ANATOMY_RESIDUAL', 'residual_exp43_seed_{s}.pt', (42, 43, 44)),
    'residual_shield': ('CROSS_ANATOMY_RESIDUAL_SHIELD', 'residual_shield_exp43_seed_{s}.pt', (42, 43, 44)),
}
anatomies = json.loads((ROOT/'configs/evaluation_splits.json').read_text())['anatomy_order']
jobs = [(m, s, a) for m, (_, _, seeds) in METHODS.items() for s in seeds for a in anatomies]
CPUS = ['0-1', '2-3', '4-5', '8-9', '10-11', '12-13', '14-15', '16-17', '18-19', '20-21', '22-23']


def run(i_job):
    i, (m, s, a) = i_job
    proto, pattern, _ = METHODS[m]
    cmd = [PY, 'scripts/evaluate_mca_sealed_test.py', '--protocol', f'configs/experiments/{proto}.json',
           '--checkpoint', f'{INIT}/{pattern.format(s=s)}', '--study', 'CROSS_ANATOMY_20261003',
           '--label', f'{m}_seed_{s}_{a}', '--anatomy', a, '--workers', '2', '--declare-candidates', str(len(jobs))]
    env = dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    r = subprocess.run(['taskset', '-c', CPUS[i % len(CPUS)], *cmd], cwd=ROOT, env=env, capture_output=True, text=True)
    line = (r.stdout.strip().splitlines() or [''])[-1]
    print(json.dumps(dict(method=m, seed=s, anatomy=a, rc=r.returncode, out=line[:200], err=r.stderr[-300:] if r.returncode else '')), flush=True)


# The first call declares the budget; run it alone so concurrent first accesses cannot race.
run((0, jobs[0]))
with ThreadPoolExecutor(len(CPUS)) as pool:
    list(pool.map(run, enumerate(jobs[1:], 1)))
print('DONE', flush=True)
