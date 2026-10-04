"""Separate debug reproductions; never replace failed performance attempts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.evaluate_multicluster import source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    before = source_hashes()
    jobs = [('plain', i) for i in range(12)]+[('gdb', i) for i in range(4)]
    env = os.environ.copy()
    env.update(PYTHONPATH=str(root), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONFAULTHANDLER='1')
    manifest = dict(registered_at=datetime.now().astimezone().isoformat(),
        purpose='debug-only; no replacement of matrix failure', workers=4,
        scene_seed=1302010005, control_seed=42, source_hashes=before, jobs=jobs)
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    def run(job):
        kind, index = job
        label = f'{kind}_{index:02d}'
        output = (args.out/(label+'.jsonl')).resolve()
        command = [sys.executable, '-X', 'faulthandler', 'scripts/evaluate_multicluster.py',
            '--method', 'multi_unshielded', '--clusters', '3', '--d-min-mm', '2',
            '--episodes', '1', '--seed-base', '1302010005', '--control-seed', '42',
            '--budget', 'fixed_total', '--duration-s', '180', '--out', str(output)]
        if kind == 'gdb':
            command = ['gdb', '-q', '-batch', '-ex', 'set pagination off', '-ex',
                'set debuginfod enabled off', '-ex', 'run', '-ex', 'thread apply all bt',
                '-ex', 'x/8i $pc-8', '--args']+command
        start = time.monotonic()
        try:
            child = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                   text=True, timeout=120)
            (args.out/(label+'.stdout.txt')).write_text(child.stdout)
            (args.out/(label+'.stderr.txt')).write_text(child.stderr)
            lines = output.read_text().splitlines() if output.exists() else []
            result = json.loads(lines[0]) if lines else None
            # gdb may return 1 after a normal inferior exit when querying $pc.
            native_signal = ('Program received signal' in child.stdout or
                             'Fatal Python error' in child.stderr or child.returncode < 0)
            row = dict(label=label, kind=kind, returncode=child.returncode,
                native_signal=native_signal, completed=bool(result and result.get('status') == 'completed'),
                final_state_hash=result['final_state_hash'] if result else None,
                walltime_s=time.monotonic()-start)
        except subprocess.TimeoutExpired:
            row = dict(label=label, kind=kind, timeout=True, completed=False,
                       walltime_s=time.monotonic()-start)
        return row
    rows = []
    with (args.out/'attempts.jsonl').open('x') as f, ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(run, job) for job in jobs]):
            row = future.result(); rows.append(row)
            f.write(json.dumps(row)+'\n'); f.flush()
            print(json.dumps(row), flush=True)
    assert before == source_hashes(), 'Source changed during reproduction'
    summary = dict(requested=len(jobs), completed=sum(r['completed'] for r in rows),
        native_signals=sum(r.get('native_signal', False) for r in rows),
        unique_completed_state_hashes=sorted({r['final_state_hash'] for r in rows if r['completed']}),
        source_unchanged=True, performance_rows_replaced=False, attempts=rows)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
