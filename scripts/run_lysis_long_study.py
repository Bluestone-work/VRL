"""Persistent four-arm 2M-agent-step study with fixed 1M/2M checkpoints.

Do not select checkpoints using the held-out topology evaluation. Both fixed
budgets are reported. The older 500-update jobs are separately named pilots.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'research/runs/EXP0074_2M_GENERALIZATION_20261009'
ARMS = ['nav_tf_v3', 'flow_aux', 'belief', 'belief_rolling_tpg']
SEEDS = [7100, 7101, 7102]

def status(**kw):
    (STUDY/'status.json').write_text(json.dumps(dict(updated_at=time.time(), **kw), indent=2))

def curves():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target = ROOT/'research/figures/EXP0074_2M_GENERALIZATION_20261009'
    target.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), dpi=180)
    for arm in ARMS:
        for seed in SEEDS:
            p = STUDY/f'{arm}_s{seed}'/'log.jsonl'
            if not p.exists(): continue
            rows = [json.loads(x) for x in p.read_text().splitlines() if x]
            for ax, key in zip(axs, ['ret', 'success', 'removal']):
                vals = [(r['agent_steps'], r[key]) for r in rows if r[key] is not None]
                if vals: ax.plot(*zip(*vals), label=f'{arm}/{seed}', linewidth=.8)
    for ax, title in zip(axs, ['Training return', 'Training task success', 'Training removal']):
        ax.set_title(title); ax.set_xlabel('Retained active-agent transitions'); ax.grid(alpha=.2)
    axs[-1].legend(fontsize=6); fig.tight_layout()
    fig.savefig(target/'training_curves.png'); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--wait-pids', default='')
    a = ap.parse_args(); STUDY.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1')
    waiting = [int(x) for x in a.wait_pids.split(',') if x]
    while waiting:
        live=[]
        for pid in waiting:
            try: os.kill(pid, 0); live.append(pid)
            except ProcessLookupError: pass
        waiting=live; status(phase='waiting_for_1M_pilots', pids=waiting)
        if waiting: time.sleep(20)
    pending=[(arm, seed) for arm in ARMS for seed in SEEDS]
    running={}; completed=[]
    while pending or running:
        for device in (0, 1):
            if device not in running and pending:
                arm,seed=pending.pop(0); out=STUDY/f'{arm}_s{seed}'
                if out.exists(): raise RuntimeError(f'Refusing to overwrite {out}')
                log=(STUDY/f'{arm}_s{seed}.stdout.log').open('w')
                cmd=[sys.executable, 'scripts/train_lysis_nav.py', '--out', str(out),
                     '--agent-steps', '2000000', '--workers', '4', '--seed', str(seed),
                     '--device', f'cuda:{device}', '--matrix-arm', arm, '--speed-prior',
                     '--latency-max', '3', '--flow-levels', '0.025,0.05,0.1',
                     '--scene-seed-base', '2300000000']
                proc=subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                running[device]=(proc,log,arm,seed)
        for device,(proc,log,arm,seed) in list(running.items()):
            rc=proc.poll()
            if rc is not None:
                log.close(); del running[device]
                if rc: raise RuntimeError(f'{arm}/{seed} failed with exit {rc}')
                completed.append([arm,seed])
        curves()
        status(phase='training', completed=completed, pending=pending,
               running=[dict(arm=x[2], seed=x[3], pid=x[0].pid) for x in running.values()])
        if pending or running: time.sleep(30)
    # Fixed budgets; no checkpoint selection or optimisation on either split.
    eval_completed=[]
    for budget in (1000000,2000000):
        for arm in ARMS:
            for seed in SEEDS:
                policy=STUDY/f'{arm}_s{seed}'/f'policy_steps{budget}.pt'
                out=ROOT/'research/validation/EXP0074_2M_GENERALIZATION_20261009'/f'{arm}_s{seed}_{budget}'
                status(phase='evaluation', budget=budget, arm=arm, seed=seed, completed=eval_completed)
                with (STUDY/f'{arm}_s{seed}_{budget}.eval.log').open('w') as log:
                    subprocess.run([sys.executable, 'scripts/evaluate_lysis_nav.py',
                                    '--policy', str(policy), '--method', arm, '--out', str(out),
                                    '--workers', '16'], cwd=ROOT, env=env, check=True,
                                   stdout=log, stderr=subprocess.STDOUT)
                rows=[json.loads(x) for x in (out/'episodes.jsonl').read_text().splitlines()]
                manifest=json.loads((out/'manifest.json').read_text())
                if len(rows)!=len(manifest['jobs']) or any('error' in r for r in rows):
                    raise RuntimeError(f'Incomplete/failed evaluation: {out}')
                eval_completed.append([arm,seed,budget])
    status(phase='complete', training=completed, evaluation=eval_completed)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        if STUDY.exists(): status(phase='failed', error=repr(exc))
        raise
