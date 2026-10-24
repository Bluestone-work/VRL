"""Bounded SAC development: matched fixed/automatic entropy budget curves."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'research/runs/EXP0110_SAC_LEARNING_CURVES_20261013'
RESULTS = ROOT / 'research/validation/EXP0110_SAC_LEARNING_CURVES_20261013'


def report(xlabel='training iterations'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = ['strict_success', 'removal', 'removal_auc', 'wall_contact_s',
               'spacing_violation_pair_s', 'particle_contact_s', 'particle_collision_events', 'lost']
    summaries = []
    for file in sorted(RESULTS.glob('*/episodes.jsonl')):
        groups = {}
        for line in file.read_text().splitlines():
            row = json.loads(line)
            groups.setdefault(row['method'], []).append(row)
        for method, rows in groups.items():
            summary = dict(evaluation=file.parent.name, method=method, episodes=len(rows))
            summary.update({k:sum(float(r[k]) for r in rows)/len(rows) for k in metrics+['task_success']})
            reached = [r['t90_s'] for r in rows if r['t90_s'] is not None]
            summary.update(T90_300=sum(r['t90_s'] if r['t90_s'] is not None else 300. for r in rows)/len(rows),
                           t90_reached=len(reached)/len(rows),
                           conditional_T90=sum(reached)/len(reached) if reached else None)
            summaries.append(summary)
    (RESULTS/'learning_curve.json').write_text(json.dumps(summaries,indent=2))
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for arm in ('fixed', 'auto'):
        rows = [r for r in summaries if r['evaluation'].startswith(arm+'_') and r['method'].startswith('sac=')]
        rows.sort(key=lambda r:int(r['evaluation'].split('_')[-1]))
        for ax, metric in zip(axes.flat, metrics):
            ax.plot([int(r['evaluation'].split('_')[-1]) for r in rows], [r[metric] for r in rows], 'o-', label=arm)
            ax.set_title(metric);ax.set_xlabel(xlabel);ax.grid(alpha=.2)
    for method in ('parent','adaptive_settle'):
        baseline=next((r for r in summaries if r['method']==method),None)
        if baseline:
            for ax,metric in zip(axes.flat,metrics):ax.axhline(baseline[metric],ls='--',alpha=.6,label=method)
    axes.flat[0].legend();fig.tight_layout();fig.savefig(RESULTS/'validation_curves.png');plt.close(fig)
    keys = ['alpha','entropy','q_mean','q_target_mean','td_absolute_error','action_saturation','actor_grad_norm','critic_grad_norm']
    fig, axes = plt.subplots(2,4,figsize=(16,7))
    for arm in ('fixed','auto'):
        file = RUNS/arm/'seed0/training.jsonl'
        if not file.exists():continue
        rows = [json.loads(l) for l in file.read_text().splitlines()]
        rows = [r for r in rows if 'sac_diagnostics' in r]
        for ax,key in zip(axes.flat,keys):
            ax.plot([r['environment_steps'] for r in rows],[r['sac_diagnostics'][key] for r in rows],label=arm)
            ax.set_title(key);ax.set_xlabel('physical environment steps');ax.grid(alpha=.2)
    axes.flat[0].legend();fig.tight_layout();fig.savefig(RESULTS/'training_diagnostics.png');plt.close(fig)


def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    commands=[]
    for arm in ('fixed','auto'):
        folder=RUNS/arm/'seed0'
        cmd=[sys.executable,'-m','scripts.train_finite_image_flow_sac','--out',str(folder),
             '--seed','0','--workers','2','--iterations','480','--rollout','96',
             '--gradient-steps','16','--batch','256','--capacity','30000','--device','cpu',
             '--critic-warmup-iterations','16','--action-space','vector','--dynamic-obstacles',
             '--dynamic-obstacle-count','16','--obstacle-features','--team-reward']
        if arm=='auto':cmd+=['--auto-alpha']
        commands.append(cmd)
        (RESULTS/'commands.json').write_text(json.dumps(commands,indent=2))
        with (RESULTS/f'{arm}_training.log').open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            try:
                for iteration in (0,120,240,480):
                    checkpoint=folder/('initial.pt' if iteration==0 else f'policy_it{iteration}.pt')
                    while not checkpoint.exists():
                        if proc.poll() is not None:raise RuntimeError(f'{arm} trainer exited before checkpoint {iteration}: {proc.returncode}')
                        time.sleep(10)
                    methods=f'sac={checkpoint}'
                    if arm=='fixed' and iteration==0:methods='parent,adaptive_settle,'+methods
                    ev=[sys.executable,'-m','scripts.eval_image_flow_sac','--out',str(RESULTS/f'{arm}_{iteration}'),
                        '--pool','screen','--workers','4','--methods',methods,
                        '--dynamic-obstacles','--dynamic-obstacle-count','16','--attribution']
                    commands.append(ev);(RESULTS/'commands.json').write_text(json.dumps(commands,indent=2))
                    subprocess.run(ev,cwd=ROOT,check=True)
                    failures=RESULTS/f'{arm}_{iteration}'/'failures.jsonl'
                    if failures.exists() and failures.stat().st_size:raise RuntimeError(f'Evaluation errors: {failures}')
                    report()
                if proc.wait()!=0:raise RuntimeError(f'{arm} training failed')
            finally:
                if proc.poll() is None:
                    proc.terminate();proc.wait()
    report()
    (RESULTS/'COMPLETE.json').write_text(json.dumps({'complete':True}))


if __name__=='__main__':main()
