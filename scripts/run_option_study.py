"""Process-isolated, fixed-scene EXP0051 pilot; every attempt is retained."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.stats import beta
from scripts.option_learning_episode import ROOT, PROTOCOL, option_hashes


def dump(path, value):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n');tmp.replace(path)


def baseline_arms():
    return [dict(label='memory_n3', policy='memory', clusters=3, variant='hierarchical'),
        dict(label='balanced_n3', policy='balanced', clusters=3, variant='hierarchical'),
        dict(label='priority_n3', policy='priority', clusters=3, variant='hierarchical'),
        dict(label='memory_n1', policy='memory', clusters=1, variant='hierarchical'),
        dict(label='legacy_memory_n3', policy='legacy_memory', clusters=3, variant='hierarchical'),
        dict(label='untrained_n3', policy='untrained', clusters=3, variant='hierarchical')]


def environment():
    env=os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONFAULTHANDLER='1')
    return env


def run_process(command, log, timeout):
    started=time.monotonic()
    try:
        with Path(log).open('x') as stream:
            p=subprocess.run([sys.executable,'-X','faulthandler']+command,cwd=ROOT,env=environment(),
                             stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
        return dict(status='completed' if p.returncode==0 else 'process_exit',returncode=p.returncode,
                    wall_s=time.monotonic()-started,console=str(log))
    except subprocess.TimeoutExpired:
        return dict(status='timeout',wall_s=time.monotonic()-started,console=str(log))


def evaluate_matrix(folder, arms, scenes, *, workers=3, coupling=0., phase='validation'):
    folder.mkdir(parents=True,exist_ok=False)
    parts=folder/'parts';parts.mkdir()
    source=option_hashes()
    manifest=dict(experiment='EXP0051',phase=phase,stage='validation',scenes=list(scenes),
        protocol=json.loads(PROTOCOL.read_text()),
        arms=arms,requested=len(arms)*len(scenes),source_hashes=source,
        observation_contract='tracked_options',actuator_coupling=coupling,
        confirmation_accessed=False,registered_at=datetime.now().astimezone().isoformat())
    dump(folder/'manifest.json',manifest)
    def run(arm,seed):
        prefix=parts/f"{arm['label']}_scene{seed}"
        command=['scripts/run_option_learning.py','evaluate','--policy',arm['policy'],
            '--clusters',str(arm['clusters']),'--variant',arm['variant'],
            '--scene-base',str(seed),'--coupling',str(coupling),'--out',str(prefix.with_suffix('.jsonl'))]
        if arm.get('checkpoint'):command+=['--checkpoint',arm['checkpoint']]
        row=dict(arm=arm['label'],scene_seed=seed,clusters=arm['clusters'])
        process=run_process(command,prefix.with_suffix('.console.txt'),240.)
        row.update(process)
        if process['status']=='completed':
            try:
                measured=json.loads(prefix.with_suffix('.jsonl').read_text().splitlines()[0])
                assert measured['source_hashes']==source and measured['scene_seed']==seed
                assert measured['status']=='completed'
                row.update(measured)
            except (OSError,ValueError,AssertionError,IndexError) as exc:
                row.update(status='invalid_output',error=repr(exc))
        return row
    rows=[]
    with (folder/'attempts.jsonl').open('x',buffering=1) as stream,ThreadPoolExecutor(workers) as pool:
        futures=[pool.submit(run,a,s) for a in arms for s in scenes]
        for future in as_completed(futures):
            row=future.result();rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            progress=dict(recorded=len(rows),requested=manifest['requested'],
                failed=sum(r['status']!='completed' for r in rows),last_arm=row['arm'])
            dump(folder/'progress.json',progress)
            print(json.dumps(dict(phase=phase,**progress)),flush=True)
    summary=analyze(folder,manifest,rows)
    return summary


def bootstrap(x,seed=51051):
    x=np.asarray(x,float);rng=np.random.default_rng(seed)
    samples=rng.choice(x,size=(5000,len(x)),replace=True).mean(axis=1)
    return dict(mean=float(x.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),scenes=len(x))


def analyze(folder,manifest,rows):
    metrics=('cluster_safe_success','removal','removal_auc_180','wall_contact_s',
             'spacing_violation_pair_s','minimum_spacing_mm','supervisor_infeasible_steps')
    issues=[];by_arm=defaultdict(list);by_scene=defaultdict(list)
    for row in rows:
        by_arm[row['arm']].append(row)
        if row['status']!='completed':
            issues.append(f"Failed {row['arm']} scene {row['scene_seed']}: {row['status']}")
            continue
        by_scene[row['scene_seed']].append(row)
        if row['source_hashes']!=manifest['source_hashes']:issues.append('Source mismatch')
        if not row.get('completion_uses_observed_confirmation'):issues.append('Oracle completion')
        if row.get('actuator_coupling')!=manifest['actuator_coupling']:issues.append('Actuator mismatch')
    for arm in manifest['arms']:
        if sorted(r['scene_seed'] for r in by_arm[arm['label']])!=sorted(manifest['scenes']):
            issues.append('Missing/duplicate scene '+arm['label'])
    for seed,rr in by_scene.items():
        for field in ('scenario_hash','sensor_assumptions','supervisor_settings'):
            if len({json.dumps(r[field],sort_keys=True) for r in rr})!=1:issues.append(f'Unpaired {field}: {seed}')
        for n in (1,3):
            sub=[r for r in rr if r['clusters']==n]
            for field in ('actual_initial_snapshot_hash','actual_initial_snapshot'):
                if len({json.dumps(r[field],sort_keys=True) for r in sub})>1:issues.append(f'Unpaired actual reset {seed} N={n}')
            if len({json.dumps(r['reset_info']['actual_config'],sort_keys=True) for r in sub})>1:
                issues.append(f'Physical config mismatch {seed} N={n}')
        budgets=[r['reset_info']['aggregate_lysis_capacity_mass_per_s'] for r in rr]
        if max(budgets)-min(budgets)>1e-12:issues.append('Catalytic proxy mismatch')
    summary=dict(audit_passed=not issues,issues=issues,requested=manifest['requested'],
        completed=sum(r['status']=='completed' for r in rows),scenes=len(manifest['scenes']),
        phase=manifest['phase'],confirmation_accessed=False,learning_superiority_established=False,
        arms={},comparisons=[],untrained_equivalence=None)
    for label,data in by_arm.items():
        complete=[r for r in data if r['status']=='completed']
        success=sum(bool(r['cluster_safe_success']) for r in complete)
        low=0. if success==0 else float(beta.ppf(.025,success,len(data)-success+1))
        highsuccess=success+len(data)-len(complete)
        high=1. if highsuccess==len(data) else float(beta.ppf(.975,highsuccess+1,len(data)-highsuccess))
        summary['arms'][label]=dict(completed=len(complete),requested=len(data),safe_successes=success,
            safe_rate_ci95_missing_envelope=[low,high],
            conditional_means={k:float(np.mean([r[k] for r in complete if r[k] is not None]))
                if any(r[k] is not None for r in complete) else None for k in metrics},
            spacing_violating_episodes=sum(not r['spacing_compliant'] for r in complete))
    if 'memory_n3' in by_arm and 'untrained_n3' in by_arm:
        a={r['scene_seed']:r.get('final_state_hash') for r in by_arm['memory_n3'] if r['status']=='completed'}
        b={r['scene_seed']:r.get('final_state_hash') for r in by_arm['untrained_n3'] if r['status']=='completed'}
        summary['untrained_equivalence']=a==b and len(a)==len(manifest['scenes'])
        if not summary['untrained_equivalence']:issues.append('Untrained memory equivalence failed or incomplete')
    if not issues:
        grouped={}
        for variant in ('hierarchical','flat'):
            group=defaultdict(list)
            for label,data in by_arm.items():
                if label.startswith(variant+'_s'):
                    for r in data:group[r['scene_seed']].append(r)
            grouped[variant]=group
            if not group:continue
            for baseline in ('memory_n3','balanced_n3','priority_n3'):
                if baseline not in by_arm:continue
                reference={r['scene_seed']:r for r in by_arm[baseline]}
                if group.keys()!=reference.keys():raise ValueError('Learning comparison not paired')
                comparison=dict(learning=variant,baseline=baseline,
                    inference_unit='scene; all three training seeds averaged within scene',metrics={})
                for k in metrics:
                    differences=[np.mean([r[k] for r in group[s]])-reference[s][k] for s in sorted(group)]
                    comparison['metrics'][k]=bootstrap(differences)
                summary['comparisons'].append(comparison)
        if grouped['hierarchical'] and grouped['flat']:
            a,b=grouped['hierarchical'],grouped['flat']
            if a.keys()!=b.keys():raise ValueError('Hierarchy ablation not paired')
            summary['hierarchy_vs_flat']={k:bootstrap([np.mean([r[k] for r in a[s]])-
                np.mean([r[k] for r in b[s]]) for s in sorted(a)]) for k in metrics}
    summary['audit_passed']=not issues
    dump(folder/'analysis.json',summary)
    lines=['# EXP0051 '+manifest['phase'],'',
        f"Audit: {'PASS' if not issues else 'BLOCKED'}; {summary['completed']}/{summary['requested']} attempts complete; {summary['scenes']} paired scenes.",'',
        'Development only. Training-seed repeats are not independent scenes. Intervals are exploratory, not multiplicity adjusted. Zero/zero bootstrap differences do not prove equivalence. No calibrated magnetic independence or safety certificate.','',
        '| Arm | Complete/requested | Safe success | Removal % | AUC % | Wall cluster-s | Spacing pair-s | Violating scenes |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,a in summary['arms'].items():
        m=a['conditional_means']
        if a['completed']:
            lines.append(f"| {label} | {a['completed']}/{a['requested']} | {a['safe_successes']}/{a['requested']} | {100*m['removal']:.2f} | {100*m['removal_auc_180']:.2f} | {m['wall_contact_s']:.3f} | {m['spacing_violation_pair_s']:.3f} | {a['spacing_violating_episodes']} |")
        else:lines.append(f"| {label} | 0/{a['requested']} | unknown | — | — | — | — | unknown |")
    lines+=['','## Gates','',f"Untrained equals memory: {summary['untrained_equivalence']}",
        'All failures are retained; incomplete attempts block formal comparisons.',
        'The legacy-memory arm has a different safety layer, so any difference alone is not a learning gain.']
    lines += ['- '+issue for issue in issues]
    for item in summary['comparisons']:
        lines+=['',f"## {item['learning']} minus {item['baseline']}",'']
        for k in ('cluster_safe_success','removal','removal_auc_180','wall_contact_s','spacing_violation_pair_s'):
            v=item['metrics'][k]
            lines.append(f"- {k}: {v['mean']:.6f}, paired scene bootstrap 95% [{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}].")
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('preflight','study'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();protocol=json.loads(PROTOCOL.read_text())
    if args.phase=='preflight':
        seeds=range(protocol['preflight_scene_base'],protocol['preflight_scene_base']+2)
        summary=evaluate_matrix(args.out,baseline_arms(),seeds,workers=args.workers,phase='preflight')
        return 0 if summary['audit_passed'] else 2
    args.out.mkdir(parents=True,exist_ok=False)
    train=args.out/'training';train.mkdir()
    tasks=[]
    for variant in protocol['variants']:
        for index,seed in enumerate(protocol['training_seeds']):
            tasks.append(dict(label=f'{variant}_s{seed}',variant=variant,seed=seed,
                scene_base=protocol['training_scene_base']+index*protocol['training_scene_stride']))
    dump(args.out/'study_manifest.json',dict(protocol=protocol,source_hashes=option_hashes(),
        tasks=tasks,confirmation_accessed=False,started_at=datetime.now().astimezone().isoformat()))
    # Preserve exact frozen sources, including untracked files, for replay.
    for name in option_hashes():
        destination=args.out/'source_snapshot'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    def training(task):
        label=task['label'];out=train/label
        command=['scripts/run_option_learning.py','train','--variant',task['variant'],
            '--seed',str(task['seed']),'--scene-base',str(task['scene_base']),
            '--steps',str(protocol['pilot_steps_per_seed']),'--out',str(out)]
        result=dict(task,**run_process(command,train/(label+'.console.txt'),2400.))
        dump(train/(label+'.process.json'),result);print(json.dumps(result),flush=True)
        return result
    dump(args.out/'status.json',dict(phase='training',requested_runs=len(tasks)))
    with ThreadPoolExecutor(args.workers) as pool:
        result=list(pool.map(training,tasks))
    if any(r['status']!='completed' for r in result):
        dump(args.out/'status.json',dict(phase='training_failed_retained',runs=result,confirmation_accessed=False))
        return 2
    arms=baseline_arms()
    for task in tasks:
        path=train/task['label']/f"policy_{protocol['pilot_steps_per_seed']}.pt"
        arms.append(dict(label=task['label'],variant=task['variant'],policy='learned',clusters=3,
            checkpoint=str(path.resolve()),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    dump(args.out/'status.json',dict(phase='paired_validation',training_runs=result))
    main_result=evaluate_matrix(args.out/'validation',arms,
        range(protocol['validation_scene_base'],protocol['validation_scene_base']+protocol['validation_scenes']),
        workers=args.workers,phase='paired_validation')
    dump(args.out/'status.json',dict(phase='synthetic_coupling_stress',main_audit=main_result['audit_passed']))
    stress_arms=[a for a in arms if a['label'] not in ('legacy_memory_n3','untrained_n3')]
    stress=evaluate_matrix(args.out/'coupling_stress',stress_arms,
        range(protocol['stress_scene_base'],protocol['stress_scene_base']+protocol['stress_scenes']),
        workers=args.workers,coupling=protocol['actuator_stress']['transfer_coupling'],phase='synthetic_coupling_stress')
    final=dict(phase='completed' if main_result['audit_passed'] and stress['audit_passed'] else 'completed_with_retained_failures',
        training_runs=result,main_audit=main_result['audit_passed'],stress_audit=stress['audit_passed'],
        requested_evaluations=main_result['requested']+stress['requested'],
        completed_evaluations=main_result['completed']+stress['completed'],confirmation_accessed=False,
        learning_superiority_established=False)
    dump(args.out/'status.json',final)
    print(json.dumps(final),flush=True)
    return 0 if main_result['audit_passed'] and stress['audit_passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
