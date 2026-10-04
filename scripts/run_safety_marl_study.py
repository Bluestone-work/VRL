"""Conditional matched reward-control follow-up; never weaken comparison methods."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scripts.run_safety_marl import ROOT, PROTOCOL, hashes
from scripts.run_measured_marl import enforce_runtime
from scripts.run_option_learning import atomic_json


def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text());p=manifest['protocol']
    parent=ROOT/p['parent_study'];parent_audit=json.loads((parent/'AUDIT.json').read_text())
    groups=defaultdict(list);rows=[]
    for path in sorted((root/'results').glob('*.jsonl')):
        row=json.loads(path.read_text());rows.append(row)
        groups[row['reward_regime'],row['variant']].append(row)
    attempts=[json.loads(x) for x in (root/'attempts.jsonl').read_text().splitlines()]
    requested=[j for name in ('preflight','training','evaluation') for j in json.loads((root/f'{name}_requested.json').read_text())]
    failed=[r for r in attempts if r['returncode']!=0 or not r['source_unchanged']]
    pair_checks=[];reference_rows={}
    for scene in range(p['validation_scene_base'],p['validation_scene_base']+p['validation_scenes']):
        refpath=parent/'results'/f'tpg_{scene}.jsonl';ref=json.loads(refpath.read_text())
        reference_rows[scene]=ref
        group=[r for r in rows if r['scene_seed']==scene]
        pair_checks.append(dict(scene=scene,rows=len(group),
            same_initial=all(r['actual_initial_snapshot_hash']==ref['actual_initial_snapshot_hash'] for r in group),
            same_scenario=all(r['scenario_hash']==ref['scenario_hash'] for r in group)))
    checkpoint_checks=[]
    for regime in p['regimes']:
        for variant in p['variants']:
            for seed in p['training_seeds']:
                folder=root/'training'/f'{regime}_{variant}_{seed}'
                stat=json.loads((folder/'status.json').read_text())
                checkpoint=stat['checkpoints'][-1]
                digest=hashlib.sha256(Path(checkpoint['path']).read_bytes()).hexdigest()
                matching=[r for r in groups[regime,variant] if r['training_seed']==seed]
                checkpoint_checks.append(dict(regime=regime,variant=variant,seed=seed,
                    completed=stat['phase']=='completed',hash_matches=digest==checkpoint['sha256'],
                    rows_reference_checkpoint=all(r['checkpoint_sha256']==digest for r in matching),
                    matching_scenes=sorted(r['scene_seed'] for r in matching)==list(range(p['validation_scene_base'],p['validation_scene_base']+p['validation_scenes']))))
    baseline_references=[]
    for label in ('tpg','balanced_1s','memory_n1'):
        for scene in reference_rows:
            path=parent/'results'/f'{label}_{scene}.jsonl'
            baseline_references.append(dict(method=label,scene=scene,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cached_from_admitted_parent=True))
    expected=len(p['variants'])*len(p['regimes'])*len(p['training_seeds'])*p['validation_scenes']
    identity=json.loads((root/'preflight_audit.json').read_text())['passed']
    gate=bool(parent_audit['valid_comparison_gate'] and identity and not failed and len(rows)==expected
        and manifest['source_hashes']==hashes() and all(c['same_initial'] and c['same_scenario'] for c in pair_checks)
        and sorted(r['name'] for r in attempts)==sorted(r['name'] for r in requested)
        and len({r['name'] for r in attempts})==len(attempts)
        and all(c['completed'] and c['hash_matches'] and c['rows_reference_checkpoint'] and c['matching_scenes'] for c in checkpoint_checks))
    metrics=('removal','removal_auc_180','wall_contact_s','spacing_violation_pair_s','scheduler_calls','low_nonrule_choices','high_nonrule_choices')
    summaries=[];per_seed=[]
    for (regime,variant),group in sorted(groups.items()):
        s=dict(regime=regime,variant=variant,rows=len(group),independent_scenes=len({r['scene_seed'] for r in group}),safe_successes=sum(r['cluster_safe_success'] for r in group))
        for key in metrics:
            scene_values=defaultdict(list)
            for r in group:scene_values[r['scene_seed']].append(r[key])
            s[key]=float(np.mean([np.mean(v) for v in scene_values.values()]))
        summaries.append(s)
        for seed in p['training_seeds']:
            a=[r for r in group if r['training_seed']==seed]
            per_seed.append(dict(regime=regime,variant=variant,seed=seed,safe_successes=sum(r['cluster_safe_success'] for r in a),
                **{key:float(np.mean([r[key] for r in a])) for key in metrics}))
    paired=[]
    for variant in p['variants']:
        for seed in p['training_seeds']:
            for scene in reference_rows:
                a=next(r for r in groups['original_reward',variant] if r['training_seed']==seed and r['scene_seed']==scene)
                b=next(r for r in groups['safety_reward',variant] if r['training_seed']==seed and r['scene_seed']==scene)
                paired.append(dict(variant=variant,seed=seed,scene=scene,
                    **{key:b[key]-a[key] for key in metrics}))
    audit=dict(valid_comparison_gate=gate,identity_passed=identity,evaluations=len(rows),expected_evaluations=expected,
        attempts=len(attempts),failed_attempts=failed,pairing=pair_checks,checkpoints=checkpoint_checks,
        summaries=summaries,per_seed=per_seed,paired_safety_minus_original=paired,baseline_references=baseline_references,
        additional_control_steps=len(p['regimes'])*len(p['variants'])*len(p['training_seeds'])*p['training_steps'],
        cumulative_controls_per_policy=p['parent_steps']+p['training_steps'],confirmation_accessed=False,
        statement='Matched fine-tuning with fresh optimizers; reward repair is not algorithmic novelty. Development data, not confirmation.')
    atomic_json(root/'AUDIT.json',audit)
    lines=['# EXP0055：匹配继续训练的安全目标实验','',
        f"审计：{'通过' if gate else '未通过，禁止排名'}；{len(rows)}/{expected} 次新评估，共八个固定开发场景。",
        f"所有模型来自同方法的 {p['parent_steps']} 步父模型，追加 {p['training_steps']} 步，累计 {p['parent_steps']+p['training_steps']} 步。两组均重启 Adam 优化器。",
        '原奖励继续训练控制额外数据和优化量；安全奖励仅改变训练罚项，不改变评价阈值、观测或物理环境。','',
        '两组共同采用 actor/critic 分开裁剪梯度；父模型到子模型的差异不能全部归因于奖励或算法创新。','',
        '| 奖励组 | 方法 | 清除 | AUC | 壁面集群秒 | 间距对秒 | 安全成功 |','|---|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['regime']} | {s['variant']} | {s['removal']:.2%} | {s['removal_auc_180']:.2%} | {s['wall_contact_s']:.3f} | {s['spacing_violation_pair_s']:.6f} | {s['safe_successes']}/{s['rows']} |")
    lines+=['','## 同场景传统参照（缓存父实验，不计为新评估）','',
        '| 方法 | 清除 | AUC | 壁面集群秒 | 间距对秒 | 安全成功 |','|---|---:|---:|---:|---:|---:|']
    for label in ('tpg','balanced_1s','memory_n1'):
        s=next(x for x in parent_audit['summaries'] if x['arm']==label)
        gap='N/A' if label=='memory_n1' else f"{s['spacing_violation_pair_s']:.6f}"
        lines.append(f"| {label} | {s['removal']:.2%} | {s['removal_auc_180']:.2%} | {s['wall_contact_s']:.3f} | {gap} | {s['safe_successes']}/{s['rows']} |")
    lines+=['','## 全部种子','', '| 奖励组 | 方法 | 种子 | 清除 | AUC | 壁面集群秒 | 间距对秒 | 安全成功 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in per_seed:
        lines.append(f"| {s['regime']} | {s['variant']} | {s['seed']} | {s['removal']:.2%} | {s['removal_auc_180']:.2%} | {s['wall_contact_s']:.3f} | {s['spacing_violation_pair_s']:.6f} | {s['safe_successes']}/{p['validation_scenes']} |")
    lines+=['','## 归因限制','',
        '- 安全奖励组与原奖励继续训练组的差异用于训练目标诊断；不能与父模型的差值全归因于奖励。',
        '- MAPPO/IPPO 和两个 V-CTPG 变体都得到相同奖励组和追加预算，没有只优化主方法。',
        '- 若只是降低清除换来少碰壁，应报告权衡。安全成功全零时不声称安全目标已解决。',
        '- 2 mm 空间约束是未标定代理；未验证真实磁场独立性、硬件迁移或跨解剖结构泛化。',
        '- 确认池关闭，失败和所有种子保留。没有独立规划器系统比较或 MAT 结果。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=gate,summaries=summaries),indent=2))
    if not gate:raise SystemExit(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();args.out=args.out.resolve()
    p=json.loads(PROTOCOL.read_text());runtime=enforce_runtime();source=hashes()
    parent=ROOT/p['parent_study'];audit=json.loads((parent/'AUDIT.json').read_text())
    if not audit['valid_comparison_gate']:raise ValueError('Parent matrix did not pass admission')
    proposed=next(s for s in audit['summaries'] if s['arm']=='vctpg_ac' and s['steps']==p['parent_steps'])
    classical=max(s['removal_auc_180'] for s in audit['summaries'] if s['arm'] in ('tpg','balanced_1s'))
    reasons=[]
    if proposed['safe_successes']==0:reasons.append('zero_registered_safe_success')
    if proposed['spacing_violation_pair_s']>0:reasons.append('spacing_violation')
    if proposed['removal_auc_180']<=classical:reasons.append('no_AUC_gain_over_strongest_classical')
    if not reasons:
        print(json.dumps(dict(phase='not_triggered',no_new_experiment=True)));return
    args.out.mkdir(parents=True,exist_ok=False)
    atomic_json(args.out/'manifest.json',dict(protocol=p,source_hashes=source,runtime=runtime,
        trigger_reasons=reasons,parent_audit_sha256=hashlib.sha256((parent/'AUDIT.json').read_bytes()).hexdigest(),
        confirmation_accessed=False,started_at=datetime.now().astimezone().isoformat()))
    for name in source:
        path=args.out/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    environment=dict(os.environ,PYTHONPATH=str(ROOT),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONFAULTHANDLER='1')
    attempts=[]

    def job(name,command):return dict(name=name,command=[sys.executable,'-m','scripts.run_safety_marl']+command)

    def run(j):
        folder=args.out/'jobs'/j['name'];folder.mkdir(parents=True,exist_ok=False)
        row=dict(**j,requested_at=datetime.now().astimezone().isoformat())
        atomic_json(folder/'requested.json',row);start=time.monotonic()
        with (folder/'stdout.log').open('x') as stream:
            result=subprocess.run(j['command'],cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode,wall_s=time.monotonic()-start,source_unchanged=source==hashes())
        atomic_json(folder/'completion.json',row);return row

    def batch(jobs,phase):
        atomic_json(args.out/f'{phase}_requested.json',jobs)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run,j) for j in jobs]):
                row=future.result();attempts.append(row)
                with (args.out/'attempts.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                atomic_json(args.out/'status.json',dict(phase=phase,completed_attempts=len(attempts),failures=sum(x['returncode']!=0 for x in attempts)))
                print(json.dumps(row),flush=True)
        if any(x['returncode']!=0 or not x['source_unchanged'] for x in attempts):
            atomic_json(args.out/'status.json',dict(phase='failed_gate',ranking_admitted=False));raise SystemExit(2)

    pre=[]
    for scene in range(p['preflight_scene_base'],p['preflight_scene_base']+p['preflight_scenes']):
        for variant in p['variants']:
            for regime in p['regimes']:
                name=f'pre_{regime}_{variant}_{scene}'
                pre.append(job(name,['evaluate','--variant',variant,'--regime',regime,'--seed','42','--parent-policy',
                    '--scene',str(scene),'--out',str(args.out/'preflight'/f'{name}.jsonl')]))
    batch(pre,'preflight')
    identities=[]
    for scene in range(p['preflight_scene_base'],p['preflight_scene_base']+p['preflight_scenes']):
        for variant in p['variants']:
            a,b=[json.loads((args.out/'preflight'/f'pre_{regime}_{variant}_{scene}.jsonl').read_text()) for regime in p['regimes']]
            identities.append(dict(variant=variant,scene=scene,initial=a['actual_initial_snapshot_hash']==b['actual_initial_snapshot_hash'],
                final=a['final_state_hash']==b['final_state_hash'],parent=a['parent_checkpoint_sha256']==b['parent_checkpoint_sha256']))
    passed=all(x['initial'] and x['final'] and x['parent'] for x in identities)
    atomic_json(args.out/'preflight_audit.json',dict(passed=passed,identities=identities))
    if not passed:
        atomic_json(args.out/'status.json',dict(phase='failed_identity',ranking_admitted=False));raise SystemExit(2)
    jobs=[]
    for regime in p['regimes']:
        for variant in p['variants']:
            for i,seed in enumerate(p['training_seeds']):
                name=f'{regime}_{variant}_{seed}'
                jobs.append(job(name,['train','--regime',regime,'--variant',variant,'--seed',str(seed),
                    '--steps',str(p['training_steps']),'--scene-base',str(p['training_scene_base']+i*p['training_scene_stride']),
                    '--out',str(args.out/'training'/name)]))
    batch(jobs,'training')
    evaluations=[]
    for scene in range(p['validation_scene_base'],p['validation_scene_base']+p['validation_scenes']):
        for regime in p['regimes']:
            for variant in p['variants']:
                for seed in p['training_seeds']:
                    name=f'{regime}_{variant}_{seed}_{scene}'
                    checkpoint=args.out/'training'/f'{regime}_{variant}_{seed}'/f"policy_{p['training_steps']}.pt"
                    evaluations.append(job(name,['evaluate','--regime',regime,'--variant',variant,'--checkpoint',str(checkpoint),
                        '--scene',str(scene),'--out',str(args.out/'results'/f'{name}.jsonl')]))
    batch(evaluations,'evaluation')
    summarize(args.out)
    atomic_json(args.out/'status.json',dict(phase='completed',training_runs=len(jobs),evaluations=len(evaluations),
        attempts=len(attempts),confirmation_accessed=False))


if __name__=='__main__':main()
