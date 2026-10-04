"""Summarize all EXP0053 attempts and paired results without seed selection."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.tpg_episode import PROTOCOL,tpg_hashes,ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--preflight',type=Path,required=True)
    args=parser.parse_args();root=args.root
    manifest=json.loads((root/'manifest.json').read_text());p=manifest['protocol']
    preflight_manifest=json.loads((args.preflight/'manifest.json').read_text())
    preflight_status=json.loads((args.preflight/'status.json').read_text())
    identity_file=args.preflight/'untrained_identity.json'
    preflight_passed=bool(preflight_status['phase']=='completed' and identity_file.exists()
        and json.loads(identity_file.read_text())['passed']
        and preflight_manifest['source_hashes']==manifest['source_hashes'])
    runtime=json.loads((root/'runtime_context.json').read_text()) if (root/'runtime_context.json').exists() else {}
    preflight_runtime=json.loads((args.preflight/'runtime_context.json').read_text()) if (args.preflight/'runtime_context.json').exists() else {}
    runtime_matches=all(runtime.get(k)==preflight_runtime.get(k) for k in ('NPY_DISABLE_CPU_FEATURES','numpy','torch','python'))
    attempts=[json.loads(line) for line in (root/'attempts.jsonl').read_text().splitlines()]
    rows=[]
    for file in sorted((root/'results').glob('*.jsonl')):
        for line in file.read_text().splitlines():
            row=json.loads(line);row['artifact']=str(file.relative_to(ROOT) if file.is_absolute() else file)
            name=file.stem
            if name.startswith('balanced_1s_'):arm='balanced_1s'
            elif name.startswith('memory_n1_'):arm='memory_n1'
            elif name.startswith('untrained_'):arm='untrained'
            elif name.startswith('tpg_'):arm='tpg'
            else:arm=row['variant']
            row['arm']=arm;rows.append(row)
    keys=['removal','removal_auc_180','wall_contact_s','spacing_violation_pair_s','scheduler_calls','target_reallocations','wait_agent_s']
    by_arm=defaultdict(list);by_scene=defaultdict(list)
    for row in rows:
        by_arm[row['arm']].append(row);by_scene[row['scene_seed']].append(row)
    expected_scenes=set(range(p['validation_scene_base'],p['validation_scene_base']+p['validation_scenes']))
    expected_per_arm={**{v:len(p['training_seeds'])*p['validation_scenes'] for v in p['variants']},
                      **{a:p['validation_scenes'] for a in ('tpg','untrained','balanced_1s','memory_n1')}}
    missing={arm:count-len(by_arm[arm]) for arm,count in expected_per_arm.items() if len(by_arm[arm])!=count}
    pairs=[]
    for scene,group in sorted(by_scene.items()):
        n3=[r for r in group if r['clusters']==3]
        hashes={r['actual_initial_snapshot_hash'] for r in n3}
        masks={r['scenario_hash'] for r in n3}
        baseline=next((r for r in group if r['arm']=='tpg'),None)
        untrained=next((r for r in group if r['arm']=='untrained'),None)
        pairs.append(dict(scene=scene,initial_match=len(hashes)==1,scenario_match=len(masks)==1,
            untrained_final_match=bool(baseline and untrained and baseline['final_state_hash']==untrained['final_state_hash']),
            actual_rows=len(group)))
    checkpoint_audit=[]
    for status_file in sorted((root/'training').glob('*/status.json')):
        status=json.loads(status_file.read_text());checkpoint=Path(status.get('checkpoint',''))
        checkpoint_audit.append(dict(run=status_file.parent.name,phase=status['phase'],
            hash_matches=checkpoint.is_file() and hashlib.sha256(checkpoint.read_bytes()).hexdigest()==status.get('checkpoint_sha256'),
            model_l2_change=status.get('model_l2_change'),high_samples=status.get('high_samples'),low_samples=status.get('low_samples')))
    source_ok=manifest['source_hashes']==tpg_hashes()
    snapshot_ok=all(hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest()==digest for name,digest in manifest['source_hashes'].items())
    failed=[a for a in attempts if a['returncode']!=0 or not a['source_unchanged']]
    gate=bool(preflight_passed and runtime_matches and not missing and not failed and source_ok and snapshot_ok and set(by_scene)==expected_scenes
        and all(x['initial_match'] and x['scenario_match'] and x['untrained_final_match'] for x in pairs)
        and len(checkpoint_audit)==9 and all(x['phase']=='completed' and x['hash_matches'] for x in checkpoint_audit))
    summaries={}
    for arm,group in by_arm.items():
        # Per-scene means avoid treating training seed repeats as independent scenes.
        scene_groups=defaultdict(list)
        for r in group:scene_groups[r['scene_seed']].append(r)
        means={}
        for key in keys:
            per_scene=[]
            for scene_group in scene_groups.values():
                vals=[r[key] if key in r else r.get('macro_steps') if key=='scheduler_calls' else None for r in scene_group]
                if all(v is not None for v in vals):per_scene.append(float(np.mean(vals)))
            means[key]=float(np.mean(per_scene)) if per_scene else None
        summaries[arm]=dict(attempts=len(group),independent_scenes=len(scene_groups),
            safe_successes=sum(bool(r['cluster_safe_success']) for r in group),**means)
    changes={}
    for variant in p['variants']:
        for seed in p['training_seeds']:
            count=0
            for row in by_arm[variant]:
                if row['training_seed']!=seed:continue
                baseline=next(r for r in by_scene[row['scene_seed']] if r['arm']=='tpg')
                count+=row['final_state_hash']!=baseline['final_state_hash']
            changes[f'{variant}_{seed}']=int(count)
    audit=dict(experiment=p['experiment'],valid_comparison_gate=gate,missing=missing,failed_attempts=failed,
        preflight_passed=preflight_passed,runtime_matches_preflight=runtime_matches,runtime_context=runtime,
        source_matches=source_ok,snapshot_matches=snapshot_ok,initial_pair_checks=pairs,
        checkpoint_checks=checkpoint_audit,confirmation_accessed=manifest['confirmation_accessed'],
        scheduled_attempts=len(attempts),evaluations=len(rows),training_control_steps=9*p['pilot_steps_per_seed'],
        policy_outcome_changed_scenes_vs_tpg=changes,summaries=summaries,
        interpretation='Four independent development scenes only; no statistical superiority or hardware safety claim.')
    (root/'AUDIT.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    lines=['# EXP0053 测量 TPG 双层学习开发试验','',
        f"配对/来源审计：{'通过' if gate else '未通过，不允许有效排名'}。固定 {p['validation_scenes']} 个独立开发场景；9 个训练运行，各 {p['pilot_steps_per_seed']} 个物理步。",
        f"运行时变体：{runtime.get('runtime_variant','default')}。默认运行时的原生失败保留，当前变体不能证明根因已修复。",
        '学习行先在同一场景平均三个训练种子，再对场景平均。种子重复不增加独立场景数量。','',
        '| 方法 | 清除比例 | AUC/时域 | 壁面接触（集群·秒） | 间距违规（对·秒） | 高层调用/回合 | 完全安全成功/运行 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('balanced_1s','tpg','untrained','high','low','both','memory_n1'):
        if arm not in summaries:continue
        s=summaries[arm]
        lines.append(f"| {arm} | {s['removal']:.2%} | {s['removal_auc_180']:.2%} | {s['wall_contact_s']:.3f} | {s['spacing_violation_pair_s']:.6f} | {s['scheduler_calls']:.2f} | {s['safe_successes']}/{s['attempts']} |")
    lines+=['','## 每个训练种子（全部保留）','',
        '| 变体 | 种子 | 清除比例 | 壁面接触 | 间距违规 | 相对 TPG 最终状态改变场景数 |',
        '|---|---:|---:|---:|---:|---:|']
    for variant in p['variants']:
        for seed in p['training_seeds']:
            group=[r for r in by_arm[variant] if r['training_seed']==seed]
            if not group:continue
            means=[np.mean([r[k] for r in group]) for k in ('removal','wall_contact_s','spacing_violation_pair_s')]
            lines.append(f'| {variant} | {seed} | {means[0]:.2%} | {means[1]:.3f} | {means[2]:.6f} | {changes[f"{variant}_{seed}"]}/{len(group)} |')
    lines+=['','## 解读边界','',
        '- 这是开发性小预算试验，不是确认结果。没有选择最佳种子，也没有放宽间距或壁面接触定义。',
        '- 高层只学习事件时的可行通行排列；事件触发规则本身仍是手工规则。底层学习时序残差候选，并采用下一步测量速度辅助损失。',
        '- 初版图直接控制碰壁严重，已保留预检负结果；最终版保留原测量记忆控制作名义候选。图工程/基线修正不是学习收益。',
        '- 未知全局路程保持未知。当前图与 TPG 是局部原型，不保证全局可达、无物理死锁、磁场独立或实际绝对间距安全。',
        '- 高层调用下降不代表底层网络或路线刷新也降频；两者仍以物理控制周期运行。',
        '- 原 1 s 强启发式是保留自身机制的额外比较；同图同控制候选的学习消融应与 tpg/untrained 比。',
        '- 未打开确认池。旧间歇性原生崩溃未被本研究证明修复。','',
        '逐场景结果见 `results/`，完整请求与返回码见 `attempts.jsonl`，原始代码见 `source_snapshot/`，审核见 `AUDIT.json`。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit,indent=2))
    if not gate:raise SystemExit(2)


if __name__=='__main__':main()
