"""Audit every requested MARL run before reporting paired development summaries."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.run_measured_marl import ROOT, source_hashes
from scripts.run_option_learning import atomic_json


def analyze(root):
    manifest = json.loads((root/'manifest.json').read_text()); p = manifest['protocol']
    source = manifest['source_hashes']
    attempts = [json.loads(x) for x in (root/'attempts.jsonl').read_text().splitlines()]
    requested = [j for phase in ('preflight','regressions','training','evaluation')
                 for j in json.loads((root/f'{phase}_requested.json').read_text())]
    request_names = [j['name'] for j in requested]
    completed_names = [j['name'] for j in attempts]
    rows, by_scene, groups = [], defaultdict(list), defaultdict(list)
    for file in sorted((root/'results').glob('*.jsonl')):
        row = json.loads(file.read_text())
        if file.stem.startswith('balanced_1s_'):
            arm = 'balanced_1s'
        elif file.stem.startswith('memory_n1_'):
            arm = 'memory_n1'
        else:
            arm = row['variant']
        row['arm'] = arm; row['artifact'] = str(file)
        step = row.get('training_steps', 0)
        row['checkpoint_steps'] = step
        rows.append(row); by_scene[row['scene_seed']].append(row)
        groups[arm, step].append(row)
    expected = p['validation_scenes']*(len(p['variants'])*len(p['training_seeds'])*len(p['checkpoints'])+3)
    checks = []
    for scene, group in sorted(by_scene.items()):
        n3 = [r for r in group if r['clusters']==3]
        checks.append(dict(scene=scene, same_initial=len({r['actual_initial_snapshot_hash'] for r in n3})==1,
            same_scenario=len({r['scenario_hash'] for r in group})==1,
            rows=len(group)))
    checkpoint_checks = []
    for variant in p['variants']:
        for seed in p['training_seeds']:
            folder = root/'training'/f'train_{variant}_{seed}'
            stat = json.loads((folder/'status.json').read_text())
            for item in stat['checkpoints']:
                path = Path(item['path'])
                checkpoint_checks.append(dict(variant=variant, seed=seed, steps=item['steps'],
                    completed=stat['phase']=='completed', match=hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']))
    failed = [x for x in attempts if x['returncode'] != 0 or not x['source_unchanged']]
    snapshot_ok = all(hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest()==digest for name,digest in source.items())
    preflight = json.loads((root/'preflight_audit.json').read_text())['passed']
    regression = json.loads((root/'regression_audit.json').read_text())['passed']
    correct_scenes = set(by_scene)==set(range(p['validation_scene_base'],p['validation_scene_base']+p['validation_scenes']))
    count_ok = len(rows)==expected and all(len(groups[v,s])==p['validation_scenes']*len(p['training_seeds'])
        for v in p['variants'] for s in p['checkpoints']) and all(len(groups[v,0])==p['validation_scenes'] for v in p['baseline_controls'])
    gate = bool(source==source_hashes() and snapshot_ok and not failed and preflight and regression and count_ok and correct_scenes
        and sorted(request_names)==sorted(completed_names) and len(set(completed_names))==len(completed_names)
        and all(c['same_initial'] and c['same_scenario'] for c in checks)
        and len(checkpoint_checks)==len(p['variants'])*len(p['training_seeds'])*len(p['checkpoints'])
        and all(c['completed'] and c['match'] for c in checkpoint_checks))
    metrics = ('removal','removal_auc_180','wall_contact_s','spacing_violation_pair_s','scheduler_calls','low_nonrule_choices','high_nonrule_choices')
    summaries = []
    for (arm, step), group in sorted(groups.items()):
        result = dict(arm=arm, steps=step, rows=len(group), independent_scenes=len({r['scene_seed'] for r in group}),
            safe_successes=sum(bool(r['cluster_safe_success']) for r in group))
        for key in metrics:
            values = defaultdict(list)
            for r in group:
                val = r.get(key, r.get('macro_steps') if key=='scheduler_calls' else None)
                if val is not None:
                    values[r['scene_seed']].append(val)
            result[key] = float(np.mean([np.mean(v) for v in values.values()])) if values else None
        summaries.append(result)
    per_seed = []
    paired = []
    for variant in p['variants']:
        for step in p['checkpoints']:
            for seed in p['training_seeds']:
                part = [r for r in groups[variant,step] if r['training_seed']==seed]
                per_seed.append(dict(variant=variant, steps=step, seed=seed, safe=sum(r['cluster_safe_success'] for r in part),
                    **{key:float(np.mean([r[key] for r in part])) for key in ('removal','removal_auc_180','wall_contact_s','spacing_violation_pair_s','low_nonrule_choices','high_nonrule_choices')}))
    for comparison in ('r_mappo','r_ippo','vctpg_no_ac','tpg','balanced_1s'):
        for scene in sorted(by_scene):
            a = [r for r in by_scene[scene] if r['arm']=='vctpg_ac' and r['checkpoint_steps']==p['training_steps']]
            b = [r for r in by_scene[scene] if r['arm']==comparison and r['checkpoint_steps'] in (0,p['training_steps'])]
            paired.append(dict(scene=scene, versus=comparison, **{key:float(np.mean([r[key] for r in a])-np.mean([r[key] for r in b]))
                for key in ('removal','removal_auc_180','wall_contact_s','spacing_violation_pair_s')}))
    audit = dict(valid_comparison_gate=gate, source_matches=source==source_hashes(), snapshot_matches=snapshot_ok,
        requested_attempts=len(requested), completed_attempts=len(attempts), evaluations=len(rows), failed_attempts=failed,
        preflight_passed=preflight, regressions_passed=regression, pairing=checks, checkpoints=checkpoint_checks,
        summaries=summaries, per_seed=per_seed, paired_scene_differences=paired,
        confirmation_accessed=False, scope=p['scope'], primary_checkpoint_steps=p['training_steps'])
    atomic_json(root/'AUDIT.json',audit)
    lines=['# EXP0054：测量条件下多智能体学习开发对比','',
        f"审计：{'通过' if gate else '未通过，禁止排名'}；{len(rows)}/{expected} 次评估，{p['validation_scenes']} 个独立固定场景。",
        'MAPPO/IPPO 是共享 TPG 的通信测量适配，尚不是独立规划器系统对比，也不是原作者代码复现。',
        f"主开发检查点为 {p['training_steps']} 步；8192 步全部保留，仅作预登记学习过程诊断。",
        '三个训练种子先按场景平均，再汇总；没有将种子重复视为额外独立场景。','',
        '| 方法 | 步数 | 清除比例 | AUC | 壁面集群秒 | 间距对秒 | 高层调用 | 低层偏离规则/回合 | 安全成功 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in sorted(summaries,key=lambda s:(s['steps']!=p['training_steps'],s['steps'],s['arm'])):
        difference='N/A' if s['low_nonrule_choices'] is None else f"{s['low_nonrule_choices']:.1f}"
        spacing='N/A' if s['arm']=='memory_n1' else f"{s['spacing_violation_pair_s']:.6f}"
        lines.append(f"| {s['arm']} | {s['steps']} | {s['removal']:.2%} | {s['removal_auc_180']:.2%} | {s['wall_contact_s']:.3f} | {spacing} | {s['scheduler_calls']:.1f} | {difference} | {s['safe_successes']}/{s['rows']} |")
    lines += ['', '## 全部训练种子','', '| 方法 | 步数 | 种子 | 清除 | AUC | 壁面集群秒 | 间距对秒 | 安全成功 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in per_seed:
        lines.append(f"| {r['variant']} | {r['steps']} | {r['seed']} | {r['removal']:.2%} | {r['removal_auc_180']:.2%} | {r['wall_contact_s']:.3f} | {r['spacing_violation_pair_s']:.6f} | {r['safe']}/{p['validation_scenes']} |")
    lines += ['', '## 结论边界','',
        '- 不根据最优种子或早期检查点选赢家。预算是开发预算，尚无公平调参扫描或收敛证明。',
        '- 图、事件触发、任务分配、观测和安全投影是共同工程；不能算成学习收益。',
        '- 动作条件预测对象为下一次滤波测量速度，不是真实动力学或硬件标定。',
        '- 安全成功定义保持原协议，允许壁面累计小于 1 集群秒；2 mm 仍是未标定代理。',
        '- N=1 仅匹配总催化率代理，不证明材料/磁功率等资源。未打开任何确认池。',
        '- 原运行时失败保留；本次统一使用已准入的隔离 NumPy 构建。',
        '- 更高破栓率、较低壁面接触或动作不同不能单独证明整体安全有效性优势。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=gate,evaluations=len(rows),summaries=summaries),indent=2))
    if not gate:
        raise SystemExit(2)
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
