"""Complete-matrix audit and paired, descriptive EXP0057 development results."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.run_clean_residual_eval import ROOT, source_hashes
from scripts.run_option_learning import atomic_json

METRICS = ('removal','removal_auc_180','wall_contact_s','spacing_violation_pair_s',
           'mean_requested_residual','low_nonrule_choices','high_nonrule_choices')


def analyze(root):
    manifest = json.loads((root/'manifest.json').read_text())
    p, source = manifest['protocol'], manifest['source_hashes']
    attempts = [json.loads(x) for x in (root/'attempts.jsonl').read_text().splitlines()]
    requested = [j for phase in ('preflight','evaluation')
                 for j in json.loads((root/f'{phase}_requested.json').read_text())]
    rows, groups, scenes = [], defaultdict(list), defaultdict(list)
    for file in sorted((root/'results').glob('*.jsonl')):
        if not file.stat().st_size:
            continue
        row = json.loads(file.read_text())
        arm = next((a for a in ('balanced_1s','memory_n1') if file.stem.startswith(a+'_')), row.get('variant'))
        row['arm'] = arm
        groups[arm].append(row); scenes[row['scene_seed']].append(row); rows.append(row)
    expected_keys = {(v,s,c) for v in p['variants'] for s in p['training_seeds']
                     for c in range(p['validation_scene_base'],p['validation_scene_base']+p['validation_scenes'])}
    expected_keys |= {(v,None,c) for v in p['baseline_controls']
                     for c in range(p['validation_scene_base'],p['validation_scene_base']+p['validation_scenes'])}
    keys = [(r['arm'],r.get('training_seed'),r['scene_seed']) for r in rows]
    pairing = [dict(scene=c, initial=len({r['actual_initial_snapshot_hash'] for r in group if r['clusters']==3})==1,
                    scenario=len({r['scenario_hash'] for r in group})==1) for c,group in sorted(scenes.items())]
    lineage = []
    for arm in p['variants']:
        for seed in p['training_seeds']:
            folder = ROOT/p['parent_study']/'training'/f'train_{arm}_{seed}'
            status = json.loads((folder/'status.json').read_text())
            item = next(x for x in status['checkpoints'] if x['steps']==p['training_steps'])
            digest = hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()
            subset = [r for r in groups[arm] if r['training_seed']==seed]
            lineage.append(dict(arm=arm, seed=seed, passed=status['phase']=='completed' and digest==item['sha256']
                and all(r['checkpoint_sha256']==digest and r['source_hashes']==source and r['runtime']==manifest['runtime']
                        and r['training_steps']==p['training_steps'] and r['weight_transfer_exact'] for r in subset),
                high_changed=status['model_l2_change']['high_actor']>0, low_changed=status['model_l2_change']['low_actor']>0))
    # Independently reconstruct the registered safety predicate from outcome fields.
    safety = []
    for r in rows:
        predicate = (r['task_success'] and r['sensor_confirmed_all_cleared']
            and r['wall_contact_s']<1. and r['particle_contact_s']<=1e-12 and r['robot_pair_contact_s']<=1e-12
            and r['lost_clusters']==0 and r['spacing_compliant'])
        safety.append(bool(predicate)==bool(r['cluster_safe_success']))
    checks = dict(source_matches=source==source_hashes(),
        snapshot_matches=all(hashlib.sha256((root/'source_snapshot'/n).read_bytes()).hexdigest()==h for n,h in source.items()),
        all_attempts_passed=all(a['returncode']==0 and a['source_unchanged'] for a in attempts),
        all_requested_once=sorted(x['name'] for x in attempts)==sorted(x['name'] for x in requested)
                           and len({x['name'] for x in attempts})==len(attempts),
        complete_unique_matrix=set(keys)==expected_keys and len(keys)==len(expected_keys),
        exact_initial_pairing=all(x['initial'] and x['scenario'] for x in pairing),
        preflight=json.loads((root/'preflight_audit.json').read_text())['passed'],
        runtime_tests=json.loads((root/'test_admission.json').read_text())['returncode']==0,
        checkpoint_lineage=all(x['passed'] and x['high_changed'] and x['low_changed'] for x in lineage),
        safety_reconstruction=all(safety),
        residual_bound=all(r.get('max_moving_residual',0)<=p['residual_radius']+1e-7 for r in rows),
        no_confirmation=all(not r.get('confirmation_accessed',False) and not r.get('sealed_test_used',False) for r in rows))
    gate = all(checks.values())
    summaries, per_seed, paired = [], [], []
    def summarize(arm, group):
        result = dict(arm=arm, rows=len(group), safe=sum(bool(r['cluster_safe_success']) for r in group))
        for metric in METRICS:
            per_scene = defaultdict(list)
            for r in group:
                if metric in r:
                    per_scene[r['scene_seed']].append(r[metric])
            result[metric] = float(np.mean([np.mean(v) for v in per_scene.values()])) if per_scene else None
        return result
    if gate:
        summaries = [summarize(a,g) for a,g in sorted(groups.items())]
        per_seed = [dict(seed=s, **summarize(a,[r for r in groups[a] if r['training_seed']==s]))
                    for a in p['variants'] for s in p['training_seeds']]
        for a,b in (('graph_anchor','r_mappo'),('graph_anchor','graph_ppo'),('mappo_anchor','r_mappo'),
                    ('graph_ppo','r_mappo'),('graph_anchor','mappo_anchor'),('graph_anchor','tpg'),('graph_anchor','balanced_1s')):
            differences = []
            for c,g in sorted(scenes.items()):
                differences.append(dict(scene=c, **{k:float(np.mean([r[k] for r in g if r['arm']==a])-
                    np.mean([r[k] for r in g if r['arm']==b])) for k in METRICS[:4]}))
            paired.append(dict(method=a, versus=b, differences=differences,
                mean_difference={k:float(np.mean([d[k] for d in differences])) for k in METRICS[:4]}))
    audit = dict(valid_comparison_gate=gate, checks=checks, evaluations=len(rows), expected=len(expected_keys),
        attempts=len(attempts), failed_attempts=[a for a in attempts if a['returncode']!=0], pairing=pairing,
        checkpoint_checks=lineage, summaries=summaries, per_seed=per_seed, paired=paired,
        independent_scenes=p['validation_scenes'], confirmation_accessed=False)
    atomic_json(root/'AUDIT.json',audit)
    lines = ['# EXP0057 保守残差 MARL 开发实验','',
        f'审计：{"通过" if gate else "未通过，禁止排名"}；{len(rows)}/{len(expected_keys)} 次评估。',
        '相同八个已使用开发场景、三个原训练种子；完整部署评估全部十二个冻结 EXP0056 最终权重，本轮没有新训练。所有方法在统一清洁运行时评估，非确认实验、非原运行时精确重放。',
        '四组共享有界残差、默认策略先验、安全奖励、观测和投影；2×2 消融仅改变图优先级和低层 KL 约束。','',
        '| 方法 | 清除率 | AUC | 壁面集群秒 | 间距对秒 | 平均请求残差 | 安全成功 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        residual = '—' if s['mean_requested_residual'] is None else f"{s['mean_requested_residual']:.5f}"
        lines.append(f"| {s['arm']} | {s['removal']:.2%} | {s['removal_auc_180']:.2%} | {s['wall_contact_s']:.3f} | {s['spacing_violation_pair_s']:.6f} | {residual} | {s['safe']}/{s['rows']} |")
    lines += ['', '完整每种子结果及逐场景配对差值见 AUDIT.json。',
        '动作索引 7/8 在正常行进时已是小幅残差，不能继续解释成停滞/退让。请求残差不等于投影后实际轨迹差异。',
        '策略只接收测量包；奖励及独立安全统计使用仿真真值。2 mm 是未标定欧式距离代理，不能证明磁场独立。',
        '未修改旧实验或补齐 EXP0055/56 崩溃行；旧矩阵仍失败，旧运行时偶发崩溃根因仍未解决。',
        'MAPPO 为共享局部 TPG 的测量通信适配；尚无 MAT、独立端到端规划器比较或跨解剖确认。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(valid_comparison_gate=gate, checks=checks, summaries=summaries),indent=2))
    if not gate:
        raise SystemExit(2)
    return audit


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root)
