"""Independent safety/resource/outcome audit; adds no trajectories or success claims."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.run_option_learning import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root
    audit=json.loads((root/'AUDIT.json').read_text())
    if not audit['valid_comparison_gate']:raise ValueError('Matrix did not pass admission')
    manifest=json.loads((root/'manifest.json').read_text());p=manifest['protocol']
    is_followup='regimes' in p
    parent=Path(__file__).resolve().parents[1]/p['parent_study'] if is_followup else root
    rows=[]
    for path in sorted((root/'results').glob('*.jsonl')):
        row=json.loads(path.read_text())
        if not is_followup and row.get('training_steps',0) not in (0,p['training_steps']):continue
        if path.name.startswith('balanced_1s_'):arm='balanced_1s'
        elif path.name.startswith('memory_n1_'):arm='memory_n1'
        else:arm=row['variant']
        if is_followup:arm=row['reward_regime']+'/'+arm
        row['audit_arm']=arm;rows.append(row)
    if is_followup:
        for path in sorted((parent/'results').glob('*.jsonl')):
            for label in ('tpg','balanced_1s','memory_n1'):
                if path.name.startswith(label+'_'):
                    row=json.loads(path.read_text());row['audit_arm']=label;rows.append(row);break
    groups=defaultdict(list)
    for row in rows:groups[row['audit_arm']].append(row)
    failures={};outcome_changes={};metric_checks=[]
    for arm,group in sorted(groups.items()):
        counts=defaultdict(int)
        for row in group:
            flags=dict(incomplete_clearance=not row['task_success'],unconfirmed=not row['sensor_confirmed_all_cleared'],
                wall=row['wall_contact_s']>=1.,particle=row['particle_contact_s']>1e-12,
                robot_contact=row['robot_pair_contact_s']>1e-12,lost=row['lost_clusters']>0,
                spacing=not row['spacing_compliant'])
            for key,value in flags.items():counts[key]+=int(value)
            safe=not any(flags.values())
            metric_checks.append(dict(arm=arm,scene=row['scene_seed'],seed=row.get('training_seed'),
                independent_safe_matches=bool(safe)==bool(row['cluster_safe_success'])))
        failures[arm]=dict(rows=len(group),safe_successes=sum(r['cluster_safe_success'] for r in group),**counts)
        if arm not in ('tpg','balanced_1s','memory_n1'):
            outcome_changes[arm]=dict(rows=len(group),changed_vs_rule=sum(r['final_state_hash']!=next(
                b for b in groups['tpg'] if b['scene_seed']==r['scene_seed'])['final_state_hash'] for r in group))
    resource_checks=[]
    for single in groups['memory_n1']:
        team=next(r for r in groups['tpg'] if r['scene_seed']==single['scene_seed'])
        a,b=single['actual_initial_snapshot'],team['actual_initial_snapshot']
        sa,sb=single['reset_info'],team['reset_info']
        resource_checks.append(dict(scene=single['scene_seed'],
            shared_scene=single['scenario_hash']==team['scenario_hash'],
            selected_start_same=np.array_equal(a['positions_mm'][:1],b['positions_mm'][:1]),
            same_particles=np.array_equal(a['positions_mm'][1:],b['positions_mm'][3:]),
            same_targets=np.array_equal(a['clot_positions_mm'],b['clot_positions_mm']),
            same_masses=np.array_equal(a['masses'],b['masses']),
            same_catalytic_proxy=sa['aggregate_lysis_capacity_mass_per_s']==sb['aggregate_lysis_capacity_mass_per_s']))
    passed=all(r['independent_safe_matches'] for r in metric_checks) and all(all(v for k,v in r.items() if k!='scene') for r in resource_checks)
    result=dict(passed=passed,source_audit_sha256=hashlib.sha256((root/'AUDIT.json').read_bytes()).hexdigest(),
        failure_causes=failures,learned_outcome_changes=outcome_changes,metric_checks=metric_checks,
        n1_resource_checks=resource_checks,n1_equivalence='aggregate catalytic-rate proxy only; not total material or magnetic power',
        independent_scenes=p['validation_scenes'],confirmation_accessed=False)
    atomic_json(root/'OUTCOME_AUDIT.json',result)
    lines=['# 独立结果复核','',f"安全定义及资源/初态复核：{'通过' if passed else '失败'}。",'',
        '| 方法 | 回合 | 安全成功 | 未清完 | 未确认 | 壁面失败 | 粒子接触 | 集群接触 | 集群丢失 | 间距失败 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,v in failures.items():
        lines.append('| '+arm+' | '+' | '.join(str(v.get(k,0)) for k in ('rows','safe_successes','incomplete_clearance','unconfirmed','wall','particle','robot_contact','lost','spacing'))+' |')
    lines+=['','失败原因可同时发生，不能相加当作失败回合数。N=1 的间距项不适用；仅匹配总催化率代理。',
        '学习后最终状态不同只说明闭环行为不同，不等于效能或安全改善。八个同结构开发布局不是跨解剖泛化或硬件结论。']
    (root/'OUTCOME_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=passed,failure_causes=failures,learned_outcome_changes=outcome_changes),indent=2))
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
