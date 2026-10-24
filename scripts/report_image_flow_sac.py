"""Summarize complete paired scene records without hiding unsuccessful episodes."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def key(r):return tuple(r[k] for k in ('cell','anatomy','clusters','seed','flow_inlet_mm_s','latency_steps','variation_s'))
def label(method):
    if method.startswith('sac'):
        kind,_,path=method.partition('=');p=Path(path)
        return ('' if kind=='sac' else kind+':')+p.parent.name+'/'+p.name
    return method


def metrics(rows):
    def mean(k,default=None):
        values=[r.get(k,default) for r in rows];values=[v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    result=dict(n=len(rows),strict_success=mean('strict_success'),removal=mean('removal'),
        wall_contact_s=mean('wall_contact_s'),T90_300=float(np.mean([r['t90_s'] if r['t90_s'] is not None else 300. for r in rows])),
        removal_auc=mean('removal_auc'),lost=mean('lost'),spacing_pair_s=mean('spacing_violation_pair_s'),
        route_cross_track_rmse_mm=mean('route_cross_track_rmse_mm'),near_target_mean_error_mm=mean('near_target_mean_error_mm'),
        near_target_contact_fraction=mean('near_target_contact_fraction'),image_position_rmse_mm=mean('image_position_rmse_mm'),
        compensated_position_rmse_mm=mean('compensated_position_rmse_mm'),actor_changed_fraction=mean('actor_changed_fraction'),
        wall_guard_changed_fraction=mean('wall_guard_changed_fraction'),
        max_continuous_wall_contact_s=mean('max_continuous_wall_contact_s'),
        wall_ge_1s=mean('wall_ge_1s'),wall_ge_5s=mean('wall_ge_5s'),
        tpg_hold_s=mean('tpg_hold_s'),hold_wall_robot_s=mean('hold_wall_robot_s'))
    audited=[r['rl_command_audit'] for r in rows if 'rl_command_audit' in r]
    if audited:
        total={k:sum(a[k] for a in audited) for k in audited[0] if k!='executed_max'}
        result['command_attribution_steps']=total['steps']
        for prefix in ('raw','guarded','executed'):
            result['rl_'+prefix+'_mean_delta']=total[prefix+'_l1']/max(total['steps'],1)
            result['rl_'+prefix+'_nontrivial_fraction']=total[prefix+'_nontrivial']/max(total['steps'],1)
        result['rl_delta_survival_ratio']=total['executed_l1']/max(total['raw_l1'],1e-12)
        result['rl_executed_max_delta']=max(a['executed_max'] for a in audited)
    return result


def report(files,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);groups=defaultdict(dict)
    for file in files:
        for l in Path(file).open():
            row=json.loads(l);groups[label(row['method'])][key(row)]=row
    results={name:metrics(list(rows.values())) for name,rows in groups.items()};pairs={};rng=np.random.default_rng(95)
    references=['parent']+[name for name in groups if name.endswith('/initial.pt')]+['old_ppo']
    for reference in references:
        if reference not in groups:continue
        for name,rows in groups.items():
            if name==reference:continue
            shared=sorted(rows.keys()&groups[reference].keys(),key=str)
            if not shared:continue
            paired=[(rows[k],groups[reference][k]) for k in shared]
            mismatch=sum(a['scenario_hash']!=b['scenario_hash'] for a,b in paired)
            if mismatch:raise RuntimeError(f'{name} vs {reference}: mismatched scene hashes')
            deltas=np.array([float(a['strict_success'])-float(b['strict_success']) for a,b in paired])
            bs=deltas[rng.integers(0,len(paired),(4000,len(paired)))].mean(1)*100
            pairs[name+' vs '+reference]=dict(n=len(paired),strict_delta_pp=float(deltas.mean()*100),
                CI95_pp=np.percentile(bs,[2.5,97.5]).tolist(),wins=int((deltas>0).sum()),losses=int((deltas<0).sum()),
                removal_delta_pp=float(np.mean([a['removal']-b['removal'] for a,b in paired])*100),
                wall_delta_robot_s=float(np.mean([a['wall_contact_s']-b['wall_contact_s'] for a,b in paired])),
                scenario_hash_mismatches=mismatch)
    by_condition=[];by_topology=[]
    for name,rows in groups.items():
        grouped=defaultdict(list)
        for r in rows.values():grouped[r['cell'].split('|')[0]].append(r)
        for cell,rs in grouped.items():by_condition.append(dict(method=name,condition=cell,**metrics(rs)))
        split=defaultdict(list)
        for r in rows.values():split[r['cell'].split('|')[-1]].append(r)
        for topology,rs in split.items():by_topology.append(dict(method=name,topology=topology,**metrics(rs)))
    summary=dict(results=results,pairs=pairs,by_condition=by_condition,by_topology=by_topology,sources=[str(f) for f in files],development_only=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    with (out/'by_condition.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(k for r in by_condition for k in r))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(by_condition)
    with (out/'by_topology.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in by_topology for k in r)));w.writeheader();w.writerows(by_topology)
    with (out/'all_scenes.csv').open('w',newline='') as f:
        fields=['method','cell','anatomy','clusters','seed','scenario_hash','strict_success','removal','wall_contact_s','t90_s',
                'route_cross_track_rmse_mm','near_target_mean_error_mm','near_target_contact_fraction','termination']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for name,rs in groups.items():
            for r in rs.values():w.writerow({**r,'method':name})
    lines=['# 图像/运动历史 SAC 的实际配对评估','',
        'parent 为原最高成功率组合；old_ppo 为前一版类别 PPO；initial.pt 为同一新控制器的未训练 actor。所有场景含失败回合。此处使用开发评估池，不是封存测试。','',
        '| 方法 | n | Strict % | 清除 % | T90_300 s | 壁 robot·s | 路线偏离 RMSE mm | 末端平均误差 mm |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in sorted(results.items()):
        fmt=lambda v:'—' if v is None else f'{v:.3f}'
        lines.append(f"| {name} | {r['n']} | {100*r['strict_success']:.1f} | {100*r['removal']:.1f} | {r['T90_300']:.1f} | {r['wall_contact_s']:.2f} | {fmt(r['route_cross_track_rmse_mm'])} | {fmt(r['near_target_mean_error_mm'])} |")
    lines+=['','## 配对变化','']
    for name,p in pairs.items():
        lo,hi=p['CI95_pp'];lines.append(f"- {name}：n={p['n']}，Strict {p['strict_delta_pp']:+.2f} pp，95% 场景自举 CI [{lo:+.2f}, {hi:+.2f}]；清除 {p['removal_delta_pp']:+.2f} pp；壁接触 {p['wall_delta_robot_s']:+.3f} robot·s；场景哈希不一致 {p['scenario_hash_mismatches']}。")
    lines+=['','分血流/延迟条件见 by_condition.csv。末端误差只在真实位置距离当前目标 1 mm 内统计，必须结合接触时间、清除率和 T90 解读，不能单独用它挑选结果。原位等待、控制几何和观测变化的收益与 RL 参数更新的收益分别归因；单种子实验不能作为训练稳定性证明。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(results,ensure_ascii=False,indent=2));return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episodes',nargs='+',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();report(a.episodes,a.out)
