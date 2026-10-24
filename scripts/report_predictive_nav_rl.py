"""Paired PPO navigation results; retain initialization and no-RL controls."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def key(r):
    return tuple(r.get(k) for k in ('cell','anatomy','clusters','seed','flow_inlet_mm_s','latency_steps','variation_s'))

def group(r):
    if r['method']=='fac' and r['method_arg']=='A_TPG+ours':return 'parent_no_RL'
    p=Path(r.get('method_arg') or '')
    if r['method']=='rl_nav_off':return 'parent_no_RL'
    return f'{p.parent.name}/{p.name}'

def report(files,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);groups=defaultdict(dict)
    for file in files:
        for line in Path(file).open():
            r=json.loads(line)
            if r['method'] not in ('rl_nav','rl_nav_off') and not (r['method']=='fac' and r['method_arg']=='A_TPG+ours'):continue
            groups[group(r)][key(r)]=r
    results={};pairs={};rng=np.random.default_rng(94)
    for name,by in groups.items():
        rows=list(by.values());decisions=sum(r.get('rl_decisions',0) for r in rows)
        metric=lambda k:float(np.mean([r.get(k,0.) for r in rows]))
        results[name]=dict(n=len(rows),strict_success=metric('strict_success'),removal=metric('removal'),
                           wall_contact_s=metric('wall_contact_s'),T90_300=float(np.mean([r['t90_s'] if r['t90_s'] is not None else 300. for r in rows])),
                           spacing_pair_s=metric('spacing_violation_pair_s'),rl_decisions=decisions,
                           rl_output_changed_fraction=sum(r.get('rl_output_changed_decisions',0) for r in rows)/max(decisions,1),
                           rl_update_command_changed_fraction=sum(r.get('rl_update_command_changes',0) for r in rows)/max(decisions,1),
                           rl_command_changed_fraction=sum(r.get('rl_command_changed_decisions',0) for r in rows)/max(decisions,1))
        if name!='parent_no_RL' and 'parent_no_RL' in groups:
            keys=sorted(by.keys()&groups['parent_no_RL'].keys(),key=str)
            if keys:
                parent=groups['parent_no_RL'];delta=np.asarray([int(by[k]['strict_success'])-int(parent[k]['strict_success']) for k in keys])
                bs=delta[rng.integers(0,len(delta),(4000,len(delta)))].mean(1)
                hash_bad=sum(by[k]['scenario_hash']!=parent[k]['scenario_hash'] for k in keys)
                pairs[name]=dict(n=len(keys),strict_delta_pp=float(delta.mean()*100),CI95_pp=np.percentile(bs,[2.5,97.5]).tolist(),
                                 scenario_hash_mismatches=hash_bad,wins=int((delta>0).sum()),losses=int((delta<0).sum()))
                pairs[name]['CI95_pp']=[v*100 for v in pairs[name]['CI95_pp']]
    summary=dict(results=results,paired_to_parent=pairs,sources=[str(p) for p in files])
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    lines=['# PPO 局部导航的实际评估结果','',
           '策略由真实环境奖励的 PPO 更新。parent_no_RL 是关闭 RL 的原预测选择器；initial.pt 为未经过 PPO 更新的初始化策略。场景数量不足 252 的表属于调参/阶段结果，不能称为全场景结果。','',
           '| 方法 | 场景数 | Strict % | 清除 % | T90_300 s | 壁 robot·s | 间距 pair·s | RL 改变输出 % | 相对原方法 Strict pp [95% CI] |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,r in sorted(results.items()):
        p=pairs.get(name);change='—' if p is None else f"{p['strict_delta_pp']:+.1f} [{p['CI95_pp'][0]:+.1f}, {p['CI95_pp'][1]:+.1f}] (n={p['n']})"
        lines.append(f"| {name} | {r['n']} | {100*r['strict_success']:.1f} | {100*r['removal']:.1f} | {r['T90_300']:.1f} | {r['wall_contact_s']:.2f} | {r['spacing_pair_s']:.4f} | {100*r['rl_output_changed_fraction']:.2f} | {change} |")
    lines+=['','配对严格使用同一解剖、集群数、场景种子、流速、延迟和扰动，并核对 scenario_hash。失败也计入结果。置信区间反映当前场景集，不能把调参池置信区间外推为正式测试结论。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results,ensure_ascii=False,indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episodes',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args();report(a.episodes,a.out)
