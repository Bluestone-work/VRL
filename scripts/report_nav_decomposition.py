"""Aggregate paired nav decomposition episodes into CSV/Markdown/JSON."""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

METRICS = [
 ('relaxed_success','relaxed %',100),('task_success','raw/full %',100),('strict_success','strict %',100),
 ('removal','removal %',100),('removal_auc','AUC %',100),('t50_s','T50 s',1),('t90_s','T90 s',1),('t100_s','T100 s',1),
 ('wall_contact_s','wall robot·s',1),('wall_ge_1s','wall≥1s %',100),('wall_ge_5s','wall≥5s %',100),
 ('max_continuous_wall_contact_s','max wall s',1),('lost','lost/ep',1),('spacing_violation_pair_s','spacing pair·s',1),
 ('robot_pair_contact_s','robot pair·s',1),('coupling_exposure','coupling',1),('tpg_hold_s','TPG hold s',1),
 ('path_mm','path mm',1),('elapsed_s','elapsed s',1),('walltime_s','walltime s',1),('clots_cleared','clots cleared',1),('clots','clots',1),('min_pair_distance_mm','min pair mm',1)]

def mean(rows, key):
    x=[]
    for r in rows:
        v=r.get(key)
        if v is None: continue
        try: v=float(v)
        except: continue
        if math.isfinite(v): x.append(v)
    return float(np.mean(x)) if x else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); a=ap.parse_args()
    cells=defaultdict(list)
    for line in (a.root/'episodes.jsonl').read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        if r.get('error'): continue
        s=r.get('source_strength')
        if isinstance(s,(int,float)): s=f's{int(round(10*s))}'
        cells[(str(s),int(r['clusters']),r['variant'])].append(r)
    rows=[]
    for (s,n,v),rs in sorted(cells.items()):
        d={'strength':s,'clusters':n,'variant':v,'n':len(rs)}
        for k,_,_ in METRICS:
            if k in ('wall_ge_1s','wall_ge_5s'): d[k]=float(np.mean([bool(r.get(k,False)) for r in rs]))
            else: d[k]=mean(rs,k)
        rows.append(d)
    fields=['strength','clusters','variant','n']+[x[0] for x in METRICS]
    with (a.root/'aggregate.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    (a.root/'aggregate.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    def fmt(v,k):
        if v is None:return '—'
        scale=next(x[2] for x in METRICS if x[0]==k)
        if k in ('relaxed_success','task_success','strict_success','removal','removal_auc','wall_ge_1s','wall_ge_5s'): return f'{v*scale:.1f}'
        return f'{v:.2f}'
    key=['task_success','strict_success','removal','removal_auc','t90_s','wall_contact_s','wall_ge_1s','spacing_violation_pair_s','coupling_exposure','tpg_hold_s']
    lines=['# nav_tf_v3 组件归因配对消融','', '同一 V5 注册场景、同一 checkpoint、同一随机种子；`full` 保留 TPG、WallGuard 和 spacing shield。所有 23 个指标在 aggregate.csv/json 中提供。','']
    for s in sorted(set(r['strength'] for r in rows)):
        lines += [f'## {s}','']
        for n in (1,2,3):
            sub=[r for r in rows if r['strength']==s and r['clusters']==n]
            if not sub: continue
            lines += [f'### N={n}','', '| 变体 | n | '+' | '.join(next(x[1] for x in METRICS if x[0]==k) for k in key)+' |', '|---|---:|'+'---:|'*len(key)]
            for r in sub: lines.append('| '+r['variant']+' | '+str(r['n'])+' | '+' | '.join(fmt(r.get(k),k) for k in key)+' |')
            lines.append('')
    lines += ['## 变体定义','', '- `full`: 学习器 + TPG + WallGuard + spacing shield。','- `no_guard`: 保留学习器、TPG、spacing shield，关闭 WallGuard。','- `no_shield`: 保留学习器、TPG、WallGuard，关闭 spacing shield。','- `no_safety`: 同时关闭 WallGuard 和 spacing shield。','- `no_tpg`: 保留学习器和安全层，关闭 TPG；为避免完成集群转去帮助其他集群，采用 park fallback。','']
    refp=Path('research/figures/NAV_TF_V3_FULL_20261010/aggregate.csv')
    if refp.exists():
        ref=list(csv.DictReader(refp.open()))
        lines += ['## no_learning 参考','', '`classical_settle` 是同一 V5 协议中的 no-learning 规则参考：保留术前路线、TPG 和 WallGuard，由 Fixed/Settle 规则输出低层命令。它不是 PPO 学习器。','', '| strength | N | raw/full % | strict % | removal % | T90_300 s | wall robot·s | wall≥1s % |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in ref:
            if r.get('method')!='classical_settle': continue
            lines.append(f"| {r['strength']} | {r['clusters']} | {float(r['task_success'])*100:.1f} | {float(r['strict_success'])*100:.1f} | {float(r['removal'])*100:.2f} | {float(r['t90_300']):.2f} | {float(r['wall_contact_s']):.2f} | {float(r['wall_ge_1s'])*100:.1f} |")
        lines.append('')
    (a.root/'REPORT.md').write_text('\n'.join(lines))
    print(a.root/'REPORT.md')
if __name__=='__main__': main()
