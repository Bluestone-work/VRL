"""Audit and summarize EXP0059 high-level isolation."""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scripts.run_high_isolation_eval import source_hashes
from scripts.run_option_learning import atomic_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args();root=args.root
    m=json.loads((root/'manifest.json').read_text());p=m['protocol'];attempts=[json.loads(x) for x in (root/'attempts.jsonl').read_text().splitlines()]
    rows=[];groups=defaultdict(list);scenes=defaultdict(list)
    for f in sorted((root/'results').glob('*.jsonl')):
        r=json.loads(f.read_text());rows.append(r);groups[(r['high_variant'],r['low_variant'])].append(r);scenes[r['scene_seed']].append(r)
    expected=len(p['high_variants'])*len(p['low_variants'])*len(p['training_seeds'])*p['validation_scenes']
    pair=[]
    for scene,g in sorted(scenes.items()):pair.append(dict(scene=scene,initial=len({r['actual_initial_snapshot_hash'] for r in g})==1,scenario=len({r['scenario_hash'] for r in g})==1,rows=len(g)))
    checks=dict(source_matches=source_hashes()==m['source_hashes'],snapshot_sources=all(hashlib.sha256((root/'source_snapshot'/n).read_bytes()).hexdigest()==h for n,h in m['source_hashes'].items()),
                complete_unique_matrix=len(rows)==expected and len({(r['high_variant'],r['low_variant'],r['training_seed'],r['scene_seed']) for r in rows})==expected,
                all_attempts_passed=len(attempts)==expected and all(a['returncode']==0 and a['source_unchanged'] for a in attempts),
                paired=all(x['initial'] and x['scenario'] and x['rows']==len(p['high_variants'])*len(p['low_variants'])*len(p['training_seeds']) for x in pair),confirmation_accessed=all(not r.get('confirmation_accessed') for r in rows))
    metrics=('removal','removal_auc_180','wall_contact_s','spacing_violation_pair_s','cluster_safe_success','mean_requested_residual')
    summaries=[]
    for (high,low),g in sorted(groups.items()):
        s=dict(high_variant=high,low_variant=low,rows=len(g),independent_scenes=len({r['scene_seed'] for r in g}))
        for key in metrics:
            vals=defaultdict(list)
            for r in g:vals[r['scene_seed']].append(float(r[key]))
            s[key]=float(np.mean([np.mean(v) for v in vals.values()]))
        summaries.append(s)
    audit=dict(valid_comparison_gate=all(checks.values()),checks=checks,rows=len(rows),expected=expected,pairing=pair,summaries=summaries,confirmation_accessed=False)
    atomic_json(root/'AUDIT.json',audit)
    lines=['# EXP0059 高层隔离实验','',f"审计：{'通过' if audit['valid_comparison_gate'] else '失败，禁止排名'}；{len(rows)}/{expected}。",'', '| 高层 | 固定低层 | 清除率 | AUC | 壁面集群秒 | 间距对秒 | 安全成功 |', '|---|---|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"| {s['high_variant']} | {s['low_variant']} | {s['removal']:.2%} | {s['removal_auc_180']:.2%} | {s['wall_contact_s']:.3f} | {s['spacing_violation_pair_s']:.6f} | {int(s['cluster_safe_success']*s['rows'])}/{s['rows']} |")
    lines += ['', '解释：高层和低层权重来自EXP0058最终检查点，交叉加载只用于接口隔离诊断；没有重新训练。', '所有组合使用同一测量输入、事件触发、局部保留、连续残差变换和安全投影。结果不能替代端到端训练比较，也不能证明跨解剖泛化。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit,indent=2))
    if not audit['valid_comparison_gate']:raise SystemExit(2)


if __name__=='__main__':main()
