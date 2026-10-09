"""Paired training-anatomy development screen of an inference-only intervention."""
import argparse
import json
from pathlib import Path
import numpy as np

FIELDS=('anatomy','clusters','seed','flow_inlet_mm_s','latency_steps','variation')
def key(r): return tuple(r[f] for f in FIELDS)
def metric(r,f):
    if f=='T90_300': return r['t90_s'] if r['t90_s'] is not None else 300.
    return r[f]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    candidate=[json.loads(x) for x in (a.out/'episodes.jsonl').read_text().splitlines()]
    original={key(r):r for r in map(json.loads,Path('research/validation/EXP0073_LONG1M_20261009/flow_aux_s7101/episodes.jsonl').read_text().splitlines())}
    stats=[]; contrasts=[]; lines=['# Lateral residual pilot (training anatomies only)','',
        'Same seed-7101 checkpoint; inference action-mapping intervention, not retraining or a new learned module. Original Ours uses privileged flow supervision during its earlier training.',
        'No held-out anatomy is used for candidate selection. All original metrics are unchanged.', '',
        '| Condition | Method | n | Full % | Strict % | Removal % | AUC | T90_300 s | Wall s | Lost total | Spacing pair-s |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fs=('task_success','strict_success','removal','removal_auc','T90_300','wall_contact_s','lost','spacing_violation_pair_s')
    for panel in sorted(set(r['panel'] for r in candidate)):
        new=sorted([r for r in candidate if r['panel']==panel],key=key);old=[original[key(r)] for r in new]
        assert all(x['scenario_hash']==y['scenario_hash'] for x,y in zip(new,old))
        for name,rr in [('Ours original',old),('Candidate',new)]:
            m={f:float(np.mean([metric(r,f) for r in rr])) for f in fs}
            stats.append(dict(panel=panel,method=name,n=len(rr),metrics=m))
            lines.append(f'| {panel} | {name} | {len(rr)} | {100*m["task_success"]:.1f} | {100*m["strict_success"]:.1f} | {100*m["removal"]:.1f} | {m["removal_auc"]:.3f} | {m["T90_300"]:.1f} | {m["wall_contact_s"]:.3f} | {sum(r["lost"] for r in rr)} | {m["spacing_violation_pair_s"]:.3f} |')
        names=sorted(set(r['anatomy'] for r in new));sums=[];counts=[]
        for an in names:
            pairs=[(x,y) for x,y in zip(new,old) if x['anatomy']==an];counts.append(len(pairs))
            sums.append([sum(metric(x,f)-metric(y,f) for x,y in pairs) for f in fs])
        sums=np.array(sums);counts=np.array(counts);rng=np.random.default_rng(7501)
        idx=rng.integers(0,len(names),(5000,len(names)));dist=sums[idx].sum(1)/counts[idx].sum(1)[:,None]
        contrasts.append(dict(panel=panel,difference=dict(zip(fs,(sums.sum(0)/counts.sum()).tolist())),
                              anatomy_block_ci95=dict(zip(fs,np.quantile(dist,[.025,.975],axis=0).T.tolist()))))
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (a.out/'aggregate.json').write_text(json.dumps(dict(stats=stats,contrasts=contrasts),indent=2))

if __name__=='__main__':main()
