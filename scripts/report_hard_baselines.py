"""Report common-scene comparisons without treating repeated N as independent."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.evaluate_hard_baselines import PANELS

FIELDS = ('split','suite','anatomy','clusters','seed','flow_inlet_mm_s','latency_steps','variation')
METRICS = ('task_success','strict_success','removal','removal_auc','T90_300',
           'wall_contact_s','lost','spacing_violation_pair_s')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    rows=[json.loads(s) for s in (a.out/'episodes.jsonl').read_text().splitlines()]
    for method in ('nav_tf_v3','flow_aux'):
        path=Path('research/validation/EXP0073_LONG1M_20261009')/(method+'_s7101')/'episodes.jsonl'
        for r in map(json.loads,path.read_text().splitlines()):
            if 'error' in r: continue
            for panel, condition in PANELS.items():
                if (r['flow_inlet_mm_s'],r['latency_steps'],r['variation']) == condition:
                    r.update(panel=panel,method=method)
                    r['T90_300']=r['t90_s'] if r['t90_s'] is not None else 300.
                    rows.append(r)
    methods=sorted(set(r['method'] for r in rows)); indexed={}
    for r in rows:
        key=(r['method'],tuple(r[f] for f in FIELDS))
        if key in indexed: raise ValueError(f'Duplicate {key}')
        indexed[key]=r
    output=[]; contrasts=[]; strata=[]; lines=['# EXP0073 paired hard-condition screening','',
        'Ours (flow_aux) denotes the evaluated Transformer with flow auxiliary training; nav_tf_v3 is the original Transformer reference. Ours here does not include belief or rolling-TPG.',
        'Single learned training seed 7101. Classical defaults unchanged. PAC-NMPC and STPG are repository adaptations, not exact author reproductions.',
        'Infrastructure failures remain in failures.jsonl. Tables use only common completed scenes; missing cases are not counted as successes or task failures.',
        'All episodes contribute to AUC and T90_300 (unreached = 300 s). Conditional T90 is recorded separately. No sealed test access.',
        'Variation strength changes flow/pulsatility, actuation, lag, wall effects and lysis rate jointly, not response gain alone. Flow 0.1 is in the EXP0073 training support.', '']
    for panel in PANELS:
        sets={m:{tuple(r[f] for f in FIELDS) for r in rows if r['panel']==panel and r['method']==m} for m in methods}
        common=set.intersection(*sets.values())
        lines += [f'## {panel}: flow/delay/variation = {PANELS[panel]}', '',
                  f'Common completed cases: {len(common)}/42. Available by method: '+str({m:len(s) for m,s in sets.items()}), '',
                  '| Method | n | Full % | Strict % | Removal % | AUC | T90_300 s | T90 reached | conditional T90 s | Wall s | Lost total | Spacing pair-s |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for m in methods:
            rr=[indexed[m,k] for k in sorted(common)]
            if not rr: continue
            for k in common:
                hashes={indexed[x,k]['scenario_hash'] for x in methods}
                if len(hashes)!=1: raise ValueError('Scenario hash mismatch')
            avg={v:float(np.mean([r[v] for r in rr])) for v in METRICS}
            reached=[r['t90_s'] for r in rr if r['t90_s'] is not None]
            output.append(dict(panel=panel,method=m,n=len(rr),metrics=avg,t90_reached=len(reached),
                               conditional_t90=float(np.mean(reached)) if reached else None))
            for field in ('split','clusters','anatomy'):
                for value in sorted(set(r[field] for r in rr)):
                    sub=[r for r in rr if r[field]==value]
                    strata.append(dict(panel=panel,method=m,field=field,value=value,n=len(sub),
                        metrics={v:float(np.mean([r[v] for r in sub])) for v in METRICS}))
            cond=f'{np.mean(reached):.1f}' if reached else 'NA'
            label='Ours (flow_aux)' if m=='flow_aux' else m
            lines.append(f'| {label} | {len(rr)} | {100*avg["task_success"]:.1f} | {100*avg["strict_success"]:.1f} | {100*avg["removal"]:.1f} | {avg["removal_auc"]:.3f} | {avg["T90_300"]:.1f} | {len(reached)}/{len(rr)} | {cond} | {avg["wall_contact_s"]:.2f} | {sum(r["lost"] for r in rr)} | {avg["spacing_violation_pair_s"]:.3f} |')
        # Pairwise contrasts preserve all N in each anatomy cluster. Exploratory, not training-seed uncertainty.
        for base in ('settle','adaptive_settle','no_settle','stpg','pac_nmpc'):
            for learner in ('nav_tf_v3','flow_aux'):
                kk=sorted(sets.get(base,set())&sets.get(learner,set()))
                if not kk: continue
                anatomies=sorted(set(k[2] for k in kk)); sums=[]; counts=[]
                for an in anatomies:
                    sub=[k for k in kk if k[2]==an]; counts.append(len(sub))
                    sums.append([sum(indexed[learner,k][v]-indexed[base,k][v] for k in sub) for v in METRICS])
                sums=np.array(sums); counts=np.array(counts); rng=np.random.default_rng(7301)
                draw=rng.integers(0,len(counts),(5000,len(counts)))
                boots=sums[draw].sum(1)/counts[draw].sum(1)[:,None]
                contrasts.append(dict(panel=panel,contrast=learner+' minus '+base,n=len(kk),
                    difference=dict(zip(METRICS,(sums.sum(0)/counts.sum()).tolist())),
                    anatomy_block_ci95=dict(zip(METRICS,np.quantile(boots,[.025,.975],axis=0).T.tolist()))))
        lines.append('')
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (a.out/'aggregate.json').write_text(json.dumps(dict(tables=output,paired_contrasts=contrasts,strata=strata),indent=2))

if __name__=='__main__': main()
