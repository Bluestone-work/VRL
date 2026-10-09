"""Reanalyse existing EXP0062 records, without environment execution."""
import json
import hashlib
import shutil
from pathlib import Path
import numpy as np
from scripts.evaluate_lysis_abcd import RUNS

OUT=Path('research/validation/FLOW_DELAY_AUDIT')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    src=Path('research/validation/EXP0062_paired_v2')
    rows=[json.loads(l) for l in (src/'episodes.jsonl').read_text().splitlines()]
    manifest=json.loads((src/'manifest.json').read_text())
    splits=json.loads(Path('configs/evaluation_splits.json').read_text())
    train=set(splits['anatomy_holdout_v1']['train'])
    key=lambda r:(r['method'],r['suite'],r['anatomy'],r['clusters'],r['seed'])
    assert len(rows)==1470 and len({key(r) for r in rows})==1470
    assert {key(r) for r in rows}=={tuple(x) for x in manifest['jobs']}
    assert not any('error' in r for r in rows)
    checkpoint={}
    for arm,run in RUNS.items():
        d=Path('research/runs')/run
        sha=hashlib.sha256((d/'policy.pt').read_bytes()).hexdigest()
        assert sha==manifest['checkpoint_sha256'][arm]
        target=OUT/'training'/arm;target.mkdir(parents=True,exist_ok=True)
        for f in ('config.json','log.jsonl','episodes.jsonl','auxiliary.jsonl'):
            if (d/f).exists(): shutil.copyfile(d/f,target/f)
        import torch
        ck=torch.load(d/'policy.pt',map_location='cpu',weights_only=False)
        assert all(torch.isfinite(v).all() for k,v in ck['state'].items() if k != 'causal')
        checkpoint[arm]=dict(sha256=sha,cfg=ck['cfg'],updates=ck['it'],agent_steps=ck['agent_steps'])
    (OUT/'checkpoints.json').write_text(json.dumps(checkpoint,indent=2))
    ledger=dict(source_commit='2776f07',source=str(src),purpose='retroactive disclosure of accidental sealed-test access',
                records=1470,smoke_records=4,action='offset 11 permanently reclassified development, no replacement',
                scenes=[dict(anatomy=an,seed=seed,offset=11,all_N_and_conditions_development=True)
                        for an,seed in sorted({(r['anatomy'],r['seed']) for r in rows})])
    (OUT/'access_ledger.json').write_text(json.dumps(ledger,indent=2))
    for r in rows:
        r['T90_300']=r['t90_s'] if r['t90_s'] is not None else 300.
        r['t90_reached']=r['t90_s'] is not None
        r['anatomy_split']='train' if r['anatomy'] in train else 'parameter_unseen'
    index={key(r):r for r in rows}
    fields=['task_success','strict_success','removal','T90_300','t90_reached','removal_auc','wall_contact_s','lost','spacing_violation_pair_s']
    summaries=[];contrasts=[]
    rng=np.random.default_rng(630)
    # Resample the scene/anatomy block; all N and all conditions stay together.
    # There is only one scene per anatomy, so these intervals mix anatomical variability
    # and scene uncertainty; they cannot estimate independent within-anatomy variance.
    def stats(rs):
        blocks=sorted({(r['anatomy'],r['seed']) for r in rs})
        mat=np.array([[np.mean([r[f] for r in rs if (r['anatomy'],r['seed'])==b]) for f in fields] for b in blocks])
        draw=mat[rng.integers(0,len(blocks),(2000,len(blocks)))].mean(1)
        return {f:dict(mean=float(mat[:,j].mean()),ci95=np.percentile(draw[:,j],[2.5,97.5]).tolist()) for j,f in enumerate(fields)}
    for split in ('all','train','parameter_unseen'):
      for n in (0,1,2,3):
       for suite in ['all']+sorted({r['suite'] for r in rows}):
        for method in sorted({r['method'] for r in rows}):
            rs=[r for r in rows if r['method']==method and (split=='all' or r['anatomy_split']==split) and (not n or r['clusters']==n) and (suite=='all' or r['suite']==suite)]
            reached=[r['t90_s'] for r in rs if r['t90_s'] is not None]
            summaries.append(dict(split=split,N=n or 'all',suite=suite,method=method,n=len(rs),stats=stats(rs),
                                  conditional_T90=float(np.mean(reached)) if reached else None,t90_count=len(reached)))
        for c,b in [('B','A'),('C','B'),('D','C'),('no_settle','settle'),('A','A_legacy')]:
            cr=[r for r in rows if r['method']==c and (split=='all' or r['anatomy_split']==split) and (not n or r['clusters']==n) and (suite=='all' or r['suite']==suite)]
            ds=[]
            for r in cr:
                br=index[(b,*key(r)[1:])]
                assert all(r[f]==br[f] for f in ('scenario_hash','initial_state_hash','variation','actual_healthy_mean_mm_s'))
                ds.append(dict(r,**{f:float(r[f])-float(br[f]) for f in fields}))
            contrasts.append(dict(split=split,N=n or 'all',suite=suite,contrast=c+'-'+b,stats=stats(ds)))
    (OUT/'stratified.json').write_text(json.dumps(dict(summaries=summaries,paired=contrasts,bootstrap_unit='anatomy/requested scene; retain all N and conditions',limitations='one scene per anatomy; one training seed; exploratory intervals'),indent=2))
    print('1470 unique complete records, paired hashes, finite checkpoints and checkpoint hashes verified')

if __name__=='__main__':main()
