"""Verify seeded stochastic replay and that command auditing leaves physics unchanged."""
import os
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, os.sched_getaffinity(0)-{6, 7})
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scripts.eval_image_flow_sac import ROOT,work


def main():
    os.chdir(ROOT)
    output=ROOT/'research/validation/EXP0099_FINITE_IMAGE_SAC_20261011/replay_verification'
    output.mkdir(parents=True,exist_ok=True)
    method='sac_finite_stochastic=research/runs/EXP0099_FINITE_IMAGE_SAC_20261011/seed0/policy_it40.pt'
    job=('mca_m1_lvo',1,2179000001,.05,1,0.)
    specs=[(method,'upstream_regression|train',job,{},str(output/f'traces_{audit}'),audit) for audit in (False,True)]
    with mp.get_context('spawn').Pool(2) as pool:
        results=pool.map(work,specs)
    equality={}
    left=next((output/'traces_False').glob('*.npz'))
    right=next((output/'traces_True').glob('*.npz'))
    with np.load(left) as a,np.load(right) as b:
        for key in ('t','pos','mass','wall','removal','sent','hold','flow_factor','targets','reference_local','policy_local'):
            equality[key]=bool(np.array_equal(a[key],b[key]))
            if not equality[key]:raise AssertionError('Replay/audit changed '+key)
        assert a['targets'].shape==(len(a['t']),1)
        assert a['reference_local'].shape==a['policy_local'].shape==(len(a['t']),1,3)
    record=dict(equality=equality,policy_rng_seed=results[0]['policy_rng_seed'],
                attribution_does_not_change_physics=True,results=results,
                limitation='One known training-anatomy regression case; not checkpoint selection or a general improvement claim')
    (output/'VERIFIED.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    print(json.dumps({key:record[key] for key in ('equality','policy_rng_seed','attribution_does_not_change_physics')},ensure_ascii=False),flush=True)
    for row in results:print(json.dumps({key:row[key] for key in ('strict_success','removal','wall_contact_s','elapsed_s')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
