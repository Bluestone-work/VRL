"""Measure where the previous local PPO loses control authority; diagnostics only."""
import os
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,os.sched_getaffinity(0)-{6,7})
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT='research/runs/EXP0094_PPO_NAV_20261011/seed0/policy_it80.pt'


def audit(job):
    from scripts.benchmark_lysis import LysisEpisode
    from marl.deployable_sensing import DeployableConfig
    from marl.predictive_nav_rl import RLNavigator
    an,n,seed,flow,delay,strength=job
    ep=LysisEpisode(n,an,seed,flow_inlet_mm_s=flow,sense_cfg=DeployableConfig(latency_steps=delay),variation=strength or None)
    ctl=RLNavigator(ep,checkpoint=CHECKPOINT)
    stats=dict(agent_steps=0,tpg_hold_steps=0,no_target_steps=0,policy_steps=0,guard_after_policy_changes=0,
               shield_changes=0,hold_wall_robot_s=0.,held_motion_mm=0.,target_switches=0,
               candidate_total=0,candidate_valid=0)
    previous_target=None
    try:
        while True:
            est=ep.observe();tgt=ep.plan_targets_now(est);rule=ep.ctl.act(tgt,est);hold=ep.hold(est)
            F=ep.ctl.frames(est);local=ctl(ep,est,tgt,rule,hold)
            active=est.active.copy();live=(np.asarray(tgt)>=0)&active&~hold
            stats['agent_steps']+=int(active.sum());stats['tpg_hold_steps']+=int((hold&active).sum())
            stats['no_target_steps']+=int(((np.asarray(tgt)<0)&active).sum());stats['policy_steps']+=int(live.sum())
            if previous_target is not None:stats['target_switches']+=int((np.asarray(tgt)!=previous_target).sum())
            previous_target=np.asarray(tgt).copy()
            for r in ctl.records:
                stats['candidate_total']+=int((np.abs(r['cand']).sum(1)>0).sum())
                stats['candidate_valid']+=int(r['valid'].sum())
                raw=F[r['agent']]@r['selected_command']
                stats['guard_after_policy_changes']+=int(np.linalg.norm(local[r['agent']]-raw)>1e-5)
            before=ep.env.positions_mm[:n].copy()
            done,result=ep.step(est,local,hold)
            after_hold=local.copy();after_hold[hold]=0.
            stats['shield_changes']+=int(((np.linalg.norm(ep.prev_local-after_hold,axis=1)>1e-5)&active).sum())
            stats['held_motion_mm']+=float(np.linalg.norm(ep.env.positions_mm[:n]-before,axis=1)[hold&active].sum())
            stats['hold_wall_robot_s']+=float(np.asarray(result['wall'])[hold&active].sum())
            if done:break
        row=ep.row('old_PPO_authority_audit',**ctl.summary(),**stats)
        row.update(flow_inlet_mm_s=flow,latency_steps=delay,variation_s=strength)
        return row
    finally:ep.close()


def safe(job):
    try:return audit(job)
    except Exception:return dict(job=job,error=traceback.format_exc())


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();os.chdir(ROOT);a.out.mkdir(parents=True,exist_ok=True)
    anatomies=['mca_m1_lvo','ica_siphon','coronary_lm_bifurcation']
    conditions=[(.05,1,0.),(.05,3,0.),(.1,2,.625),(.05,2,1.25),(.15,3,1.5)]
    jobs=[(an,n,2148000000+i*10000+j*100+n,flow,delay,s) for i,an in enumerate(anatomies)
          for n in (1,3) for j,(flow,delay,s) in enumerate(conditions)]
    with mp.get_context('spawn').Pool(a.workers) as pool,(a.out/'episodes.jsonl').open('w') as log:
        rows=[]
        for row in pool.imap_unordered(safe,jobs):
            log.write(json.dumps(row)+'\n');log.flush();rows.append(row)
            print('audit',len(rows),'/',len(jobs),'errors',sum('error' in r for r in rows),flush=True)
    valid=[r for r in rows if 'error' not in r]
    counts={k:sum(r[k] for r in valid) for k in ('agent_steps','tpg_hold_steps','no_target_steps','policy_steps',
        'guard_after_policy_changes','shield_changes','hold_wall_robot_s','held_motion_mm','candidate_total','candidate_valid')}
    counts.update(episodes=len(valid),errors=len(rows)-len(valid),
                  tpg_hold_fraction=counts['tpg_hold_steps']/max(counts['agent_steps'],1),
                  guard_after_policy_fraction=counts['guard_after_policy_changes']/max(counts['policy_steps'],1),
                  shield_changed_fraction=counts['shield_changes']/max(counts['agent_steps'],1))
    (a.out/'summary.json').write_text(json.dumps(counts,indent=2));print(json.dumps(counts,indent=2))


if __name__=='__main__':
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
    main()
