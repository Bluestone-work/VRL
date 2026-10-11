"""Paired image-SAC/feedback/old-PPO/original evaluations; all failures retained."""
import os
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,os.sched_getaffinity(0)-{6,7})
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='1'
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OLD='research/runs/EXP0094_PPO_NAV_20261011/seed0/policy_it80.pt'


def scenes(pool):
    if pool=='v5':
        from scripts.eval_exp0090 import jobs_for
        return jobs_for(SimpleNamespace(protocol='v5',strengths='0,1,1.5',count=10))
    if pool=='hard':
        from scripts.eval_exp0090 import jobs_for
        return jobs_for(SimpleNamespace(protocol='hard',panels='low_delay,moderate_delay,high_delay,strong_flow,variable_response,ood_flow_delay',train_only=False))
    split=json.load(open(ROOT/'configs/evaluation_splits.json'))['anatomy_holdout_v1']
    anatomies=split['train'] if pool in ('screen','tune') else json.load(open(ROOT/'configs/evaluation_splits.json'))['anatomy_order']
    if pool=='screen':anatomies=['mca_m1_lvo','ica_siphon','coronary_lm_bifurcation']
    conditions=[('normal',.05,1,0.),('strong_flow',.1,2,.625),('pulsatile_response',.05,2,1.25),('strong_pulsatile_ood',.15,3,1.5)]
    jobs=[]
    for p,(name,f,L,s) in enumerate(conditions):
        for k,an in enumerate(anatomies):
            for n in ((1,3) if pool=='screen' else (1,2,3)):
                base=2150000000 if pool in ('screen','tune') else 2153000000
                seed=base+p*100000+k*1000+n
                jobs.append((name+'|'+('train' if an in split['train'] else 'held_out'),(an,n,seed,f,L,s)))
    return jobs


def work(spec):
    from marl.image_flow_sac import ControlEpisode,ImageFlowController
    from marl.predictive_nav_rl import RLNavigator,PREDICTORS,PARENT_WEIGHTS
    from marl.dyn_select import PredictiveSelector
    from marl.dyn_predictor import Ensemble
    from marl.deployable_sensing import DeployableConfig
    method,cell,job,config,trace_folder,*extra=spec
    attribution=bool(extra and extra[0])
    an,n,seed,flow,delay,strength=job
    ep=ControlEpisode(n,an,seed,flow_inlet_mm_s=flow,sense_cfg=DeployableConfig(latency_steps=delay),variation=strength or None,
                      dynamic_obstacle_count=int(config.get('dynamic_obstacle_count',0)))
    kind,_,path=method.partition('=')
    if kind=='parent':ctl=PredictiveSelector(ep,'learned',Ensemble(PREDICTORS),.0445,PARENT_WEIGHTS)
    elif kind=='old_ppo':ctl=RLNavigator(ep,checkpoint=OLD)
    elif kind=='feedback':ctl=ImageFlowController(ep,config=config)
    elif kind=='sac':ctl=ImageFlowController(ep,checkpoint=path,config=config)
    elif kind in ('sac_safe','feedback_safe'):
        from marl.sac_safe_local import SafeLocalSAC
        ctl=SafeLocalSAC(ep,checkpoint=path if kind=='sac_safe' else None,config=config)
    elif kind in ('sac_speed','sac_speed_half'):
        from marl.sac_safe_local import SafeLocalSAC
        cfg=dict(config);cfg['action_scales']=[.2 if kind=='sac_speed' else .1,0.,0.]
        ctl=SafeLocalSAC(ep,checkpoint=path,config=cfg)
    elif kind=='sac_scalar':
        from marl.sac_speed_control import SpeedOnlySAC
        ctl=SpeedOnlySAC(ep,checkpoint=path,config=config)
    elif kind=='td3':
        from marl.sac_speed_control import SpeedOnlySAC
        ctl=SpeedOnlySAC(ep,checkpoint=path,config=config,stochastic=False)
    elif kind in ('sac_enveloped','sac_finite','sac_finite_stochastic'):
        from marl.sac_local_envelope import EnvelopedSpeedSAC
        ctl=EnvelopedSpeedSAC(ep,checkpoint=path,config=config,stochastic=kind=='sac_finite_stochastic')
    elif kind=='sac_gate':
        from marl.sac_local_variants import CriticGatedSAC
        ctl=CriticGatedSAC(ep,checkpoint=path,config=config)
    elif kind in ('feedback_near','sac_near'):
        from marl.sac_local_variants import NearImageServo
        ctl=NearImageServo(ep,checkpoint=path if kind=='sac_near' else None,config=config)
    else:raise ValueError(method)
    policy_rng_seed=None
    if kind=='sac_finite_stochastic':
        import torch
        # Common policy-noise stream across checkpoints/ablations on each
        # scene. The environment uses its own RNG; the seed is not a feature.
        policy_rng_seed=int(hashlib.sha256(json.dumps(['SAC_policy_noise_v1',job],sort_keys=True).encode()).hexdigest()[:8],16)
        torch.manual_seed(policy_rng_seed)
    errors=[];near=[];dwell=0;eligible=0;image_error=[];pred_error=[]
    command_audit=dict(steps=0,raw_l1=0.,guarded_l1=0.,executed_l1=0.,
        raw_nontrivial=0,guarded_nontrivial=0,executed_nontrivial=0,executed_max=0.)
    trace=dict(t=[0.],pos=[ep.env.positions_mm[:n].copy()],mass=[ep.env.masses.copy()],wall=[0.],removal=[0.])
    if trace_folder:
        trace.update(sent=[ep.sent_world.copy()],hold=[np.zeros(n,bool)],
                     targets=[np.full(n,-1,int)],reference_local=[np.zeros((n,3))],
                     policy_local=[np.zeros((n,3))],
                     flow_factor=[ep.var.q(0.) if ep.var else 1.])
    try:
        while True:
            est=ep.observe();tgt=ep.plan_targets_now(est);rule=ep.ctl.act(tgt,est);hold=ep.hold(est)
            local=ctl(ep,est,tgt,rule,hold)
            # Same-state zero-residual counterfactual, including the same hold
            # projection, WallGuard, and spacing-filter state. This is a
            # measurement only; restore all bookkeeping before the real step.
            counterfactual=None
            if attribution and kind in ('sac','sac_safe','sac_speed','sac_speed_half','sac_scalar','sac_enveloped','sac_finite','sac_finite_stochastic'):
                saved_stats=ctl.stats.copy();saved_actions=ctl.last_actions.copy()
                counterfactual=ctl.execute(ep,est,ctl.references)
                ctl.stats=saved_stats;ctl.last_actions=saved_actions
                raw_delta=np.linalg.norm(saved_actions-ctl.references,axis=1)[est.active]
                guarded_delta=np.linalg.norm(local-counterfactual,axis=1)[est.active]
                if not ep.station_keeping:counterfactual[hold]=0.
                if ep.shield is not None:
                    from scripts.benchmark_deployable import packet_from
                    saved_shield=copy.deepcopy(ep.shield.__dict__)
                    try:counterfactual=ep.shield.filtered(counterfactual,packet_from(est,ep.ctl.frames(est),ep.prev_local))
                    finally:
                        ep.shield.__dict__.clear();ep.shield.__dict__.update(saved_shield)
                command_audit['steps']+=int(est.active.sum())
                for prefix,delta in [('raw',raw_delta),('guarded',guarded_delta)]:
                    command_audit[prefix+'_l1']+=float(delta.sum())
                    command_audit[prefix+'_nontrivial']+=int((delta>.001).sum())
            truth=ep.env.positions_mm[:n]
            image_error.extend(np.linalg.norm(est.pos-truth,axis=1)[est.active].tolist())
            if hasattr(ctl,'positions'):pred_error.extend(np.linalg.norm(ctl.positions-truth,axis=1)[est.active].tolist())
            for i in range(n):
                if tgt[i]<0 or not est.active[i] or hold[i]:continue
                R=ep.ctl.route[i]
                if R is not None:
                    k=int(ep.ctl.prog[i]);P=ep.ctl.pts[R[max(0,k-4):k+12]]
                    if len(P):errors.append(float(np.linalg.norm(P-truth[i],axis=1).min()))
                d=float(np.linalg.norm(truth[i]-ep.env.clot_positions_mm[tgt[i]]))
                if d<1.:near.append(d);eligible+=1;dwell+=int(d<.12)
            done,result=ep.step(est,local,hold)
            if counterfactual is not None:
                delta=np.linalg.norm(ep.prev_local-counterfactual,axis=1)[est.active]
                command_audit['executed_l1']+=float(delta.sum())
                command_audit['executed_nontrivial']+=int((delta>.001).sum())
                command_audit['executed_max']=max(command_audit['executed_max'],float(delta.max(initial=0.)))
            if trace_folder:
                trace['t'].append(float(ep.env.elapsed_s));trace['pos'].append(ep.env.positions_mm[:n].copy())
                trace['mass'].append(ep.env.masses.copy());trace['wall'].append(float(np.sum(result['wall'])))
                trace['removal'].append(1-float(ep.env.masses.sum())/ep.initial)
                trace['sent'].append(ep.sent_world.copy());trace['hold'].append(hold.copy())
                trace['targets'].append(np.asarray(tgt,int).copy())
                trace['reference_local'].append(np.asarray(getattr(ctl,'references',local),float).copy())
                trace['policy_local'].append(np.asarray(getattr(ctl,'last_actions',local),float).copy())
                trace['flow_factor'].append(ep.var.q(ep.env.elapsed_s) if ep.var else 1.)
            if done:break
        row=ep.row(method,**ctl.summary(),**ep.authority)
        row.update(cell=cell,flow_inlet_mm_s=flow,latency_steps=delay,variation_s=strength,
            route_cross_track_rmse_mm=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            near_target_mean_error_mm=float(np.mean(near)) if near else None,
            near_target_contact_fraction=dwell/max(eligible,1),
            image_position_rmse_mm=float(np.sqrt(np.mean(np.square(image_error)))) if image_error else None,
            compensated_position_rmse_mm=float(np.sqrt(np.mean(np.square(pred_error)))) if pred_error else None,
            deployment_action_sampling='seeded stochastic' if policy_rng_seed is not None else 'deterministic',
            policy_rng_seed=policy_rng_seed)
        if attribution and kind in ('sac','sac_safe','sac_speed','sac_speed_half','sac_scalar','sac_enveloped','sac_finite','sac_finite_stochastic'):
            row['rl_command_audit']=command_audit
            for prefix in ('raw','guarded','executed'):
                row['rl_'+prefix+'_mean_delta']=command_audit[prefix+'_l1']/max(command_audit['steps'],1)
                row['rl_'+prefix+'_nontrivial_fraction']=command_audit[prefix+'_nontrivial']/max(command_audit['steps'],1)
        if trace_folder:
            from scripts.export_vrl_v3_all_gifs import geometry
            name=hashlib.sha256(json.dumps([method,cell,job],sort_keys=True).encode()).hexdigest()[:16]
            target=Path(trace_folder)/f'{an}_N{n}_seed{seed}_{name}.npz';target.parent.mkdir(parents=True,exist_ok=True)
            meta=dict(anatomy=an,n=n,seed=seed,flow=flow,latency=delay,s=strength,method=method,cell=cell)
            np.savez_compressed(target,**{k:np.asarray(v) for k,v in trace.items()},geometry=json.dumps(geometry(ep)),metrics=json.dumps(row),meta=json.dumps(meta))
        return row
    finally:ep.close()


def safe(spec):
    try:return work(spec)
    except Exception:return dict(method=spec[0],cell=spec[1],job=spec[2],error=traceback.format_exc())


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--pool',choices=('screen','tune','hard','stress','v5'),default='screen')
    p.add_argument('--methods',default='parent,old_ppo,feedback');p.add_argument('--workers',type=int,default=16)
    p.add_argument('--controller-json',type=Path);p.add_argument('--traces',action='store_true')
    p.add_argument('--attribution',action='store_true');p.add_argument('--scene-seed-offset',type=int,default=0)
    p.add_argument('--dynamic-obstacles',action='store_true');p.add_argument('--dynamic-obstacle-count',type=int,default=8)
    a=p.parse_args();os.chdir(ROOT)
    a.out.mkdir(parents=True,exist_ok=True);config=json.load(open(a.controller_json)) if a.controller_json else {}
    config['dynamic_obstacle_count']=a.dynamic_obstacle_count if a.dynamic_obstacles else 0
    done=set()
    if (a.out/'episodes.jsonl').exists():
        for l in (a.out/'episodes.jsonl').open():
            r=json.loads(l);done.add((r['method'],r['cell'],r['anatomy'],r['clusters'],r['seed']))
    registry=[(cell,(an,n,seed+a.scene_seed_offset,f,L,s)) for cell,(an,n,seed,f,L,s) in scenes(a.pool)]
    specs=[(m,cell,job,config,str(a.out/'traces') if a.traces else None,a.attribution) for cell,job in registry for m in a.methods.split(',')
           if (m,cell,job[0],job[1],job[2]) not in done]
    manifest=dict(pool=a.pool,methods=a.methods,config=config,n_specs=len(specs),development_only=True,attribution=a.attribution,
        scene_seed_offset=a.scene_seed_offset,
        source_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('marl/image_flow_sac.py','scripts/eval_image_flow_sac.py')},
        checkpoints={m.partition('=')[2]:hashlib.sha256(Path(m.partition('=')[2]).read_bytes()).hexdigest() for m in a.methods.split(',') if m.startswith('sac')})
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with mp.get_context('spawn').Pool(a.workers,maxtasksperchild=20) as pool,(a.out/'episodes.jsonl').open('a') as log,(a.out/'failures.jsonl').open('a') as fail:
        for k,row in enumerate(pool.imap_unordered(safe,specs)):
            f=fail if 'error' in row else log;f.write(json.dumps(row)+'\n');f.flush()
            if 'error' in row:print(row['error'],flush=True)
            if k%10==0 or k+1==len(specs):print(k+1,'/',len(specs),flush=True)


if __name__=='__main__':main()
