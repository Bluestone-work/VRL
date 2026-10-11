"""Train PPO navigation, retaining the strongest frozen predictive belief and A_TPG.

All updates use real on-policy environment rewards; supervised predictor weights
are frozen. This is not supervised relabeling of the old selector. Train anatomy
and scene pools are disjoint from all tuning/development/sealed-test pools.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
# The repository records unstable CPU cores 6/7. Set affinity before importing
# NumPy/Torch, including in spawned rollout workers.
if hasattr(os,'sched_getaffinity'):
    allowed=os.sched_getaffinity(0)-{6,7}
    if allowed:os.sched_setaffinity(0,allowed)
import numpy as np
import torch
from torch import nn
from marl.predictive_nav_rl import NavigationActor,RLNavigator,PREDICTORS,advantages,observation_tensors

ROOT=Path(__file__).resolve().parents[1]

def worker(wid,conn,cfg,out):
    try:
        torch.set_num_threads(1);torch.manual_seed(cfg['seed']*1000+wid)
        from marl.deployable_sensing import DeployableConfig
        from scripts.benchmark_lysis import LysisEpisode
        from scripts.train_lysis_local import route_remaining
        from scripts.export_vrl_v3_all_gifs import geometry
        train=json.loads((ROOT/'configs/evaluation_splits.json').read_text())['anatomy_holdout_v1']['train']
        rng=np.random.default_rng([9400,cfg['seed'],wid]);net=NavigationActor(cfg['width'],cfg['proposal_bias']);episode_id=0
        def prepare(ep):
            est=ep.observe();tgt=ep.plan_targets_now(est);rule=ep.ctl.act(tgt,est);hold=ep.hold(est)
            return est,tgt,rule,hold
        def next_values(ep,ctl):
            # Critic sees the genuine next delayed observation. Temporarily add
            # it to history, then restore so the next actor appends it once.
            from collections import deque
            est,tgt,rule,hold=ep.prepared
            saved=[deque(b,maxlen=b.maxlen) for b in ctl.feat.buf]
            ctl.feat.push(ep,est,hold);result={}
            try:
                for i in range(ep.n):
                    F=ep.ctl.frames(est)[i];hist,hmask=ctl.feat.history(ep,est,i,F);ctx=ctl.feat.context(ep,est,tgt,i,F)
                    cand=np.zeros((32,50),np.float32);valid=np.zeros(32,bool);valid[0]=True
                    with torch.no_grad():
                        _,v=net.dist(torch.as_tensor(hist[None]),torch.as_tensor(hmask[None]),torch.as_tensor(ctx[None]),
                                     torch.as_tensor(cand[None]),torch.as_tensor(valid[None]))
                    result[i]=float(v[0])
            finally:ctl.feat.buf=saved
            return result
        def new_episode():
            nonlocal episode_id
            while True:
                seed=cfg['scene_seed_base']+cfg['seed']*1000000+wid*10000+episode_id;episode_id+=1
                if episode_id>=10000:raise RuntimeError('Training worker seed range exhausted')
                an=str(rng.choice(train));n=int(rng.integers(1,4));flow=float(rng.choice([.025,.05,.1]))
                lat=int(rng.integers(1,4));s=0. if rng.random()<.25 else float(rng.uniform(0,1.25))
                try:
                    sense=DeployableConfig(latency_steps=lat)
                    ep=LysisEpisode(n,an,seed,sense_cfg=sense,variation=s or None,flow_inlet_mm_s=flow,
                                    dynamic_obstacle_count=cfg['dynamic_obstacle_count'])
                    break
                except (ValueError,RuntimeError):continue
            ctl=RLNavigator(ep,net=net,stochastic=True,predictors=cfg['predictors'],uncertainty_fallback=False)
            ep.metadata=dict(anatomy=an,n=n,seed=seed,flow=flow,latency=lat,s=s,training_seed=cfg['seed'],episode_id=episode_id)
            ep.geo=geometry(ep);ep.ret=0.;ep.previous=np.zeros((n,3));ep.prev_spacing_violation=0.;ep.policy_iterations=[]
            ep.trace=dict(t=[0.],pos=[ep.env.positions_mm[:n].copy()],mass=[ep.env.masses.copy()],wall=[0.],removal=[0.],sent=[np.zeros((n,3))])
            ep.prepared=prepare(ep)
            return ep,ctl
        ep,ctl=new_episode()
        while True:
            msg=conn.recv()
            if msg is None:break
            sd,iteration=msg;net.load_state_dict(sd);buf=[];finished=[];boot={}
            for k in range(cfg['rollout']):
                est,tgt,rule,hold=ep.prepared
                local=ctl(ep,est,tgt,rule,hold)
                old=np.array([route_remaining(ep.ctl,i,ep.env.positions_mm[i]) if tgt[i]>=0 else 0. for i in range(ep.n)])
                done,res=ep.step(est,local,hold)
                new=np.array([route_remaining(ep.ctl,i,ep.env.positions_mm[i]) if tgt[i]>=0 else 0. for i in range(ep.n)])
                progress=np.clip(old-new,-.2,.2)*(np.asarray(tgt)>=0)
                # First-round team reward: physical clearance and continuous
                # route potential are dense signals; safety terms are step
                # increments (never re-charge an earlier violation).
                dt=float(ep.env.config.control_dt_s)
                delta_clear=float(res['removed'])
                route_potential_delta=np.clip(progress, -.2, .2)
                wall_delta=np.asarray(res['wall'],float)
                spacing_delta=0.
                if ep.n>1:
                    D=np.linalg.norm(ep.env.positions_mm[:ep.n,None]-ep.env.positions_mm[None,:ep.n],axis=-1)+np.eye(ep.n)*99
                    spacing_now=float(np.count_nonzero(np.triu(D<ep.d_min,1)))*dt
                    spacing_delta=max(0.,spacing_now-ep.prev_spacing_violation)
                    ep.prev_spacing_violation=spacing_now
                lost_delta=np.asarray(res['lost'],float)
                reward=(10.*delta_clear + 1.0*route_potential_delta
                        -2.*wall_delta -2.*spacing_delta
                        -10.*lost_delta -0.01*dt)
                row=None
                if done:
                    row=ep.row('ppo_training',**ctl.summary())
                    reward+=20.*float(row['strict_success'])
                for r in ctl.records:
                    i=r['agent'];r.update(stream=(wid,episode_id,i),reward=float(reward[i]),
                                         terminal=bool(res['terminated'] or not ep.env.active[i]),
                                         truncated=bool(res['truncated'] and not res['terminated']),terminal_value=0.)
                    buf.append(r)
                ep.ret+=float(reward.sum());ep.policy_iterations.append(iteration)
                tr=ep.trace;tr['t'].append(float(ep.env.elapsed_s));tr['pos'].append(ep.env.positions_mm[:ep.n].copy())
                tr['mass'].append(ep.env.masses.copy());tr['wall'].append(float(np.sum(res['wall'])))
                tr['removal'].append(1-float(ep.env.masses.sum())/ep.initial);tr['sent'].append(ep.sent_world.copy())
                ep.prepared=prepare(ep)
                if done:
                    if res['truncated'] and not res['terminated']:
                        # Bootstrap the terminal delayed observation, before reset.
                        values=next_values(ep,ctl)
                        for r in ctl.records:
                            r['terminal_value']=values[r['agent']]
                    row.update(ep.metadata,return_sum=ep.ret,policy_iterations=sorted(set(ep.policy_iterations)))
                    finished.append(row)
                    trace_dir=Path(out)/'training_traces';trace_dir.mkdir(exist_ok=True)
                    np.savez_compressed(trace_dir/f"{ep.metadata['anatomy']}_N{ep.n}_seed{ep.seed}.npz",
                        **{key:np.asarray(value) for key,value in tr.items()},geometry=json.dumps(ep.geo),metrics=json.dumps(row),meta=json.dumps(ep.metadata))
                    ep.close();ep,ctl=new_episode()
            boot={tuple(r['stream']):0. for r in buf}
            for i,v in next_values(ep,ctl).items():boot[(wid,episode_id,i)]=v
            conn.send(dict(buf=buf,bootstrap=boot,finished=finished))
        ep.close()
    except BaseException:conn.send(dict(error=traceback.format_exc()))
    finally:conn.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--workers',type=int,default=6);ap.add_argument('--updates',type=int,default=160);ap.add_argument('--rollout',type=int,default=128)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--width',type=int,default=64);ap.add_argument('--proposal-bias',type=float,default=.2)
    ap.add_argument('--lr',type=float,default=2e-4);ap.add_argument('--scene-seed-base',type=int,default=2130000000)
    ap.add_argument('--predictors',default=','.join(PREDICTORS));ap.add_argument('--resume',type=Path)
    ap.add_argument('--dynamic-obstacles',action='store_true')
    ap.add_argument('--dynamic-obstacle-count',type=int,default=8)
    a=ap.parse_args();os.chdir(ROOT);torch.set_num_threads(1);torch.manual_seed(a.seed)
    a.out.mkdir(parents=True,exist_ok=a.resume is not None)
    cfg={k:v for k,v in vars(a).items() if k not in ('out','resume')};cfg['predictors']=a.predictors.split(',')
    cfg['dynamic_obstacle_count']=a.dynamic_obstacle_count if a.dynamic_obstacles else 0
    cfg.update(algorithm='PPO categorical recurrent local navigation',global_layer='A_TPG',gamma=.995,gae_lambda=.95,
               shared_safety=True,predictor_frozen=True,observations='delayed image history, issued commands, preoperative map, frozen candidate predictions',
               training_anatomies=json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train'])
    cfg['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
        ['marl/predictive_nav_rl.py','marl/dyn_select.py','scripts/train_predictive_nav_rl.py','scripts/benchmark_lysis.py']}
    cfg['predictor_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in cfg['predictors']}
    (a.out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    net=NavigationActor(a.width,a.proposal_bias);start_iteration=steps=0
    optimizer_state=None
    if a.resume:
        payload=torch.load(a.resume,map_location='cpu',weights_only=False);net.load_state_dict(payload['state'])
        start_iteration=payload['iterations'];steps=payload['agent_steps'];optimizer_state=payload.get('optimizer')
        # Resume uses a new disjoint block of episode seeds; no episode is replayed as new on-policy data.
        cfg['scene_seed_base']+=10000000+start_iteration*100000
    context=mp.get_context('spawn');pipes=[];procs=[]
    for wid in range(a.workers):
        p,c=context.Pipe();pr=context.Process(target=worker,args=(wid,c,cfg,str(a.out)));pr.start();c.close();pipes.append(p);procs.append(pr)
    restarts=[0]*a.workers;restart_serial=0
    def recover(wid,reason,state,it):
        nonlocal restart_serial
        restarts[wid]+=1;restart_serial+=1
        if restarts[wid]>6:raise RuntimeError(f'Worker {wid} exceeded its recovery budget: {reason}')
        old=procs[wid];old.join(timeout=.2)
        if old.is_alive():old.terminate();old.join(timeout=2)
        pipes[wid].close();replacement=dict(cfg)
        replacement['scene_seed_base']+=5000000+restart_serial*100000
        p,c=context.Pipe();pr=context.Process(target=worker,args=(wid,c,replacement,str(a.out)));pr.start();c.close()
        pipes[wid]=p;procs[wid]=pr
        with (a.out/'worker_recovery.jsonl').open('a') as f:
            f.write(json.dumps(dict(iteration=it,worker=wid,old_exitcode=old.exitcode,reason=reason,
                                    restart=restart_serial,new_scene_seed_base=replacement['scene_seed_base']))+'\n')
        print(f'Recovered rollout worker {wid}: {reason}',flush=True)
        p.send((state,it))
    initial_state={k:v.clone() for k,v in net.state_dict().items()};net=net.to(a.device)
    optimizer=torch.optim.Adam(net.parameters(),lr=a.lr)
    if optimizer_state:optimizer.load_state_dict(optimizer_state)
    start=time.monotonic();iteration=start_iteration
    def save(name):
        temp=a.out/(name+'.tmp')
        torch.save(dict(state={k:v.detach().cpu() for k,v in net.state_dict().items()},optimizer=optimizer.state_dict(),
                        config=cfg,iterations=iteration,agent_steps=steps,trained_with_rl=iteration>0),temp)
        temp.replace(a.out/name)
    save('initial.pt' if a.resume is None else 'resumed_initial.pt')
    try:
        with (a.out/'training.jsonl').open('a') as log,(a.out/'episodes.jsonl').open('a') as elog:
            while iteration<a.updates:
                state={k:v.detach().cpu() for k,v in net.state_dict().items()}
                for wid,p in enumerate(pipes):
                    try:p.send((state,iteration))
                    except (BrokenPipeError,EOFError):recover(wid,'pipe closed before rollout',state,iteration)
                data=[]
                for wid in range(a.workers):
                    while True:
                        p=pipes[wid]
                        try:
                            if not p.poll(600):raise TimeoutError('RL rollout worker exceeded 600 s')
                            d=p.recv()
                            if 'error' in d:raise RuntimeError(d['error'])
                            data.append(d);break
                        except (EOFError,TimeoutError,RuntimeError,ConnectionError) as exc:
                            recover(wid,str(exc) or 'rollout subprocess exited',state,iteration)
                buf=sum((d['buf'] for d in data),[]);boot={};finished=[]
                for d in data:boot.update(d['bootstrap']);finished.extend(d['finished'])
                for r in finished:elog.write(json.dumps(r,ensure_ascii=False)+'\n')
                elog.flush()
                if not buf:raise RuntimeError('Empty PPO batch')
                adv,ret=advantages(buf,boot,cfg['gamma'],cfg['gae_lambda']);obs=observation_tensors(buf,a.device)
                T=lambda k,dtype=torch.float32:torch.as_tensor([r[k] for r in buf],dtype=dtype,device=a.device)
                actions=T('action',torch.long);old_logp=T('logp');learn=T('learn');proposal=T('proposal',torch.long)
                A=torch.as_tensor(adv,device=a.device);R=torch.as_tensor(ret,device=a.device)
                live=learn>0
                A=(A-A[live].mean())/(A[live].std(unbiased=False)+1e-6) if live.any() else A*0
                beta=.01+.04*.5*(1+math.cos(math.pi*iteration/a.updates));losses=[]
                for epoch in range(3):
                    for index in torch.randperm(len(buf),device=a.device).split(512):
                        d,v=net.dist(*(o[index] for o in obs));lp=d.log_prob(actions[index]);ratio=(lp-old_logp[index]).exp()
                        w=learn[index];den=w.sum().clamp(min=1)
                        policy_loss=-(torch.minimum(ratio*A[index],ratio.clamp(.9,1.1)*A[index])*w).sum()/den
                        value_loss=((v-R[index])**2).mean()
                        prior_loss=-(d.log_prob(proposal[index])*w).sum()/den
                        entropy=(d.entropy()*w).sum()/den
                        loss=policy_loss+.25*value_loss+beta*prior_loss-.008*entropy
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),.5);optimizer.step()
                        losses.append([float(policy_loss.detach()),float(value_loss.detach()),float(entropy.detach()),float(prior_loss.detach())])
                iteration+=1;steps+=int(learn.sum())
                distance=sum(float((v.detach().cpu()-initial_state[k]).square().sum()) for k,v in net.state_dict().items())**.5
                row=dict(iteration=iteration,agent_steps=steps,environment_steps=iteration*a.workers*a.rollout,minutes=(time.monotonic()-start)/60,
                         ppo_policy_loss=float(np.mean(losses,0)[0]),value_loss=float(np.mean(losses,0)[1]),entropy=float(np.mean(losses,0)[2]),
                         proposal_cross_entropy=float(np.mean(losses,0)[3]),parameter_change_l2=distance,completed_episodes=len(finished),
                         action_change_fraction=float(np.mean([r['action']!=r['proposal'] for r in buf if r['learn']])),
                         strict_success=float(np.mean([r['strict_success'] for r in finished])) if finished else None,
                         removal=float(np.mean([r['removal'] for r in finished])) if finished else None)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                save('policy.pt')
                if iteration==1 or iteration%10==0:save(f'policy_it{iteration}.pt')
            save('policy.pt');(a.out/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(iterations=iteration,agent_steps=steps)))
    finally:
        for p in pipes:
            try:p.send(None)
            except (BrokenPipeError,EOFError):pass
        for pr in procs:
            pr.join(timeout=5)
            if pr.is_alive():pr.terminate()

if __name__=='__main__':main()
