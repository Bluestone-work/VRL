"""EXP0061 finite predictive-PPO / same-budget PPO continuation studies."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import numpy as np
import torch
from marl.vascular_predictive_rl import PredictivePolicy, model_loss
from scripts.vascular_option_study import Episode, ROOT, DEV_BASE, sha


class PredictiveEpisode(Episode):
    def __init__(self,*args,wall_weight=4.,particle_weight=8.,**kwargs):
        super().__init__(*args,**kwargs)
        self.wall_weight=wall_weight;self.particle_weight=particle_weight
        self.wall_seconds=np.zeros(3);self.particle_counts=np.zeros(3)
        self.hazards=np.zeros((3,3),np.float32)
        original=self.env.step
        def step(command):
            spacing_before=self.spacing.pair_violation_s
            result=original(command); info=result[-1]
            self.wall_seconds[:self.n]+=np.asarray(info['wall_contact_s'])
            self.particle_counts[:self.n]+=np.asarray(info['particle_collision_events'])
            self.hazards[:self.n,0]=np.maximum(self.hazards[:self.n,0],np.asarray(info['wall_contact_s'])>0)
            self.hazards[:self.n,1]=np.maximum(self.hazards[:self.n,1],np.asarray(info['particle_collision_events'])>0)
            if self.spacing.pair_violation_s>spacing_before:self.hazards[:self.n,2]=1.
            return result
        self.env.step=step

    def step(self,action,capture=False):
        self.hazards[:]=0.
        self.wall_seconds[:]=0.;self.particle_counts[:]=0.
        obs,reward,done=super().step(action,capture=capture)
        reward-=(self.wall_weight-4.)*self.wall_seconds+(self.particle_weight-8.)*self.particle_counts
        return obs,reward,done


def worker(conn,wid,seed,round_id,horizon,wall_weight,particle_weight):
    torch.set_num_threads(1)
    anatomies=json.loads((ROOT/'configs/evaluation_splits.json').read_text())['anatomy_holdout_v1']['train']
    rng=np.random.default_rng([6100,round_id,seed,wid]); count=0
    def new():
        nonlocal count
        if count>=5000:raise RuntimeError('Training scene block exhausted')
        s=2820000000+round_id*1000000+seed*100000+wid*5000+count; count+=1
        return PredictiveEpisode(str(rng.choice(anatomies)),int(rng.integers(1,4)),s,horizon=horizon,
                                 wall_weight=wall_weight,particle_weight=particle_weight)
    ep=new(); conn.send(ep.observation())
    try:
        while True:
            ac=conn.recv()
            if ac is None:break
            ob,reward,done=ep.step(ac)
            # Preserve terminal observation BEFORE reset for the model target.
            target=ob[0][:,-1].copy(); active_next=ob[1].copy(); hazards=ep.hazards.copy()
            row=ep.metrics() if done else None
            if done:ep.env.close();ep=new();ob=ep.observation()
            conn.send((ob,reward,done,row,target,hazards,active_next))
    finally:ep.env.close();conn.close()


def gae(rewards,values,dones,bootstrap,gamma=.995,lam=.95):
    advantages=np.zeros_like(rewards); g=np.zeros_like(bootstrap)
    for t in reversed(range(len(rewards))):
        next_v=bootstrap if t==len(rewards)-1 else values[t+1]
        cont=1-dones[t]
        g=rewards[t]+gamma*next_v*cont-values[t]+gamma*lam*cont*g
        advantages[t]=g
    return advantages,advantages+values


def train(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(a.seed)
    p=PredictivePolicy(a.arm=='world_ppo')
    parent=torch.load(a.parent,map_location='cpu');p.load_parent(parent)
    torch.manual_seed(a.seed+6100)
    cfg=dict(vars(a),memory=True,central=True,use_world=p.use_world,parent_sha256=sha(a.parent),
             parent_steps=parent['steps'],parent_iterations=parent['iterations'],
             source_sha={name:sha(ROOT/name) for name in ['marl/vascular_predictive_rl.py','scripts/vascular_predictive_study.py',
                       'marl/vascular_option_rl.py','scripts/vascular_option_study.py']})
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    opt=torch.optim.Adam(p.parameters(),lr=a.lr);ctx=mp.get_context('spawn');pipes=[];procs=[]
    for w in range(a.workers):
        c,d=ctx.Pipe();pr=ctx.Process(target=worker,args=(d,w,a.seed,a.round,a.horizon,a.wall_weight,a.particle_weight));pr.start();d.close();pipes.append(c);procs.append(pr)
    def receive(c):
        if not c.poll(180):raise TimeoutError('Predictive worker timeout')
        return c.recv()
    start=time.monotonic();steps=0;iteration=0
    def save(name):
        tmp=out/(name+'.tmp');torch.save(dict(state=p.state_dict(),config=cfg,steps=steps,iterations=iteration,
                                             world_active=p.world_active),tmp);tmp.replace(out/name)
    try:
        ob=[receive(c) for c in pipes];p.world_active=False;save('initial.pt')
        with (out/'training.jsonl').open('a') as log,(out/'episodes.jsonl').open('a') as elog:
            while iteration<a.updates and time.monotonic()-start<a.minutes*60:
                p.world_active=p.use_world and iteration>=a.model_warmup
                X=[];M=[];AC=[];LP=[];V=[];RW=[];DN=[];NX=[];HZ=[];MM=[];finished=[]
                for _ in range(a.rollout):
                    x=np.stack([v[0] for v in ob]);m=np.stack([v[1] for v in ob])
                    with torch.no_grad():dist,val=p(torch.from_numpy(x),torch.from_numpy(m));ac=dist.sample();lp=dist.log_prob(ac)
                    for c,action in zip(pipes,ac.numpy()):c.send(action)
                    data=[receive(c) for c in pipes]
                    terminal=np.array([d[2] for d in data],np.float32)[:,None]
                    active_next=np.stack([d[6] for d in data])
                    X.append(x);M.append(m);AC.append(ac.numpy());LP.append(lp.numpy());V.append(val.numpy())
                    RW.append(np.stack([d[1] for d in data]));DN.append(np.maximum(terminal,1-active_next))
                    NX.append(np.stack([d[4] for d in data]));HZ.append(np.stack([d[5] for d in data]));MM.append(m*active_next)
                    ob=[d[0] for d in data];steps+=int(m.sum())*5
                    for d in data:
                        if d[3] is not None:finished.append(d[3]);elog.write(json.dumps(d[3])+'\n')
                with torch.no_grad():
                    _,boot=p(torch.from_numpy(np.stack([v[0] for v in ob])),torch.from_numpy(np.stack([v[1] for v in ob])))
                adv,ret=gae(np.array(RW),np.array(V),np.array(DN),boot.numpy(),gamma=a.gamma)
                T=lambda z:torch.from_numpy(np.asarray(z)).flatten(0,1)
                x,m,ac,lp0,A,R,nx,hz,mm=map(T,[X,M,AC,LP,adv,ret,NX,HZ,MM])
                valid=m.bool();A=(A-A[valid].mean())/(A[valid].std()+1e-8)
                # Last quarter of every rollout is excluded from fitting the world
                # model and used for next-observation/reward/risk diagnostics.
                train_model_mask=torch.ones_like(m)
                train_model_mask[int(.75*len(m)):]=0.
                losses=[];wloss=[]
                for _ in range(4):
                    for ids in torch.randperm(len(x)).split(128):
                        dist,val,pred=p(x[ids],m[ids],return_model=True)
                        ratio=(dist.log_prob(ac[ids])-lp0[ids]).exp()
                        obj=-torch.minimum(ratio*A[ids],ratio.clamp(.8,1.2)*A[ids])
                        loss=((obj+.5*(val-R[ids]).square()-a.entropy*dist.entropy())*m[ids]).sum()/m[ids].sum().clamp(min=1)
                        aux,_=model_loss(pred,ac[ids],nx[ids],T(RW)[ids],hz[ids],mm[ids]*train_model_mask[ids])
                        total=loss+(a.world_weight*aux if p.use_world else 0.)
                        opt.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),.5);opt.step()
                        losses.append(float(loss.detach()));wloss.append(float(aux.detach()))
                # This temporal slice is an online diagnostic, not independent OOD validation.
                ids=torch.arange(int(.75*len(m)),len(m))
                with torch.no_grad():
                    _,_,pred=p(x[ids],m[ids],return_model=True)
                    e,b,n,_,d=pred.shape
                    selected=pred.gather(-2,ac[ids][None,...,None,None].expand(e,b,n,1,d)).squeeze(-2).mean(0)
                    mask=mm[ids];den=mask.sum().clamp(min=1)
                    state_mse=float(((selected[...,:24]-nx[ids,...,:24]).square().mean(-1)*mask).sum()/den)
                    persistence=float(((x[ids,:,-1,:24]-nx[ids,...,:24]).square().mean(-1)*mask).sum()/den)
                    brier=float(((selected[...,25:].sigmoid()-hz[ids]).square().mean(-1)*mask).sum()/den)
                iteration+=1;save('latest.pt')
                row=dict(iteration=iteration,agent_control_steps=steps,seconds=time.monotonic()-start,
                    safe=float(np.mean([r['cluster_safe_success'] for r in finished])) if finished else None,
                    loss=float(np.mean(losses)),world_loss=float(np.mean(wloss)),state_mse=state_mse,
                    persistence_mse=persistence,hazard_brier=brier,world_active=p.world_active)
                log.write(json.dumps(row)+'\n');log.flush();elog.flush();print(json.dumps(row),flush=True)
            save('final.pt');(out/'DONE.json').write_text(json.dumps(dict(updates=iteration,requested_updates=a.updates,
                   budget_completed=iteration==a.updates,seconds=time.monotonic()-start,sha256=sha(out/'final.pt'))))
    finally:
        for c in pipes:
            try:c.send(None)
            except (EOFError,BrokenPipeError):pass
        for pr in procs:
            pr.join(timeout=5)
            if pr.is_alive():pr.terminate();pr.join()


def evaluate(a):
    torch.set_num_threads(1);p=None
    if a.mode!='baseline':
        ck=torch.load(a.checkpoint,map_location='cpu')
        p=PredictivePolicy(ck['config']['use_world']);p.load_state_dict(ck['state'],strict=True)
        p.world_active=ck['world_active'] and not a.disable_world;p.eval()
    order=json.loads((ROOT/'configs/evaluation_splits.json').read_text())['anatomy_order']
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);checkpoint_hash=sha(a.checkpoint) if p else None
    with out.open('a') as f:
        for anatomy in a.anatomies.split(','):
            for scene in range(a.scene_start,a.scene_start+a.scenes):
                seed=DEV_BASE+order.index(anatomy)*10000+scene
                ep=Episode(anatomy,a.clusters,seed,sensing=a.sensing)
                while not ep.done:
                    x,m=ep.observation()
                    if p is None:ac=np.full(3,a.fixed_option,int)
                    else:
                        with torch.no_grad():dist,_=p(torch.from_numpy(x[None]),torch.from_numpy(m[None]));ac=dist.logits.argmax(-1)[0].numpy()
                    ep.step(ac,capture=a.capture)
                row=ep.metrics();row.update(method=a.arm,checkpoint_sha256=checkpoint_hash,world_active=p.world_active if p else False)
                f.write(json.dumps(row)+'\n');f.flush()
                if a.capture:
                    geometry=dict(points=ep.env.transport.points.tolist(),ends=ep.env.transport.ends.tolist(),
                        healthy_radius=ep.env.flow_model.healthy_radius_mm.tolist(),clot_positions=ep.env.clot_positions_mm.tolist(),
                        initial_mass=ep.env.initial_mass.tolist())
                    out.with_suffix('.trace.json').write_text(json.dumps(dict(metrics=row,manifest=ep.manifest,trace=ep.trace,geometry=geometry),default=str))
                ep.env.close()


def main():
    os.sched_setaffinity(0,os.sched_getaffinity(0)-{6,7})
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['train','eval','baseline'])
    p.add_argument('--out',required=True);p.add_argument('--arm',default='world_ppo');p.add_argument('--parent')
    p.add_argument('--checkpoint');p.add_argument('--seed',type=int,default=0);p.add_argument('--round',type=int,default=0)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--updates',type=int,default=120)
    p.add_argument('--minutes',type=float,default=40);p.add_argument('--rollout',type=int,default=128)
    p.add_argument('--horizon',type=float,default=300.);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--gamma',type=float,default=.995);p.add_argument('--entropy',type=float,default=.01)
    p.add_argument('--world-weight',type=float,default=.1);p.add_argument('--model-warmup',type=int,default=8)
    p.add_argument('--wall-weight',type=float,default=12.);p.add_argument('--particle-weight',type=float,default=12.)
    p.add_argument('--anatomies',default='mca_m1_lvo,ica_terminus_t,renal_artery');p.add_argument('--scenes',type=int,default=4)
    p.add_argument('--scene-start',type=int,default=0);p.add_argument('--clusters',type=int,default=3)
    p.add_argument('--sensing',choices=['noise','image'],default='noise');p.add_argument('--disable-world',action='store_true')
    p.add_argument('--capture',action='store_true')
    p.add_argument('--fixed-option',type=int,default=0)
    a=p.parse_args();train(a) if a.mode=='train' else evaluate(a)


if __name__=='__main__':main()
