"""Off-policy SAC from genuine image-based vascular rollouts, train anatomies only."""
import os
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,os.sched_getaffinity(0)-{6,7})
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='1'
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from marl.image_flow_sac import ControlEpisode,ImageActor,ImageCritic,ImageFlowController,observation_tensors
ROOT=Path(__file__).resolve().parents[1]


def worker(wid,pipe,cfg,folder):
    torch.set_num_threads(1);torch.manual_seed(cfg['seed']*1000+wid)
    from marl.deployable_sensing import DeployableConfig
    from scripts.train_lysis_local import route_remaining
    from scripts.export_vrl_v3_all_gifs import geometry
    rng=np.random.default_rng([9500,cfg['seed'],wid]);actor=ImageActor();serial=0
    train=json.load(open(ROOT/'configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']
    def prepare(ep,ctl):
        est=ep.observe();target=ep.plan_targets_now(est);rule=ep.ctl.act(target,est);hold=ep.hold(est)
        return est,target,hold,ctl.observe(ep,est,target,rule,hold)
    def reset():
        nonlocal serial
        while True:
            seed=cfg['scene_seed_base']+cfg['seed']*1000000+wid*10000+serial;serial+=1
            an=str(rng.choice(train));n=int(rng.integers(1,4));flow=float(rng.choice([.025,.05,.1]))
            delay=int(rng.integers(1,4));s=0. if rng.random()<.2 else float(rng.uniform(.25,1.25))
            try:ep=ControlEpisode(n,an,seed,flow_inlet_mm_s=flow,sense_cfg=DeployableConfig(latency_steps=delay),variation=s or None);break
            except (ValueError,RuntimeError):continue
        ctl=ImageFlowController(ep,actor=actor,stochastic=True,config=cfg['controller'])
        ep.meta=dict(anatomy=an,n=n,seed=seed,flow=flow,latency=delay,s=s,training_seed=cfg['seed'])
        ep.trace=dict(t=[0.],pos=[ep.env.positions_mm[:n].copy()],mass=[ep.env.masses.copy()],wall=[0.],removal=[0.],sent=[ep.sent_world.copy()])
        ep.images=[];ep.geo=geometry(ep);ep.versions=set();ep.reward_sum=0.;ep.prepared=prepare(ep,ctl)
        return ep,ctl
    ep=ctl=None
    try:
        ep,ctl=reset()
        while True:
            msg=pipe.recv()
            if msg is None:break
            state,iteration=msg;actor.load_state_dict(state);transitions=[];finished=[]
            for step in range(cfg['rollout']):
                est,target,hold,obs=ep.prepared
                with torch.no_grad():actions,_=actor.sample(*observation_tensors(obs))
                action=actions.numpy();local=ctl.execute(ep,est,ctl.decode(action))
                old=np.array([route_remaining(ep.ctl,i,ep.env.positions_mm[i]) if target[i]>=0 else 0. for i in range(ep.n)])
                done,out=ep.step(est,local,hold)
                new=np.array([route_remaining(ep.ctl,i,ep.env.positions_mm[i]) if target[i]>=0 else 0. for i in range(ep.n)])
                progress=np.clip(old-new,-.2,.2)*(np.asarray(target)>=0)*(~hold)
                reward=.4*progress+100.*out['agent_removed']/ep.initial-25.*out['wall']-30.*out['lost']-.004*out['active_before']
                # These distances are reward labels only, never actor features.
                for i,t in enumerate(target):
                    if t>=0 and not hold[i] and out['active_before'][i]:
                        distance=np.linalg.norm(ep.env.positions_mm[i]-ep.env.clot_positions_mm[t])
                        reward[i]+=.015*max(0.,1-distance/.15)
                if ep.n>1:
                    D=np.linalg.norm(ep.env.positions_mm[:ep.n,None]-ep.env.positions_mm[None,:ep.n],axis=-1)+np.eye(ep.n)*99
                    reward-=2.*(D<ep.d_min).any(1)
                row=None
                if done:
                    row=ep.row('image_SAC_training',**ctl.summary(),**ep.authority)
                    reward+=10.*float(row['strict_success'])
                    if not row['task_success']:reward-=3.
                next_prepared=prepare(ep,ctl);next_obs=next_prepared[3]
                for i in range(ep.n):
                    if not out['active_before'][i]:continue
                    transitions.append(dict(obs=obs[i],next=next_obs[i],action=action[i],reward=float(reward[i]),
                        terminal=bool(out['terminated'] or not ep.env.active[i]),
                        observed_velocity=ctl.F[i]@next_prepared[0].vel[i]))
                ep.reward_sum+=float(reward.sum());ep.versions.add(iteration)
                tr=ep.trace;tr['t'].append(float(ep.env.elapsed_s));tr['pos'].append(ep.env.positions_mm[:ep.n].copy())
                tr['mass'].append(ep.env.masses.copy());tr['wall'].append(float(np.sum(out['wall'])))
                tr['removal'].append(1-float(ep.env.masses.sum())/ep.initial);tr['sent'].append(ep.sent_world.copy())
                if ep.steps%10==0:ep.images.append(np.asarray(ep.sensor.crops,np.float16).copy())
                ep.prepared=next_prepared
                if done:
                    row.update(ep.meta,return_sum=ep.reward_sum,policy_iterations=sorted(ep.versions));finished.append(row)
                    path=Path(folder)/'training_traces';path.mkdir(exist_ok=True)
                    np.savez_compressed(path/f'{ep.meta["anatomy"]}_N{ep.n}_seed{ep.seed}.npz',
                        **{k:np.asarray(v) for k,v in tr.items()},image_samples=np.asarray(ep.images),
                        geometry=json.dumps(ep.geo),metrics=json.dumps(row),meta=json.dumps(ep.meta))
                    ep.close();ep,ctl=reset()
            pipe.send(dict(transitions=transitions,finished=finished))
    except BaseException:
        try:pipe.send(dict(error=traceback.format_exc()))
        except (BrokenPipeError,EOFError):pass
    finally:
        if ep:ep.close()
        pipe.close()


class Replay:
    def __init__(self,capacity):self.capacity=capacity;self.rows=[];self.cursor=0
    def add(self,rows):
        for row in rows:
            if len(self.rows)<self.capacity:self.rows.append(row)
            else:self.rows[self.cursor]=row
            self.cursor=(self.cursor+1)%self.capacity
    def sample(self,n,rng):return [self.rows[i] for i in rng.integers(0,len(self.rows),n)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--iterations',type=int,default=240)
    p.add_argument('--rollout',type=int,default=96);p.add_argument('--gradient-steps',type=int,default=12)
    p.add_argument('--batch',type=int,default=256);p.add_argument('--capacity',type=int,default=100000)
    p.add_argument('--device',default='cuda:0');p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--alpha',type=float,default=.002)
    p.add_argument('--scene-seed-base',type=int,default=2141000000)
    p.add_argument('--controller-json',type=Path);a=p.parse_args();os.chdir(ROOT);a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    actor=ImageActor().to(a.device);critic=ImageCritic().to(a.device);target=copy.deepcopy(critic).eval()
    ao=torch.optim.Adam(actor.parameters(),lr=a.lr);qo=torch.optim.Adam(critic.parameters(),lr=a.lr)
    controller=json.load(open(a.controller_json)) if a.controller_json else {}
    cfg={k:v for k,v in vars(a).items() if k not in ('out','controller_json')};cfg.update(controller=controller,action_mode='residual',
        algorithm='SAC continuous local navigation from biplane images and aligned measured history',gamma=.995,tau=.005,
        image_input='current and previous delayed 32x32 biplane crops',
        cardiac_input='frequency/phase inferred from timestamp-aligned image motion; no true cardiac phase',
        training_anatomies=json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train'])
    cfg['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
        ('marl/image_flow_sac.py','scripts/train_image_flow_sac.py','marl/image_sensing.py','scripts/benchmark_lysis.py')}
    (a.out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    init={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()};replay=Replay(a.capacity);steps=0;gradients=0;iteration=0
    def save(name):
        payload=dict(actor={k:v.detach().cpu() for k,v in actor.state_dict().items()},critic=critic.state_dict(),
                     config=cfg,iterations=iteration,gradient_updates=gradients,agent_steps=steps,trained_with_rl=gradients>0)
        tmp=a.out/(name+'.tmp');torch.save(payload,tmp);tmp.replace(a.out/name)
    save('initial.pt');context=mp.get_context('spawn');pipes=[];workers=[]
    for wid in range(a.workers):
        parent,child=context.Pipe();proc=context.Process(target=worker,args=(wid,child,cfg,str(a.out)));proc.start();child.close()
        pipes.append(parent);workers.append(proc)
    start=time.monotonic()
    try:
        with (a.out/'training.jsonl').open('w') as log,(a.out/'episodes.jsonl').open('w') as elog:
            while iteration<a.iterations:
                state={k:v.detach().cpu() for k,v in actor.state_dict().items()}
                for pipe in pipes:pipe.send((state,iteration))
                finished=[]
                for wid,pipe in enumerate(pipes):
                    if not pipe.poll(600):raise TimeoutError(f'SAC rollout worker {wid} timed out')
                    data=pipe.recv()
                    if 'error' in data:raise RuntimeError(data['error'])
                    replay.add(data['transitions']);steps+=len(data['transitions']);finished.extend(data['finished'])
                for row in finished:elog.write(json.dumps(row,ensure_ascii=False)+'\n')
                elog.flush();losses=[]
                if iteration>=4:
                    for update in range(a.gradient_steps):
                        rows=replay.sample(a.batch,rng);obs=observation_tensors([r['obs'] for r in rows],a.device)
                        nxt=observation_tensors([r['next'] for r in rows],a.device)
                        actions=torch.as_tensor(np.stack([r['action'] for r in rows]),device=a.device)
                        reward=torch.tensor([r['reward'] for r in rows],device=a.device);terminal=torch.tensor([r['terminal'] for r in rows],device=a.device)
                        with torch.no_grad():
                            na,nlp=actor.sample(*nxt);q1,q2=target(*nxt,na)
                            y=reward+cfg['gamma']*(~terminal)*(torch.minimum(q1,q2)-a.alpha*nlp)
                        q1,q2=critic(*obs,actions);qloss=nn_loss(q1,y)+nn_loss(q2,y)
                        if not torch.isfinite(qloss):raise FloatingPointError('Nonfinite SAC critic loss')
                        qo.zero_grad();qloss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),5.);qo.step()
                        for param in critic.parameters():param.requires_grad_(False)
                        act,lp=actor.sample(*obs);q1,q2=critic(*obs,act);aloss=(a.alpha*lp-torch.minimum(q1,q2)).mean()
                        if not torch.isfinite(aloss):raise FloatingPointError('Nonfinite SAC actor loss')
                        ao.zero_grad();aloss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);ao.step()
                        for param in critic.parameters():param.requires_grad_(True)
                        if not torch.isfinite(qloss+aloss):raise FloatingPointError('Nonfinite SAC loss')
                        with torch.no_grad():
                            for t,c in zip(target.parameters(),critic.parameters()):t.lerp_(c,cfg['tau'])
                        losses.append([float(qloss.detach()),float(aloss.detach()),float(lp.detach().mean())]);gradients+=1
                iteration+=1
                distance=sum(float((v.detach().cpu()-init[k]).square().sum()) for k,v in actor.state_dict().items())**.5
                row=dict(iteration=iteration,agent_steps=steps,gradient_updates=gradients,replay_size=len(replay.rows),
                    actor_parameter_change_l2=distance,minutes=(time.monotonic()-start)/60,completed_episodes=len(finished),
                    losses=np.mean(losses,axis=0).tolist() if losses else None,
                    strict_success=float(np.mean([r['strict_success'] for r in finished])) if finished else None,
                    removal=float(np.mean([r['removal'] for r in finished])) if finished else None)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                if iteration%20==0:save(f'policy_it{iteration}.pt');save('policy.pt')
            save('policy.pt');(a.out/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(iterations=iteration,gradient_updates=gradients,agent_steps=steps)))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (BrokenPipeError,EOFError):pass
        for proc in workers:
            proc.join(timeout=3)
            if proc.is_alive():proc.terminate()


def nn_loss(pred,target):return torch.nn.functional.smooth_l1_loss(pred,target)


if __name__=='__main__':main()
