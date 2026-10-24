"""Off-policy SAC from genuine image-based vascular rollouts, train anatomies only."""
import os
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,os.sched_getaffinity(0)-{6,7})
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='1'
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import math
import random
import cloudpickle
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from marl.image_flow_sac import ControlEpisode,observation_tensors
from marl.sac_speed_control import SpeedActor as ImageActor,SpeedCritic as ImageCritic,SpeedOnlySAC
from marl.sac_local_envelope import EnvelopedSpeedSAC as ImageFlowController
ROOT=Path(__file__).resolve().parents[1]


def worker(wid,pipe,cfg,folder,resume_state=None):
    torch.set_num_threads(1);torch.manual_seed(cfg['seed']*1000+wid)
    from marl.deployable_sensing import DeployableConfig
    from scripts.train_lysis_local import route_remaining
    from scripts.export_vrl_v3_all_gifs import geometry
    from marl.td3_local import TD3SpeedActor,TD3Actor,TD3Critic
    from marl.image_flow_sac import ImageActor as VectorActor, ImageFlowController as VectorController
    rng=np.random.default_rng([9500,cfg['seed'],wid]);actor=(TD3Actor(zero_init=cfg.get('zero_init',True)) if cfg.get('algorithm_kind')=='td3' and cfg.get('action_space')=='vector' else VectorActor('direct',zero_init=cfg.get('zero_init',True)) if cfg.get('action_space')=='vector' else TD3SpeedActor() if cfg.get('algorithm_kind')=='td3' else ImageActor());serial=0
    train=json.load(open(ROOT/'configs/evaluation_splits.json'))['anatomy_holdout_v1']['train']
    def prepare(ep,ctl):
        est=ep.observe();target=ep.plan_targets_now(est);rule=ep.ctl.act(target,est);hold=ep.hold(est)
        return est,target,hold,ctl.observe(ep,est,target,rule,hold)
    def reset():
        nonlocal serial
        while True:
            seed=cfg['scene_seed_base']+cfg['seed']*1000000+wid*10000+serial;serial+=1
            an=str(rng.choice(train));n=int(rng.integers(1,4));flow=float(rng.choice([.025,.05,.1]))
            delay=int(rng.integers(1,3));s=0.;obstacles=int(cfg.get('dynamic_obstacle_count',0))
            if cfg.get('curriculum'):
                frac=float(cfg.get('curriculum_iteration',0))/max(float(cfg.get('iterations',1)),1.)
                if frac < .2: n,flow,delay,obstacles=1,.05,1,0
                elif frac < .5: n,flow,delay,obstacles=1,float(rng.choice([.025,.05])),1,4
                else: n,flow,delay,obstacles=int(rng.integers(1,4)),float(rng.choice([.025,.05,.1])),int(rng.integers(1,3)),int(cfg.get('dynamic_obstacle_count',16))
            try:ep=ControlEpisode(n,an,seed,flow_inlet_mm_s=flow,sense_cfg=DeployableConfig(latency_steps=delay),variation=s or None,
                                  dynamic_obstacle_count=obstacles);break
            except (ValueError,RuntimeError):continue
        if cfg.get('action_space')=='vector':
            ctl=VectorController(ep,actor=actor,stochastic=True,config=cfg['controller'])
        elif cfg.get('algorithm_kind')=='td3':
            from marl.sac_speed_control import SpeedOnlySAC
            ctl=SpeedOnlySAC(ep,actor=actor,stochastic=True,config=cfg['controller'])
        else:
            ctl=ImageFlowController(ep,actor=actor,stochastic=True,config=cfg['controller'])
        ep.meta=dict(anatomy=an,n=n,seed=seed,flow=flow,latency=delay,s=s,training_seed=cfg['seed'])
        ep.trace=dict(t=[0.],pos=[ep.env.positions_mm[:n].copy()],mass=[ep.env.masses.copy()],wall=[0.],removal=[0.],sent=[ep.sent_world.copy()])
        ep.images=[];ep.geo=geometry(ep);ep.versions=set();ep.reward_sum=0.;ep.prepared=prepare(ep,ctl)
        return ep,ctl
    ep=ctl=None
    try:
        if resume_state is None:
            ep,ctl=reset()
        else:
            saved=cloudpickle.loads(resume_state)
            ep,ctl,serial=saved['episode'],saved['controller'],saved['serial']
            actor=ctl.actor
            rng.bit_generator.state=saved['rng']
            torch.set_rng_state(saved['torch_rng'])
            np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        while True:
            msg=pipe.recv()
            if msg is None:break
            if isinstance(msg,str) and msg=='snapshot':
                pipe.send(cloudpickle.dumps(dict(episode=ep,controller=ctl,serial=serial,
                    rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                    numpy_rng=np.random.get_state(),python_rng=random.getstate())))
                continue
            state,iteration=msg;actor.load_state_dict(state);cfg['curriculum_iteration']=iteration;transitions=[];finished=[];exposure={}
            for step in range(cfg['rollout']):
                key=ep.meta['seed']
                if key not in exposure:exposure[key]=dict(ep.meta,physical_steps=0)
                exposure[key]['physical_steps']+=1
                est,target,hold,obs=ep.prepared
                with torch.no_grad():
                    tensors=observation_tensors(obs)
                    actions=(actor(*tensors) if cfg.get('algorithm_kind')=='td3' and cfg.get('action_space')=='vector' else actor.sample(*tensors)[0])
                action=actions.numpy();local=ctl.execute(ep,est,ctl.decode(action))
                old=np.array([route_remaining(ep.ctl,i,ep.env.positions_mm[i]) if target[i]>=0 else 0. for i in range(ep.n)])
                done,out=ep.step(est,local,hold)
                new=np.array([route_remaining(ep.ctl,i,ep.env.positions_mm[i]) if target[i]>=0 else 0. for i in range(ep.n)])
                progress=np.clip(old-new,-.2,.2)*(np.asarray(target)>=0)*(~hold)
                reward=4.*progress+100.*out['agent_removed']/ep.initial-25.*out['wall']-30.*out['lost']-.004*out['active_before']
                # These distances are reward labels only, never actor features.
                for i,t in enumerate(target):
                    if t>=0 and not hold[i] and out['active_before'][i]:
                        distance=np.linalg.norm(ep.env.positions_mm[i]-ep.env.clot_positions_mm[t])
                        reward[i]+=.015*max(0.,1-distance/.15)
                if ep.n>1:
                    D=np.linalg.norm(ep.env.positions_mm[:ep.n,None]-ep.env.positions_mm[None,:ep.n],axis=-1)+np.eye(ep.n)*99
                    reward-=2.*(D<ep.d_min).any(1)
                if cfg.get('team_reward'):
                    spacing=float(ep.spacing.summary()['spacing_violation_pair_s'])
                    ds=spacing-getattr(ep,'reward_spacing',0.);ep.reward_spacing=spacing
                    shared=(10.*out['removed']-2.*float(np.sum(out['wall']))-2.*ds
                            -10.*float(np.sum(out['lost']))-.01*float(ep.info['step_duration_s'])
                            -cfg['obstacle_penalty']*float(np.sum(ep.info['particle_contact_s'])))
                    reward=np.full(ep.n,shared)
                row=None
                if done:
                    row=ep.row('image_SAC_training',**ctl.summary(),**ep.authority)
                    reward+=(20. if cfg.get('team_reward') else 10.)*float(row['strict_success'])
                    if not cfg.get('team_reward') and not row['task_success']:reward-=3.
                next_prepared=prepare(ep,ctl);next_obs=next_prepared[3]
                for i in range(ep.n):
                    if not out['active_before'][i]:continue
                    transitions.append(dict(obs=obs[i],next=next_obs[i],action=action[i],reward=float(reward[i]),
                        learn=bool(target[i]>=0 and not hold[i]),
                        next_learn=bool(next_prepared[1][i]>=0 and not next_prepared[2][i]),
                        terminal=bool(done or not ep.env.active[i]),
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
            pipe.send(dict(transitions=transitions,finished=finished,exposure=list(exposure.values())))
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
    p.add_argument('--env-steps',type=int,help='Total physical environment steps, rounded up to a complete rollout batch')
    p.add_argument('--resume',type=Path,help='Full resume.pt only; policy-only checkpoints are rejected')
    p.add_argument('--resume-every',type=int,default=100,help='Full state snapshot interval in iterations')
    p.add_argument('--milestone-steps',default='100000,250000,500000,1000000')
    p.add_argument('--rollout',type=int,default=96);p.add_argument('--gradient-steps',type=int,default=12)
    p.add_argument('--batch',type=int,default=256);p.add_argument('--capacity',type=int,default=100000)
    p.add_argument('--device',default='cuda:0');p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--alpha',type=float,default=.002)
    p.add_argument('--scene-seed-base',type=int,default=2191000000)
    p.add_argument('--dynamic-obstacles',action='store_true');p.add_argument('--dynamic-obstacle-count',type=int,default=8)
    p.add_argument('--obstacle-features',action='store_true')
    p.add_argument('--algorithm-kind',choices=['sac','td3'],default='sac')
    p.add_argument('--zero-init',action='store_true',default=False,help='Use zero output initialization; default is independent random initialization')
    p.add_argument('--curriculum',action='store_true',help='Pure-RL difficulty curriculum; no Parent action supervision')
    p.add_argument('--action-space',choices=['scalar','vector'],default='scalar')
    p.add_argument('--auto-alpha',action='store_true')
    p.add_argument('--target-entropy',type=float,default=None,
                   help='Differential entropy of normalized tanh actions; defaults to minus action dimension')
    p.add_argument('--alpha-lr',type=float,default=1e-4)
    p.add_argument('--team-reward',action='store_true')
    p.add_argument('--obstacle-penalty',type=float,default=2.)
    p.add_argument('--forecast-horizon',type=float,default=1.)
    p.add_argument('--critic-warmup-iterations',type=int,default=32)
    p.add_argument('--controller-json',type=Path);a=p.parse_args()
    if a.auto_alpha and a.algorithm_kind!='sac':p.error('automatic entropy temperature is SAC only')
    if a.alpha<=0:p.error('alpha must be positive')
    if min(a.workers,a.rollout,a.resume_every,a.capacity,a.batch,a.gradient_steps)<=0:p.error('training counts must be positive')
    if a.env_steps is not None:
        if a.env_steps<=0:p.error('env-steps must be positive')
        a.iterations=math.ceil(a.env_steps/(a.workers*a.rollout))
    milestones=sorted(set(int(x) for x in a.milestone_steps.split(',') if x))
    restored=torch.load(a.resume,map_location='cpu',weights_only=False) if a.resume else None
    if restored is not None and restored.get('resume_format')!=1:p.error('Not a full-state checkpoint; cannot resume without replay and workers')
    os.chdir(ROOT);a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    from marl.td3_local import TD3SpeedActor,TD3Actor,TD3Critic
    from marl.sac_speed_control import SpeedCritic
    from marl.image_flow_sac import ImageActor as VectorActor, ImageCritic as VectorCritic
    actor=(TD3Actor(zero_init=a.zero_init) if a.algorithm_kind=='td3' and a.action_space=='vector' else VectorActor('direct',zero_init=a.zero_init) if a.action_space=='vector' else TD3SpeedActor() if a.algorithm_kind=='td3' else ImageActor()).to(a.device)
    critic=(TD3Critic() if a.algorithm_kind=='td3' and a.action_space=='vector' else VectorCritic() if a.action_space=='vector' else SpeedCritic()).to(a.device)
    target=copy.deepcopy(critic).eval();target_actor=copy.deepcopy(actor).eval()
    ao=torch.optim.Adam(actor.parameters(),lr=a.lr);qo=torch.optim.Adam(critic.parameters(),lr=a.lr)
    log_alpha=torch.tensor(np.log(a.alpha),dtype=torch.float32,device=a.device,requires_grad=a.auto_alpha)
    alpha_optimizer=torch.optim.Adam([log_alpha],lr=a.alpha_lr) if a.auto_alpha else None
    target_entropy=a.target_entropy if a.target_entropy is not None else (-3. if a.action_space=='vector' else -1.)
    controller=json.load(open(a.controller_json)) if a.controller_json else {}
    controller['station_keeping']=False
    controller.setdefault('speed_logit_gain',4.)
    controller.setdefault('transit_logit_gain',.08)
    controller.setdefault('contact_logit_gain',4.)
    controller.setdefault('envelope_outer_mm',.35)
    controller.setdefault('envelope_inner_mm',.12)
    controller['zero_reference_policy']='preserve original stationary command'
    controller['action_parameterization']='scalar bounded log-odds speed shift along the original reference direction'
    cfg={k:v for k,v in vars(a).items() if k not in ('out','controller_json','resume')};cfg['dynamic_obstacle_count']=a.dynamic_obstacle_count if a.dynamic_obstacles else 0
    cfg['resume_from']=str(a.resume.resolve()) if a.resume else None
    controller['obstacle_features']=bool(a.obstacle_features)
    controller['forecast_horizon']=a.forecast_horizon
    cfg.update(controller=controller,action_mode='speed_relative_logit',
        terminal_boundary='finite benchmark episode, including the 300-second deadline',
        algorithm='Finite-horizon critic-warmed one-dimensional distance-enveloped SAC with small transit adaptation, stronger contact control, original TPG/parking and biplane history',gamma=.9995,tau=.005,
        image_input='current and previous delayed 32x32 biplane crops',
        cardiac_input='frequency/phase inferred from timestamp-aligned image motion; no true cardiac phase',
        training_anatomies=json.load(open('configs/evaluation_splits.json'))['anatomy_holdout_v1']['train'])
    cfg['algorithm']=a.algorithm_kind.upper()+' scalar speed, same action envelope; not full-vector navigation'
    if a.action_space=='vector':
        controller['action_scales']=[.2,.06,.06]
        controller['action_parameterization']='route-frame axial/normal/binormal residual on Parent; unit-ball clipping and original safety filters'
        controller['zero_reference_policy']='residual allowed except TPG wait and completed targets'
        cfg['action_mode']='direct'
        cfg['algorithm']=a.algorithm_kind.upper()+' three-dimensional bounded route-frame action from independent random initialization; no Parent warm start'
        cfg['zero_init']=bool(a.zero_init)
    cfg['reward_protocol']='shared clearance/cost increments; shaping disabled' if a.team_reward else 'legacy scalar SAC'
    cfg.update(target_entropy=target_entropy,
               entropy_protocol='normalized tanh action density; current/next forced-wait entropy masked; v2')
    cfg['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
        ('marl/image_flow_sac.py','marl/sac_safe_local.py','marl/sac_speed_control.py','marl/sac_local_envelope.py','scripts/train_finite_image_flow_sac.py','marl/image_sensing.py','scripts/benchmark_lysis.py')}
    (a.out/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    init={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()};replay=Replay(a.capacity);steps=0;gradients=0;actor_gradients=0;iteration=0
    if restored:
        ignored={'iterations','env_steps','resume_from','resume_every','milestone_steps','source_sha256'}
        mismatches=[k for k in cfg if k not in ignored and cfg[k]!=restored['config'].get(k)]
        if mismatches:raise ValueError(f'Resume configuration mismatch: {mismatches}')
        if cfg['source_sha256']!=restored['config']['source_sha256']:raise ValueError('Source changed since snapshot; exact continuation rejected')
        actor.load_state_dict(restored['actor']);critic.load_state_dict(restored['critic'])
        target.load_state_dict(restored['target_critic']);target_actor.load_state_dict(restored['target_actor'])
        ao.load_state_dict(restored['actor_optimizer']);qo.load_state_dict(restored['critic_optimizer'])
        with torch.no_grad():log_alpha.copy_(restored['log_alpha'].to(a.device))
        if alpha_optimizer:alpha_optimizer.load_state_dict(restored['alpha_optimizer'])
        replay.rows=restored['replay_rows'];replay.cursor=restored['replay_cursor']
        steps=restored['agent_steps'];gradients=restored['gradient_updates'];actor_gradients=restored['actor_gradient_updates'];iteration=restored['iterations']
        init=restored['initial_actor'];rng.bit_generator.state=restored['numpy_generator']
        torch.set_rng_state(restored['torch_rng']);np.random.set_state(restored['numpy_rng']);random.setstate(restored['python_rng'])
        if restored['cuda_rng'] is not None:torch.cuda.set_rng_state_all(restored['cuda_rng'])
        if iteration>=a.iterations:raise ValueError('Requested total budget must exceed restored iteration')
    def save(name,full=False):
        payload=dict(actor={k:v.detach().cpu() for k,v in actor.state_dict().items()},critic=critic.state_dict(),
                     config=cfg,iterations=iteration,gradient_updates=gradients,agent_steps=steps,actor_gradient_updates=actor_gradients,trained_with_rl=actor_gradients>0)
        payload.update(log_alpha=log_alpha.detach().cpu(),actor_optimizer=ao.state_dict(),
                       critic_optimizer=qo.state_dict(),target_critic=target.state_dict(),target_actor=target_actor.state_dict(),
                       alpha_optimizer=alpha_optimizer.state_dict() if alpha_optimizer else None)
        payload['environment_steps']=iteration*a.workers*a.rollout
        if full:
            for pipe in pipes:pipe.send('snapshot')
            worker_states=[]
            for pipe in pipes:
                if not pipe.poll(600):raise TimeoutError('Worker snapshot timeout')
                snapshot=pipe.recv()
                if not isinstance(snapshot,bytes):raise RuntimeError(f'Worker snapshot failed: {snapshot}')
                worker_states.append(snapshot)
            payload.update(resume_format=1,worker_states=worker_states,replay_rows=replay.rows,replay_cursor=replay.cursor,
                initial_actor=init,numpy_generator=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
        tmp=a.out/(name+'.tmp');torch.save(payload,tmp);tmp.replace(a.out/name)
    save('initial.pt');context=mp.get_context('spawn');pipes=[];workers=[]
    for wid in range(a.workers):
        parent,child=context.Pipe();proc=context.Process(target=worker,args=(wid,child,cfg,str(a.out),restored['worker_states'][wid] if restored else None));proc.start();child.close()
        pipes.append(parent);workers.append(proc)
    start=time.monotonic()
    try:
        with (a.out/'training.jsonl').open('w') as log,(a.out/'episodes.jsonl').open('w') as elog,(a.out/'exposure.jsonl').open('w') as xlog:
            while iteration<a.iterations:
                state={k:v.detach().cpu() for k,v in actor.state_dict().items()}
                for pipe in pipes:pipe.send((state,iteration))
                finished=[]
                for wid,pipe in enumerate(pipes):
                    if not pipe.poll(600):raise TimeoutError(f'SAC rollout worker {wid} timed out')
                    data=pipe.recv()
                    if 'error' in data:raise RuntimeError(data['error'])
                    replay.add(data['transitions']);steps+=len(data['transitions']);finished.extend(data['finished'])
                    for exposure in data['exposure']:xlog.write(json.dumps(dict(exposure,iteration=iteration+1))+'\n')
                xlog.flush()
                for row in finished:elog.write(json.dumps(row,ensure_ascii=False)+'\n')
                elog.flush();losses=[];diagnostics=[]
                if iteration>=4:
                    for update in range(a.gradient_steps):
                        rows=replay.sample(a.batch,rng);obs=observation_tensors([r['obs'] for r in rows],a.device)
                        nxt=observation_tensors([r['next'] for r in rows],a.device)
                        actions=torch.as_tensor(np.stack([r['action'] for r in rows]),device=a.device)
                        reward=torch.tensor([r['reward'] for r in rows],device=a.device);terminal=torch.tensor([r['terminal'] for r in rows],device=a.device)
                        next_learn=torch.tensor([r['next_learn'] for r in rows],device=a.device,dtype=torch.float32)
                        temperature=log_alpha.exp().detach()
                        with torch.no_grad():
                            if a.algorithm_kind=='td3':
                                na=(target_actor(*nxt) if a.action_space=='vector' else target_actor.sample(*nxt,deterministic=True)[0])
                                na=(na+(.2*torch.randn_like(na)).clamp(-.5,.5)).clamp(-1.,1.)
                                nlp=torch.zeros_like(reward)
                            else:na,nlp=actor.sample(*nxt)
                            q1,q2=target(*nxt,na)
                            y=reward+cfg['gamma']*(~terminal)*(torch.minimum(q1,q2)-temperature*nlp*next_learn)
                        q1,q2=critic(*obs,actions);qloss=nn_loss(q1,y)+nn_loss(q2,y)
                        td_error=((q1-y).abs()+(q2-y).abs()).mean().detach()/2
                        replay_q=torch.minimum(q1,q2).mean().detach()
                        q_gap=(q1-q2).abs().mean().detach()
                        if not torch.isfinite(qloss):raise FloatingPointError('Nonfinite SAC critic loss')
                        qo.zero_grad();qloss.backward();qgrad=torch.nn.utils.clip_grad_norm_(critic.parameters(),5.);qo.step()
                        for param in critic.parameters():param.requires_grad_(False)
                        act,lp=actor.sample(*obs,deterministic=a.algorithm_kind=='td3');q1,q2=critic(*obs,act)
                        learn=torch.tensor([r['learn'] for r in rows],device=a.device,dtype=torch.float32)
                        loss_per=temperature*lp-torch.minimum(q1,q2) if a.algorithm_kind=='sac' else -q1
                        aloss=(loss_per*learn).sum()/learn.sum().clamp(min=1)
                        if not torch.isfinite(aloss):raise FloatingPointError('Nonfinite SAC actor loss')
                        update_actor=iteration>=a.critic_warmup_iterations and (a.algorithm_kind=='sac' or gradients%2==0)
                        agrad=torch.tensor(0.)
                        if update_actor and learn.sum()>0:
                            ao.zero_grad();aloss.backward();agrad=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);ao.step();actor_gradients+=1
                            if alpha_optimizer:
                                alpha_loss=entropy_temperature_loss(log_alpha,lp,target_entropy,learn)
                                alpha_optimizer.zero_grad();alpha_loss.backward();alpha_optimizer.step()
                                if not torch.isfinite(log_alpha.exp()):raise FloatingPointError('Nonfinite entropy temperature')
                        with torch.no_grad():
                            denom=learn.sum().clamp(min=1)
                            diagnostics.append(dict(alpha=float(log_alpha.exp()),
                                entropy=float((-lp*learn).sum()/denom),
                                entropy_term=float((temperature*lp*learn).sum()/denom),
                                q_mean=float(replay_q),q_target_mean=float(y.mean()),q_gap=float(q_gap),
                                td_absolute_error=float(td_error),reward_mean=float(reward.mean()),
                                reward_std=float(reward.std()),learn_fraction=float(learn.mean()),
                                action_saturation=float(((act.abs()>.95).float().mean(-1)*learn).sum()/denom),
                                action_abs_mean=float((act.abs().mean(-1)*learn).sum()/denom),
                                actor_grad_norm=float(agrad),critic_grad_norm=float(qgrad)))
                        for param in critic.parameters():param.requires_grad_(True)
                        if not torch.isfinite(qloss+aloss):raise FloatingPointError('Nonfinite SAC loss')
                        with torch.no_grad():
                            if a.algorithm_kind=='sac' or update_actor:
                                for t,c in zip(target.parameters(),critic.parameters()):t.lerp_(c,cfg['tau'])
                                for t,c in zip(target_actor.parameters(),actor.parameters()):t.lerp_(c,cfg['tau'])
                        losses.append([float(qloss.detach()),float(aloss.detach()),float(lp.detach().mean())]);gradients+=1
                iteration+=1
                distance=sum(float((v.detach().cpu()-init[k]).square().sum()) for k,v in actor.state_dict().items())**.5
                row=dict(iteration=iteration,agent_steps=steps,gradient_updates=gradients,actor_gradient_updates=actor_gradients,replay_size=len(replay.rows),
                    environment_steps=iteration*a.workers*a.rollout,
                    actor_parameter_change_l2=distance,minutes=(time.monotonic()-start)/60,completed_episodes=len(finished),
                    losses=np.mean(losses,axis=0).tolist() if losses else None,
                    strict_success=float(np.mean([r['strict_success'] for r in finished])) if finished else None,
                    removal=float(np.mean([r['removal'] for r in finished])) if finished else None)
                if diagnostics:row['sac_diagnostics']={k:float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]}
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                if iteration%20==0:save(f'policy_it{iteration}.pt');save('policy.pt')
                for milestone in milestones:
                    if (iteration-1)*a.workers*a.rollout<milestone<=iteration*a.workers*a.rollout:
                        save(f'policy_steps{milestone}.pt')
                if iteration%a.resume_every==0:save('resume.pt',full=True)
            save('policy.pt');save('resume.pt',full=True)
            (a.out/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(iterations=iteration,environment_steps=iteration*a.workers*a.rollout,gradient_updates=gradients,actor_gradient_updates=actor_gradients,agent_steps=steps)))
    finally:
        for pipe in pipes:
            try:pipe.send(None)
            except (BrokenPipeError,EOFError):pass
        for proc in workers:
            proc.join(timeout=3)
            if proc.is_alive():proc.terminate()


def nn_loss(pred,target):return torch.nn.functional.smooth_l1_loss(pred,target)


def entropy_temperature_loss(log_alpha,log_probability,target_entropy,learn):
    """Only free decisions set temperature; no actor gradient through this loss."""
    return -(log_alpha*((log_probability.detach()+target_entropy)*learn).sum()/learn.sum().clamp(min=1))


if __name__=='__main__':main()
