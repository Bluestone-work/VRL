"""Image/history SAC local control with measured-flow compensation and authority audit.

The actor sees delayed biplane crops, detections, issued commands and the
pre-operative map. Current plant positions, flow, mass and cardiac phase are
never actor inputs. The original prediction selector is an input/initialization;
SAC's continuous actor produces the final command, without a cost argmin after it.
"""
from __future__ import annotations
from collections import deque
import math
import numpy as np
import torch
from torch import nn
from marl.dyn_predictor import Ensemble
from marl.dyn_select import PredictiveSelector
from marl.predictive_nav_rl import PREDICTORS,PARENT_WEIGHTS
from marl.observed_motion import ObservedMotion
from marl.vessel_sdf import VesselSDF
from scripts.benchmark_lysis import LysisEpisode
from marl.obstacle_features import forecast_features

CTX=48
IMAGE_CHANNELS=4


class ControlEpisode(LysisEpisode):
    """Keep original TPG precedence, optionally permit bounded station keeping.

    The controller must explicitly opt in. A hold remains a prohibition on
    advancing into a conflict zone; its command need not be zero in moving fluid.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.station_keeping=False
        self.authority=dict(active_agent_steps=0,tpg_hold_steps=0,shield_changed_steps=0,
                            hold_nonzero_steps=0,hold_motion_mm=0.,hold_wall_robot_s=0.)

    def step(self,est,local,hold):
        raw=np.asarray(local,float).copy();hold=np.asarray(hold,bool);active=est.active.copy()
        pos=self.env.positions_mm[:self.n].copy()  # metrics only, never sent to controller
        effective=np.zeros(self.n,bool) if self.station_keeping else hold
        done,result=super().step(est,raw,effective)
        if self.station_keeping:
            self.hold_s+=float((hold&active).sum())*float(self.info['step_duration_s'])
        pre_shield=raw.copy();pre_shield[effective]=0.
        s=self.authority;s['active_agent_steps']+=int(active.sum());s['tpg_hold_steps']+=int((hold&active).sum())
        s['shield_changed_steps']+=int(((np.linalg.norm(self.prev_local-pre_shield,axis=1)>1e-5)&active).sum())
        s['hold_nonzero_steps']+=int(((np.linalg.norm(self.prev_local,axis=1)>1e-5)&hold&active).sum())
        s['hold_motion_mm']+=float(np.linalg.norm(self.env.positions_mm[:self.n]-pos,axis=1)[hold&active].sum())
        s['hold_wall_robot_s']+=float(np.asarray(result['wall'])[hold&active].sum())
        return done,result


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(IMAGE_CHANNELS,16,5,2,2),nn.ReLU(),
            nn.Conv2d(16,24,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(384,32),nn.Tanh())
        self.gru=nn.GRU(11,64,batch_first=True)
        self.ctx=nn.Sequential(nn.Linear(CTX,64),nn.Tanh())
    def forward(self,hist,mask,ctx,image):
        h=self.gru(hist.masked_fill(mask[...,None],0.))[0][:,-1]
        return torch.cat([h,self.ctx(ctx),self.cnn(image)],-1)


class ImageActor(nn.Module):
    def __init__(self,action_mode='residual'):
        super().__init__();self.encoder=Encoder();self.action_mode=action_mode
        self.head=nn.Sequential(nn.Linear(160,128),nn.Tanh(),nn.Linear(128,6))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():self.head[-1].bias[3:]=-2.5
    def sample(self,hist,mask,ctx,image,deterministic=False):
        out=self.head(self.encoder(hist,mask,ctx,image))
        # A warm start in command space; there is no fixed-cost score competing
        # with the actor and no imitation loss constraining its final action.
        prior=ctx[:,19:22].clamp(-.97,.97)
        mean=(torch.atanh(prior) if self.action_mode=='direct' else 0.)+out[:,:3]
        std=out[:,3:].clamp(-5.,0.).exp()
        distribution=torch.distributions.Normal(mean,std)
        z=mean if deterministic else distribution.rsample();a=torch.tanh(z)
        logp=(distribution.log_prob(z)-torch.log(1-a.square()+1e-6)).sum(-1)
        return a,logp


class ImageCritic(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder()
        self.q1=nn.Sequential(nn.Linear(163,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
        self.q2=nn.Sequential(nn.Linear(163,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
    def forward(self,hist,mask,ctx,image,action):
        z=torch.cat([self.encoder(hist,mask,ctx,image),action],-1)
        return self.q1(z).squeeze(-1),self.q2(z).squeeze(-1)


class ImageFlowController:
    def __init__(self,ep,checkpoint=None,actor=None,stochastic=False,config=None):
        self.cfg=dict(compensation=0.,near_radius=0.,target_gain=3.,barrier=0.,
                      station_keeping=True,pulse=True,use_images=True,action_scales=[.2,.06,.06],
                      obstacle_features=False)
        if checkpoint:
            p=torch.load(checkpoint,map_location='cpu',weights_only=False)
            self.cfg.update(p['config']['controller']);actor=ImageActor(p['config'].get('action_mode','direct'));actor.load_state_dict(p['actor'])
            self.iterations=p['iterations']
        else:self.iterations=0
        if config:self.cfg.update(config)
        self.actor=actor;self.stochastic=stochastic
        if actor:actor.eval()
        self.parent=PredictiveSelector(ep,'learned',Ensemble(PREDICTORS),.0445,PARENT_WEIGHTS)
        self.parent.selection_hook=self._capture
        self.motion=ObservedMotion(ep.n,dt=float(ep.env.config.control_dt_s),window=32)
        self.sdf=VesselSDF(ep.sensor);self.clots=ep.env.clot_positions_mm.copy()
        self.speed=float(ep.env.config.robot_speed_mm_s);self.body=float(ep.ctl.body)
        self.goal_anchor=np.zeros((ep.n,3));self.held=np.zeros(ep.n,bool)
        self.previous_crops=np.zeros((ep.n,2,32,32),np.float32)
        self.previous_frame=None;self.captured={};self.last_obs=[];self.last_actions=None
        self.pulse_samples=[deque(maxlen=48) for _ in range(ep.n)];self.pulse_fit=[None]*ep.n
        self.obstacle_history=[deque(maxlen=3) for _ in range(ep.n)]
        self.stats=dict(local_steps=0,actor_changed_steps=0,wall_guard_changes=0,barrier_changes=0,
                        hold_projection_changes=0,observed_target_error_sum=0.,near_target_steps=0)
        ep.station_keeping=bool(self.cfg['station_keeping'])

    def _capture(self,ep,est,tgt,i,F,C,mu,var,epi,clr,prog,cost,proposal):
        self.captured[i]=dict(offset=F.T@mu[0,0],aux=self.parent.ens.last_aux[0].copy())
        return proposal

    def _belief(self,ep,est):
        now=float(ep.env.elapsed_s);frame=float(getattr(est,'frame_time_s',now))
        self.motion.update(now,ep.sent_world*self.speed,frame,est.pos,est.active,est.edge)
        drift=self.motion.drift.copy();gain=self.motion.gain.copy()
        for i in range(ep.n):
            if self.motion.fresh and self.motion.valid[i]:
                residual=self.motion.velocity[i]-gain[i]*self.motion.aligned_command[i]
                self.pulse_samples[i].append((frame-self.motion.interval*.5,np.clip(residual,-.5,.5)))
            samples=self.pulse_samples[i]
            if self.cfg['pulse'] and len(samples)>=20 and len(samples)%3==0 and self.motion.fresh:
                tt=np.array([s[0] for s in samples]);vv=np.array([s[1] for s in samples]);best=None
                for hz in np.linspace(.8,2.1,10):
                    X=np.c_[np.ones(len(tt)),np.sin(2*np.pi*hz*tt),np.cos(2*np.pi*hz*tt)]
                    beta=np.linalg.solve(X.T@X+np.diag([.05,.4,.4]),X.T@vv)
                    error=float(np.mean((X@beta-vv)**2))
                    if best is None or error<best[0]:best=(error,hz,beta)
                self.pulse_fit[i]=best
            fit=self.pulse_fit[i]
            if self.cfg['pulse'] and fit and self.motion.confidence[i]>.05:
                error,hz,beta=fit;pred=np.array([1.,math.sin(2*np.pi*hz*now),math.cos(2*np.pi*hz*now)])@beta
                w=float(np.clip(self.motion.confidence[i]/(1+error/.02),0.,.7))
                drift[i]=(1-w)*drift[i]+w*np.clip(pred,-.5,.5)
        return np.clip(drift,-.5,.5),np.clip(gain,.35,1.8),max(0.,now-frame)

    def observe(self,ep,est,tgt,rule,hold):
        self.captured={};base=self.parent(ep,est,tgt,rule,hold);F=ep.ctl.frames(est)
        drift,gain,age=self._belief(ep,est);conf=self.motion.confidence
        frame=float(getattr(est,'frame_time_s',ep.env.elapsed_s))
        images=np.asarray(ep.sensor.crops,np.float32)
        image_pair=np.concatenate([images,self.previous_crops],axis=1)
        if frame!=self.previous_frame:self.previous_crops=images.copy();self.previous_frame=frame
        self.base=base.copy();obs=[];positions=[];references=[]
        for i in range(ep.n):
            snapshot=self.captured.get(i)
            # Delayed detections plus learned delay compensation / aligned log.
            offset=(gain[i]*ep.sent_world[i]*self.speed+drift[i])*age
            if snapshot:offset=(1-conf[i])*snapshot['offset']+conf[i]*offset
            position=est.pos[i]+np.clip(offset,-.5,.5);positions.append(position)
            waiting=bool(hold[i] or tgt[i]<0)
            if waiting and not self.held[i]:self.goal_anchor[i]=position
            self.held[i]=waiting
            if waiting:goal=self.goal_anchor[i];desired=np.clip(2.*(goal-position),-.25,.25)
            else:
                goal=self.clots[int(tgt[i])];distance=float(np.linalg.norm(goal-position))
                desired=F[i].T@base[i]*self.speed
                if distance<self.cfg['near_radius']:
                    desired=self.cfg['target_gain']*(goal-position)
                    desired/=max(1.,np.linalg.norm(desired)/.6)
            compensation=float(self.cfg['compensation'])*float(np.clip(conf[i]*2.,0.,1.))
            if waiting:compensation=max(compensation,.5 if len(self.motion.samples[i])>=6 else 0.)
            command=(desired-compensation*drift[i])/((1-compensation)+compensation*gain[i])/self.speed
            command=F[i]@command;command/=max(1.,np.linalg.norm(command)/.96)
            if waiting and not self.cfg['station_keeping']:command[:]=0.
            h,mask=self.parent.feat.history(ep,est,i,F[i]);ctx=np.zeros(CTX,np.float32)
            ctx[:19]=self.parent.feat.context(ep,est,tgt,i,F[i]);ctx[19:22]=command
            ctx[22:25]=F[i]@drift[i]/self.speed;ctx[25:28]=F[i]@est.vel[i]/self.speed
            ctx[28:31]=[gain[i],min(self.motion.rmse[i],2.),conf[i]];ctx[31:34]=[age/.3,float(hold[i]),float(tgt[i]<0)]
            D,grad=self.sdf(position,int(est.edge[i]));clear=D-self.body
            ctx[34:38]=np.r_[np.clip(clear,-1,2),F[i]@grad]
            ctx[38:41]=F[i]@(goal-position)/2.;ctx[41]=min(np.linalg.norm(goal-position)/2.,3.)
            fit=self.pulse_fit[i]
            if fit:ctx[42:45]=[fit[1]/2.,np.linalg.norm(fit[2][1:]),min(fit[0],1.)]
            ctx[45]=float(est.active[i]);ctx[46]=min(float(ep.env.elapsed_s)/300.,1.)
            ctx[47]=float(np.asarray(est.clot_alive).mean())
            if self.cfg.get('obstacle_features', False):
                # Keep the established 48-D interface for checkpoint and
                # memory compatibility. In obstacle-aware runs, the six
                # pulse/lifecycle tail slots are replaced by a compact
                # deployable particle forecast; this mode is trained from
                # scratch and is never mixed with old checkpoints.
                current=est.particles[i] if hasattr(est,'particles') else []
                self.obstacle_history[i].append([(np.asarray(r),np.asarray(v),0.) for r,v in current])
                past=[]
                for age_idx, batch in enumerate(reversed(list(self.obstacle_history[i])[:-1]), 1):
                    for r,v,_ in batch: past.append((r,v,age_idx*float(ep.env.config.control_dt_s)))
                ctx[42:48]=forecast_features(ep, est, i, history=past)
            im=np.clip(image_pair[i],-.3,2.) if self.cfg['use_images'] else np.zeros_like(image_pair[i])
            obs.append(dict(hist=h,hmask=mask,ctx=ctx,image=im.astype(np.float16)))
            references.append(command)
        self.last_obs=obs;self.F=F;self.positions=np.asarray(positions);self.drift=drift;self.gain=gain
        self.references=np.asarray(references);self.age=age;self.targets=np.asarray(tgt);self.hold=np.asarray(hold)
        return obs

    def execute(self,ep,est,actions):
        raw=np.asarray(actions,float).copy();raw/=np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1.)
        raw[~est.active]=0.;out=raw.copy();F=self.F
        for i in range(ep.n):
            if not est.active[i]:continue
            world=F[i].T@out[i];position=self.positions[i]
            waiting=self.held[i]
            if waiting and self.cfg['station_keeping']:
                desired=2.*(self.goal_anchor[i]-position)
                desired/=max(1.,np.linalg.norm(desired)/.2)
                # TPG still prohibits advancement; RL can regulate transverse
                # motion while axial station keeping cancels measured drift.
                axis=F[i][0];v=self.gain[i]*world*self.speed+self.drift[i]
                correction=float((v-desired)@axis)
                world-=correction*axis/(self.gain[i]*self.speed)
                self.stats['hold_projection_changes']+=int(abs(correction)>1e-5)
            elif waiting:world[:]=0.
            D,g=self.sdf(position,int(est.edge[i]));clear=D-self.body
            if np.linalg.norm(g)>1e-6 and self.cfg['barrier']>0:
                outward=-g/np.linalg.norm(g);velocity=self.gain[i]*world*self.speed+self.drift[i]
                limit=max(-.15,(clear-float(self.cfg['barrier']))/max(.25,self.age+.2))
                excess=float(velocity@outward)-limit
                if excess>0:
                    world-=excess*outward/(self.gain[i]*self.speed);self.stats['barrier_changes']+=1
            world/=max(1.,np.linalg.norm(world));out[i]=F[i]@world
        # The parent reference has ALREADY passed WallGuard. Adding its inward
        # recovery bias a second time caused a large regression in the initial
        # feedback screen. Filter only the policy's new outward correction.
        guarded=out.copy();executor=self.parent.guard.exec
        for i in range(ep.n):
            if not est.active[i] or (self.held[i] and not self.cfg['station_keeping']):
                guarded[i]=0.;continue
            if not self.held[i] and ep.env.elapsed_s<executor.recover_until[i]:
                guarded[i]=self.base[i];continue
            D,g=self.sdf(est.pos[i],int(est.edge[i]));heading=F[i][0]
            g=g-float(g@heading)*heading
            if D-self.body<executor.cfg.recovery_margin_mm and np.linalg.norm(g)>1e-6:
                inward=g/np.linalg.norm(g);delta=F[i].T@(guarded[i]-self.base[i])
                outward=min(float(delta@inward),0.)
                scale=float(np.clip(1-(D-self.body)/executor.cfg.recovery_margin_mm,0.,1.))
                guarded[i]-=F[i]@(scale*outward*inward)
            guarded[i]/=max(1.,np.linalg.norm(guarded[i]))
        self.stats['wall_guard_changes']+=int(((np.linalg.norm(guarded-out,axis=1)>1e-5)&est.active).sum())
        self.stats['actor_changed_steps']+=int(((np.linalg.norm(raw-self.references,axis=1)>1e-5)&est.active).sum())
        self.stats['local_steps']+=int(est.active.sum())
        for i in range(ep.n):
            if self.targets[i]>=0 and est.active[i] and np.linalg.norm(self.positions[i]-self.clots[self.targets[i]])<1.:
                self.stats['observed_target_error_sum']+=float(np.linalg.norm(self.positions[i]-self.clots[self.targets[i]]))
                self.stats['near_target_steps']+=1
        self.last_actions=np.asarray(actions,np.float32).copy()
        return guarded

    def __call__(self,ep,est,tgt,rule,hold):
        obs=self.observe(ep,est,tgt,rule,hold)
        if self.actor:
            with torch.no_grad():actions,_=self.actor.sample(*observation_tensors(obs),deterministic=not self.stochastic)
            actions=self.decode(actions.numpy())
        else:actions=self.references.copy()
        return self.execute(ep,est,actions)

    def decode(self,action):
        if self.actor is not None and self.actor.action_mode=='residual':
            return self.references+np.asarray(self.cfg['action_scales'])*np.asarray(action)
        return np.asarray(action)

    def summary(self):
        s=dict(self.stats);n=max(s['local_steps'],1)
        s.update(actor_changed_fraction=s['actor_changed_steps']/n,wall_guard_changed_fraction=s['wall_guard_changes']/n,
                 barrier_changed_fraction=s['barrier_changes']/n,
                 observed_target_error_mm=s['observed_target_error_sum']/max(s['near_target_steps'],1),
                 rl_algorithm='image/history SAC' if self.actor else 'image/history feedback (no RL)',
                 checkpoint_updates=self.iterations,**self.parent.summary())
        return s


def observation_tensors(rows,device='cpu'):
    return [torch.as_tensor(np.stack([r[k] for r in rows]),device=device,
            dtype=torch.bool if k=='hmask' else torch.float32) for k in ('hist','hmask','ctx','image')]
