"""PPO local navigation on the strongest A_TPG + predictive controller.

The frozen motion ensemble is a belief input. The categorical actor, optimized
by environment-reward PPO, chooses the command direction AND speed. A heuristic
proposal is an input/warm start, not the final decision. No plant truth is read
by the actor or its features. WallGuard, uncertainty fallback, TPG, and spacing
shield are unchanged; disabling RL reproduces the parent controller exactly.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from collections import deque
from marl.dyn_predictor import CTX_DIM,H_HIST,HIST_DIM,Ensemble
from marl.dyn_select import PredictiveSelector,CONTACT_MM,HORIZONS

MAX_CANDIDATES=32
# command(3), predicted motion(15), uncertainty(2), clearance(5), route progress(1),
# target distance(5), peer clearance(5), obstacle clearance(1), candidate flags(3),
# parent cost(1), parent proposal flag(1)
CAND_DIM=50
PREDICTORS=[f'research/runs/EXP0090_20261011/predictor/predictor_s{i}.pt' for i in range(3)]
PARENT_WEIGHTS=dict(delta=.05,prog=1.5,appr=3.,kappa=0.)


class NavigationActor(nn.Module):
    """Shared recurrent actor with a variable candidate set and a per-agent value."""
    def __init__(self,width=64,proposal_bias=.2):
        super().__init__();self.width=width;self.initial_proposal_bias=proposal_bias
        self.history=nn.GRU(HIST_DIM,width,batch_first=True)
        self.context=nn.Sequential(nn.Linear(CTX_DIM,width),nn.Tanh())
        self.candidate=nn.Sequential(nn.Linear(CAND_DIM,width),nn.Tanh())
        self.actor=nn.Sequential(nn.Linear(3*width,width),nn.Tanh(),nn.Linear(width,1))
        self.proposal=nn.Linear(CAND_DIM,1,bias=False)
        self.value=nn.Sequential(nn.Linear(2*width,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.proposal.weight)
        with torch.no_grad():
            # Learnable score initialized from the parent, including its 0.05
            # anchor margin. PPO updates both this score and the recurrent head;
            # there is no hard heuristic argmin after the network.
            self.proposal.weight[0,-2]=-40.
            self.proposal.weight[0,-1]=proposal_bias

    def dist(self,hist,hmask,ctx,cand,valid):
        hist=hist.masked_fill(hmask[...,None],0.)
        h=self.history(hist)[0][:,-1];c=self.context(ctx);base=torch.cat([h,c],-1)
        features=self.candidate(cand)
        logits=self.actor(torch.cat([base[:,None].expand(-1,cand.shape[1],-1),features],-1)).squeeze(-1)
        logits=logits+self.proposal(cand).squeeze(-1)
        logits=logits.masked_fill(~valid,-1e9)
        return torch.distributions.Categorical(logits=logits),self.value(base).squeeze(-1)


class RLNavigator(PredictiveSelector):
    def __init__(self,ep,checkpoint=None,net=None,stochastic=False,rl_off=False,predictors=None,
                 uncertainty_fallback=False):
        payload=None
        if checkpoint:
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            cfg=payload['config'];net=NavigationActor(cfg['width'],cfg['proposal_bias']);net.load_state_dict(payload['state'])
            predictors=cfg['predictors']
        # The first PPO round must decide among all legal candidates. An
        # uncertainty fallback is an explicit later ablation, never implicit.
        super().__init__(ep,'learned',Ensemble(predictors or PREDICTORS),
                         .0445 if uncertainty_fallback else None,PARENT_WEIGHTS)
        self.net=net if net is not None else NavigationActor();self.net.eval()
        self.stochastic=stochastic;self.rl_off=rl_off;self.records=[]
        self.rl_stats=dict(rl_decisions=0,rl_changed_decisions=0,rl_command_changed_decisions=0,rl_command_deviation_sum=0.,
                           rl_output_changed_decisions=0,rl_output_deviation_sum=0.,rl_update_command_changes=0)
        self.selection_hook=None if rl_off else self._choose
        self.checkpoint_iterations=payload.get('iterations',0) if payload else None
        self.map_clots=ep.env.clot_positions_mm.copy()
        self.obstacle_history=[deque(maxlen=3) for _ in range(ep.n)]

    def _features(self,ep,est,tgt,i,F,C,mu,var,epi,clr,prog,cost,proposal):
        hist,hmask=self.feat.history(ep,est,i,F);ctx=self.feat.context(ep,est,tgt,i,F)
        # Deduplicate only the learned candidate action set.  The parent
        # selector keeps its historical proposal ordering and behavior.
        keep=[]; remap={}
        for j,cmd in enumerate(C):
            hit=next((q for q,k in enumerate(keep) if np.allclose(cmd,C[k],rtol=0.,atol=1e-8)),None)
            if hit is None: remap[j]=len(keep); keep.append(j)
            else: remap[j]=hit
        C=np.asarray(C)[keep]; mu=mu[keep]; var=var[keep]; epi=epi[keep]; clr=clr[keep]
        prog=np.asarray(prog)[keep]; cost=np.asarray(cost)[keep]; proposal=remap[int(proposal)]
        K=len(C)
        if K>MAX_CANDIDATES:raise ValueError('Increase candidate padding bound')
        Pw=est.pos[i][None,None]+np.einsum('ab,kha->khb',F,mu)
        dq=np.linalg.norm(Pw-self.map_clots[int(tgt[i])],axis=-1)
        # Clot positions are the immutable pre-operative map, never remaining mass.
        peer=np.full((K,len(HORIZONS)),6.,float)
        obstacle=np.full((K,len(HORIZONS)),6.,float)
        particle_rows = est.particles[i] if hasattr(est, 'particles') and i < len(est.particles) else []
        self.obstacle_history[i].append([(np.asarray(r),np.asarray(v),0.) for r,v in particle_rows])
        for batch in reversed(list(self.obstacle_history[i])[:-1]):
            particle_rows = list(particle_rows)+[(r+v*float(ep.env.config.control_dt_s),v) for r,v,_ in batch]
        for rel, relv in particle_rows:
            rel=np.asarray(rel,float); relv=np.asarray(relv,float)
            t=np.asarray(HORIZONS,float)*float(ep.env.config.control_dt_s)
            Q=est.pos[i][None]+rel[None]+relv[None]*t[:,None]
            obstacle=np.minimum(obstacle,np.linalg.norm(Pw-Q[None],axis=-1))
        for j in range(ep.n):
            if j==i or not est.active[j]:continue
            Q=est.pos[j]+np.asarray(HORIZONS)[:,None]*float(ep.env.config.control_dt_s)*est.vel[j]
            peer=np.minimum(peer,np.linalg.norm(Pw-Q,axis=-1))
        flags=np.zeros((K,3));flags[0,0]=1;flags[1,1]=1;flags[2,2]=1
        proposed=np.zeros((K,1));proposed[proposal]=1
        obstacle_clear=np.maximum(obstacle-float(getattr(ep.env.config,'robot_radius_mm',.08))-
                                  float(getattr(ep.env.config,'particle_radius_mm',.02)),0.)
        features=np.concatenate([C@F.T,mu.reshape(K,-1),np.sqrt(var.sum(-1)),np.sqrt(epi.sum(-1)),
                                  clr,np.asarray(prog)[:,None],dq/2.,peer/2.,obstacle_clear[:,-1:]/2.,flags,
                                  np.clip(np.asarray(cost)[:,None],-30,30)/10.,proposed],axis=1)
        if features.shape[1]!=CAND_DIM:raise AssertionError(features.shape)
        cand=np.zeros((MAX_CANDIDATES,CAND_DIM),np.float32);cand[:K]=np.clip(features,-10,10)
        valid=np.arange(MAX_CANDIDATES)<K
        # All generated candidates are legal PPO actions in the first round.
        # A later uncertainty-fallback ablation may mask them explicitly.
        if self.fallback_std is not None:
            valid[:K]&=np.sqrt(epi[:,-1].sum(-1))<=float(self.fallback_std)
        valid[0]=True
        if not valid[proposal]:proposal=0;cand[:,-1]=0;cand[0,-1]=1
        return dict(hist=hist,hmask=hmask,ctx=ctx,cand=cand,valid=valid,proposal=int(proposal),source_indices=np.asarray(keep,dtype=int))

    def _choose(self,ep,est,tgt,i,F,C,mu,var,epi,clr,prog,cost,proposal):
        obs=self._features(ep,est,tgt,i,F,C,mu,var,epi,clr,prog,cost,proposal)
        T=lambda k:torch.as_tensor(obs[k][None])
        with torch.no_grad():
            d,v=self.net.dist(T('hist'),T('hmask'),T('ctx'),T('cand'),T('valid'))
            action=d.sample() if self.stochastic else d.logits.argmax(-1)
            a=int(action[0]);lp=float(d.log_prob(action)[0])
        self.rl_stats['rl_decisions']+=1
        changed=a!=obs['proposal'];self.rl_stats['rl_changed_decisions']+=int(changed)
        source=obs['source_indices']
        deviation=float(np.linalg.norm(C[source[a]]-C[source[obs['proposal']]]))
        self.rl_stats['rl_command_deviation_sum']+=deviation;self.rl_stats['rl_command_changed_decisions']+=int(deviation>1e-5)
        initial_score=-40.*obs['cand'][:,-2]+self.net.initial_proposal_bias*obs['cand'][:,-1]
        initial_score[~obs['valid']]=-1e9
        initial_action=int(np.argmax(initial_score))
        self.rl_stats['rl_update_command_changes']+=int(np.linalg.norm(C[source[a]]-C[source[initial_action]])>1e-5)
        self.records.append(dict(obs,agent=i,action=a,logp=lp,value=float(v[0]),learn=True,proposed_command=C[source[obs['proposal']]].copy(),selected_command=C[source[a]].copy()))
        return a

    def __call__(self,*args):
        self.records=[]
        out=super().__call__(*args)
        ep,est,tgt,rule,hold=args
        for r in self.records:
            i=r['agent'];F=ep.ctl.frames(est)[i];parent=F@r['proposed_command']
            if r['proposal']>=2:
                command=np.zeros_like(out);command[i]=parent
                parent=self.guard.exec.act(ep.env.elapsed_s,est,ep.ctl,command,hold)[i]
            deviation=float(np.linalg.norm(out[i]-parent))
            self.rl_stats['rl_output_deviation_sum']+=deviation;self.rl_stats['rl_output_changed_decisions']+=int(deviation>1e-5)
        if self.stochastic:
            recorded={r['agent'] for r in self.records}
            for i in range(ep.n):
                if not est.active[i] or i in recorded:continue
                F=ep.ctl.frames(est)[i];hist,hmask=self.feat.history(ep,est,i,F);ctx=self.feat.context(ep,est,tgt,i,F)
                cand=np.zeros((MAX_CANDIDATES,CAND_DIM),np.float32);cand[0,-1]=1
                valid=np.zeros(MAX_CANDIDATES,bool);valid[0]=True
                obs=dict(hist=hist,hmask=hmask,ctx=ctx,cand=cand,valid=valid,proposal=0)
                T=lambda k:torch.as_tensor(obs[k][None])
                with torch.no_grad():_,v=self.net.dist(T('hist'),T('hmask'),T('ctx'),T('cand'),T('valid'))
                self.records.append(dict(obs,agent=i,action=0,logp=0.,value=float(v[0]),learn=False,
                                         proposed_command=np.zeros(3),selected_command=np.zeros(3)))
        return out

    def summary(self):
        result=super().summary();D=max(self.rl_stats['rl_decisions'],1)
        result.update(self.rl_stats,rl_changed_fraction=self.rl_stats['rl_changed_decisions']/D,
                      rl_mean_command_deviation=self.rl_stats['rl_command_deviation_sum']/D,
                      rl_command_changed_fraction=self.rl_stats['rl_command_changed_decisions']/D,
                      rl_output_changed_fraction=self.rl_stats['rl_output_changed_decisions']/D,
                      rl_mean_output_deviation=self.rl_stats['rl_output_deviation_sum']/D,
                      rl_update_command_changed_fraction=self.rl_stats['rl_update_command_changes']/D,
                      rl_enabled=not self.rl_off,rl_checkpoint_iterations=self.checkpoint_iterations)
        return result


def observation_tensors(rows,device='cpu'):
    return [torch.as_tensor(np.stack([r[k] for r in rows]),device=device) for k in ('hist','hmask','ctx','cand','valid')]


def advantages(buf,bootstrap,gamma=.995,lam=.95):
    """GAE per episode/agent; holds remain in the critic stream, no fake actions."""
    adv=np.zeros(len(buf),np.float32);following={}
    for k in range(len(buf)-1,-1,-1):
        r=buf[k];key=tuple(r['stream'])
        nv,ga=following.get(key,(bootstrap.get(key,0.),0.))
        if r['terminal']:nv=0.;ga=0.
        if r.get('truncated'):nv=float(r.get('terminal_value',0.));ga=0.
        delta=float(r['reward'])+gamma*nv-float(r['value'])
        adv[k]=delta+gamma*lam*ga
        following[key]=(float(r['value']),float(adv[k]))
    return adv,adv+np.asarray([r['value'] for r in buf],np.float32)
