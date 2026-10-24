"""One-dimensional image/history SAC for reference-aligned velocity adaptation.

Keep the strongest predictor's direction and original TPG/parking semantics.
The actor learns a bounded log-odds speed shift, including braking from full
reference speed. Its mean-zero initialization exactly preserves the parent.
"""
import numpy as np
import torch
from torch import nn
from marl.image_flow_sac import Encoder
from marl.sac_safe_local import SafeLocalSAC


class SpeedActor(nn.Module):
    action_mode='speed_relative_logit'
    def __init__(self):
        super().__init__();self.encoder=Encoder()
        self.head=nn.Sequential(nn.Linear(160,128),nn.Tanh(),nn.Linear(128,2))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():self.head[-1].bias[1]=-2.5
    def sample(self,hist,mask,ctx,image,deterministic=False):
        out=self.head(self.encoder(hist,mask,ctx,image));mean=out[:,:1];std=out[:,1:].clamp(-5.,0.).exp()
        distribution=torch.distributions.Normal(mean,std)
        z=mean if deterministic else distribution.rsample();a=torch.tanh(z)
        logp=(distribution.log_prob(z)-torch.log(1-a.square()+1e-6)).sum(-1)
        return a,logp


class SpeedCritic(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder()
        self.q1=nn.Sequential(nn.Linear(161,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
        self.q2=nn.Sequential(nn.Linear(161,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1))
    def forward(self,hist,mask,ctx,image,action):
        z=torch.cat([self.encoder(hist,mask,ctx,image),action],-1)
        return self.q1(z).squeeze(-1),self.q2(z).squeeze(-1)


class SpeedOnlySAC(SafeLocalSAC):
    def __init__(self,ep,checkpoint=None,actor=None,stochastic=False,config=None):
        cfg=dict(speed_logit_gain=4.)
        iteration=0
        if checkpoint:
            p=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if p['config']['action_mode']!='speed_relative_logit':raise ValueError('Expected scalar speed SAC checkpoint')
            cfg.update(p['config']['controller']);actor=SpeedActor();actor.load_state_dict(p['actor']);iteration=p['iterations']
        cfg.update(config or {})
        super().__init__(ep,actor=actor,stochastic=stochastic,config=cfg)
        self.iterations=iteration

    def decode(self,action):
        latent=np.asarray(action,float).reshape(-1)
        length=np.linalg.norm(self.references,axis=1)
        bounded=np.clip(length,.05,.95)
        logit=np.log(bounded/(1-bounded))
        speed=np.clip(1/(1+np.exp(-(logit+self.cfg['speed_logit_gain']*latent)))+length-bounded,0.,1.)
        direction=self.references/np.maximum(length[:,None],1e-12)
        command=speed[:,None]*direction
        # A zero reference still permits learned start/back-off along the map
        # tangent; the zero actor itself must remain an exact no-op.
        command[length<1e-12,0]=latent[length<1e-12]
        command[latent==0.]=self.references[latent==0.]
        return command

    def summary(self):
        result=super().summary();result.update(rl_action_dimension=1,
            action_parameterization='bounded log-odds speed adaptation along the original reference direction')
        return result
