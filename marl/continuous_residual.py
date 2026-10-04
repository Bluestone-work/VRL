"""Continuous measured residuals remove categorical nominal-action lock-in."""
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal, kl_divergence
from marl.measured_marl import MeasuredMARL, MeasuredEncoder, mlp
from marl.conservative_residual import VARIANTS, measured_anchor_weight


class ContinuousActor(nn.Module):
    def __init__(self,hidden=64,clusters=3):
        super().__init__()
        self.encoder=MeasuredEncoder(hidden,clusters)
        self.score=mlp(2*hidden+9,hidden,3)
        self.log_std=nn.Parameter(torch.full((3,),math.log(.35)))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)

    def forward(self,state):
        local,team=self.encoder(state['history'],state['nodes'],state['pairs'])
        preferred=state['candidates'][...,8].argmax(-1)
        nominal=state['candidates'].gather(2,preferred[...,None,None].expand(-1,-1,1,9)).squeeze(2)
        mean=self.score(torch.cat((local,team,nominal),-1))
        std=self.log_std.clamp(-2.5,-.7).exp().expand_as(mean)
        return Independent(Normal(mean,std),1)


class ContinuousMARL(MeasuredMARL):
    def __init__(self,variant,hidden=64,clusters=3,prior=0.):
        if variant not in VARIANTS:raise ValueError(variant)
        super().__init__('vctpg_no_ac' if variant.startswith('graph') else 'r_mappo',hidden,clusters,prior)
        self.low_actor=ContinuousActor(hidden,clusters)
        self.variant=variant


def continuous_anchor(dist,state,prior=None):
    ref=Independent(Normal(torch.zeros_like(dist.mean),torch.full_like(dist.mean,.35)),1)
    return measured_anchor_weight(state)*kl_divergence(dist,ref)


def residual_command(nominal,raw,enabled,radius):
    """Bounded latent-action transform, with exact zero-residual identity.

    Log probabilities are those of the sampled latent normal. Its fixed
    tanh/control transform is part of the environment; no sampled action is
    incorrectly scored under a clipped normal density.
    """
    nominal,raw=np.asarray(nominal,float),np.asarray(raw,float)
    if nominal.shape!=raw.shape or not np.isfinite(raw).all():raise ValueError('Invalid continuous action')
    output=nominal.copy()
    live=np.asarray(enabled,bool)&np.any(raw!=0.,axis=-1)
    output[live]+=radius/math.sqrt(3)*np.tanh(raw[live])
    output[live]/=np.maximum(np.linalg.norm(output[live],axis=-1,keepdims=True),1.)
    return output
