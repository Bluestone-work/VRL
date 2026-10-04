"""Measured allocation prior with learned corrections; no environment access."""
import torch
from torch import nn
from torch.distributions import Categorical
from marl.measured_options import OPTION_FEATURES, N_OPTIONS


class AllocationPriorActorCritic(nn.Module):
    def __init__(self, hidden=96, memory_bias=.5):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(OPTION_FEATURES+N_OPTIONS,hidden),nn.Tanh(),
                                   nn.Linear(hidden,hidden),nn.Tanh())
        self.actor=nn.Sequential(nn.Linear(2*hidden,hidden),nn.Tanh(),nn.Linear(hidden,N_OPTIONS))
        self.critic=nn.Sequential(nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.prior_strength=nn.Parameter(torch.tensor(float(memory_bias)))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)

    def forward(self,features,valid):
        encoded=self.encoder(features)
        active=features[...,OPTION_FEATURES-1:OPTION_FEATURES]
        context=(encoded*active).sum(dim=1)/active.sum(dim=1).clamp_min(1.)
        together=torch.cat((encoded,context[:,None].expand_as(encoded)),dim=-1)
        logits=self.actor(together)+self.prior_strength*features[...,OPTION_FEATURES:]
        return Categorical(logits=logits.masked_fill(~valid,-1e9)),self.critic(context).squeeze(-1)
