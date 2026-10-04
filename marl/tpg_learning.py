"""Graph priority policy and causal 3-D residual control with measured dynamics loss.

The route graph, triggers and safety projection are shared engineering. Learned
graph priorities and temporal residual selection are independently ablatable.
"""
from itertools import permutations
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

NODE_DIM = 24
PAIR_DIM = 6
LOW_DIM = 124
HISTORY = 6
N_LOW = 9
CANDIDATE_DIM = 9


class GraphPriorityPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(NODE_DIM,hidden),nn.Tanh())
        self.message = nn.Sequential(nn.Linear(2*hidden+PAIR_DIM,hidden),nn.Tanh())
        self.score = nn.Sequential(nn.Linear(2*hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.critic = nn.Sequential(nn.Linear(2*hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        nn.init.zeros_(self.score[-1].weight); nn.init.zeros_(self.score[-1].bias)

    def forward(self,nodes,pairs):
        b,n,_ = nodes.shape
        h = self.encode(nodes)
        mask = pairs[...,5:6]
        messages = self.message(torch.cat((h[:,:,None].expand(-1,-1,n,-1),
            h[:,None,:].expand(-1,n,-1,-1),pairs),dim=-1))*mask
        neighbor = messages.sum(dim=2)/mask.sum(dim=2).clamp_min(1.)
        z = torch.cat((h,neighbor),dim=-1)
        prior = torch.where(nodes[...,15]>.5,-nodes[...,14],-nodes[...,17]-.2)
        prior = prior-.0001*torch.arange(n,device=nodes.device)
        scores = self.score(z).squeeze(-1)+.5*prior
        orders = list(permutations(range(n)))
        logits = torch.stack([sum((n-rank)*scores[:,i] for rank,i in enumerate(order)) for order in orders],dim=-1)
        # Reject permutations that would preempt an existing reservation.
        admitted=[]
        for order in orders:
            rank={node:r for r,node in enumerate(order)}
            bad=torch.zeros(b,dtype=torch.bool,device=nodes.device)
            for i in range(n):
                for j in range(n):
                    if rank[i]>rank[j]:bad |= pairs[:,i,j,4]>.5
            admitted.append(~bad)
        logits=logits.masked_fill(~torch.stack(admitted,dim=-1),-1e9)
        active = nodes[...,23:24]
        pooled = (z*active).sum(dim=1)/active.sum(dim=1).clamp_min(1.)
        return Categorical(logits=logits),self.critic(pooled).squeeze(-1)


class ResidualControlPolicy(nn.Module):
    def __init__(self,hidden=64):
        super().__init__()
        self.temporal = nn.GRU(LOW_DIM,hidden,batch_first=True)
        self.candidate = nn.Sequential(nn.Linear(CANDIDATE_DIM,32),nn.Tanh())
        self.score = nn.Sequential(nn.Linear(hidden+32,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.critic = nn.Sequential(nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.dynamics = nn.Sequential(nn.Linear(hidden+3,hidden),nn.Tanh(),nn.Linear(hidden,3))
        nn.init.zeros_(self.score[-1].weight); nn.init.zeros_(self.score[-1].bias)

    def encode(self,history):
        b,n,t,d = history.shape
        _,state = self.temporal(history.reshape(b*n,t,d))
        return state[-1].reshape(b,n,-1)

    def forward(self,history,candidates,valid,active):
        h = self.encode(history)
        e = self.candidate(candidates)
        z = torch.cat((h[:,:,None].expand(-1,-1,N_LOW,-1),e),dim=-1)
        logits = self.score(z).squeeze(-1)+.5*candidates[...,8]
        weights = active[...,None].float()
        pooled = (h*weights).sum(dim=1)/weights.sum(dim=1).clamp_min(1.)
        return Categorical(logits=logits.masked_fill(~valid,-1e9)),self.critic(pooled).squeeze(-1)

    def predict_measured_velocity(self,history,applied_commands):
        return self.dynamics(torch.cat((self.encode(history),applied_commands),dim=-1))


class TPGAgent(nn.Module):
    def __init__(self,hidden=64):
        super().__init__()
        self.high = GraphPriorityPolicy(hidden)
        self.low = ResidualControlPolicy(hidden)


def causal_history(frames):
    current=frames[-1]
    padded=[np.zeros_like(current)]*max(0,HISTORY-len(frames))+list(frames)[-HISTORY:]
    return np.stack(padded,axis=1).astype(np.float32)
