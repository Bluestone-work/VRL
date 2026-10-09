"""EXP0061: action-conditioned observation/outcome ensemble used by a PPO actor.

This is a small one-step learned world model, not Dreamer/RSSM or a physics oracle.
Inputs are the same measured 66-d histories as EXP0060. Outcome labels may use
simulator training rewards; inference never queries the simulator for candidate
action consequences. The matched PPO continuation sets world_active=False.
"""
import torch
from torch import nn
from marl.vascular_option_rl import OptionPolicy, OBS_DIM, OPTIONS

MODEL_STATE_DIM = 24
MODEL_OUT = MODEL_STATE_DIM+1+3   # next measured state, scaled reward, three hazards


class PredictivePolicy(OptionPolicy):
    def __init__(self, use_world=True):
        super().__init__(memory=True,central=True)
        self.use_world=use_world; self.world_active=use_world
        self.models=nn.ModuleList([nn.Sequential(nn.Linear(64+len(OPTIONS),128),nn.Tanh(),
                            nn.Linear(128,MODEL_OUT)) for _ in range(3)])
        # Start the dynamics component at the explicit persistence baseline;
        # learn changes in measured state instead of relearning static geometry.
        for model in self.models:
            with torch.no_grad():
                model[-1].weight[:MODEL_STATE_DIM].zero_();model[-1].bias[:MODEL_STATE_DIM].zero_()
        # For each action: reward, 3 hazard probabilities, reward disagreement.
        self.world_gate=nn.Linear(len(OPTIONS)*5,len(OPTIONS),bias=False)
        nn.init.zeros_(self.world_gate.weight)

    def encode(self,history):
        b,n,h,_=history.shape
        z=self.encoder(history.reshape(b*n,h,OBS_DIM))
        return self.gru(z)[0][:,-1].reshape(b,n,64)

    def predictions(self,z):
        actions=torch.eye(len(OPTIONS),device=z.device).expand(*z.shape[:-1],len(OPTIONS),len(OPTIONS))
        inp=torch.cat((z.unsqueeze(-2).expand(*z.shape[:-1],len(OPTIONS),64),actions),-1)
        return torch.stack([model(inp) for model in self.models],dim=0)

    def forward(self,history,active,return_model=False):
        z=self.encode(history); pred=self.predictions(z)
        pred=torch.cat((pred[...,:24]+history[None,:,:,-1,None,:24],pred[...,24:]),-1)
        logits=self.actor(z)
        if self.world_active:
            # PPO cannot distort the predictor just to increase the selected logit.
            detached=pred.detach(); mean=detached.mean(0)
            features=torch.cat((mean[...,24:25],detached[...,25:].sigmoid().mean(0),
                                detached[...,24:25].std(0,unbiased=False)),-1)
            logits=logits+self.world_gate(features.flatten(-2))
        pooled=(z*active[...,None]).sum(1)/active.sum(1,keepdim=True).clamp(min=1)
        v=self.critic(torch.cat((z,pooled[:,None].expand(-1,z.shape[1],-1)),-1)).squeeze(-1)
        result=(torch.distributions.Categorical(logits=logits),v)
        return (*result,pred) if return_model else result

    def load_parent(self,payload):
        if not payload['config']['memory'] or not payload['config']['central']:
            raise ValueError('Parent must be the registered recurrent MAPPO architecture')
        expected=set(OptionPolicy(True,True).state_dict())
        if set(payload['state'])!=expected:
            raise ValueError('Parent state dictionary does not match the complete base policy')
        missing,unexpected=self.load_state_dict(payload['state'],strict=False)
        if unexpected or any(not k.startswith(('models.','world_gate.')) for k in missing):
            raise ValueError('Unexpected missing/unexpected parent parameters')


def model_loss(pred,actions,next_obs,reward,hazards,mask,bootstrap=True):
    """Ensemble bootstrap; no terminal reset observation is used as a target."""
    e,b,n,_,d=pred.shape
    index=actions[None,...,None,None].expand(e,b,n,1,d)
    selected=pred.gather(-2,index).squeeze(-2)
    state_loss=nn.functional.smooth_l1_loss(selected[...,:24],next_obs[None,...,:24].expand(e,-1,-1,-1),reduction='none').mean(-1)
    reward_loss=(selected[...,24]-reward[None]/10.).square()
    risk_loss=nn.functional.binary_cross_entropy_with_logits(selected[...,25:],hazards[None].expand(e,-1,-1,-1),
                         pos_weight=torch.full((3,),5.,device=pred.device),reduction='none').mean(-1)
    weights=mask[None].expand(e,-1,-1)
    if bootstrap:weights=weights*(torch.rand_like(weights)<.8)
    loss=((state_loss+reward_loss+risk_loss)*weights).sum()/weights.sum().clamp(min=1)
    return loss,dict(state_loss=state_loss.mean().detach(),reward_loss=reward_loss.mean().detach(),risk_loss=risk_loss.mean().detach())
