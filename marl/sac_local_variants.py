"""Measured terminal servo and learned-Q acceptance variants for paired ablation."""
import numpy as np
import torch
from marl.image_flow_sac import ImageFlowController,ImageCritic,observation_tensors


class NearImageServo(ImageFlowController):
    """Bounded terminal feedback from delay-compensated image estimates only."""
    def observe(self,ep,est,tgt,rule,hold):
        obs=super().observe(ep,est,tgt,rule,hold)
        for i in range(ep.n):
            if not est.active[i] or hold[i] or tgt[i]<0:continue
            delta=self.clots[tgt[i]]-self.positions[i];distance=np.linalg.norm(delta)
            blend=float(np.clip((.35-distance)/.2,0.,1.))
            if blend<=0:continue
            confidence=float(self.motion.confidence[i]);weight=float(np.clip(confidence,0.,1.))
            gain=(1-weight)+weight*self.gain[i]
            desired=2.5*delta-(.3+.7*weight)*self.drift[i]
            wanted=self.F[i]@desired/(gain*self.speed)
            correction=np.clip(wanted-self.references[i],[-.25,-.06,-.06],[.25,.06,.06])
            command=self.references[i]+blend*correction;command/=max(1.,np.linalg.norm(command)/.97)
            self.references[i]=command;obs[i]['ctx'][19:22]=command
        return obs


class CriticGatedSAC(ImageFlowController):
    """Accept a learned continuous correction only if both Q heads support it."""
    def __init__(self,ep,checkpoint,config=None):
        super().__init__(ep,checkpoint=checkpoint,config=config)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.critic=ImageCritic();self.critic.load_state_dict(payload['critic']);self.critic.eval()
        self.stats.update(q_accepted_steps=0,q_rejected_steps=0)

    def __call__(self,ep,est,tgt,rule,hold):
        obs=self.observe(ep,est,tgt,rule,hold);tensors=observation_tensors(obs)
        with torch.no_grad():
            action,_=self.actor.sample(*tensors,deterministic=True)
            q1,q2=self.critic(*tensors,action);b1,b2=self.critic(*tensors,torch.zeros_like(action))
            gain=torch.minimum(q1,q2)-torch.minimum(b1,b2)
            uncertainty=.5*torch.maximum((q1-q2).abs(),(b1-b2).abs())
            accepted=gain>.01+uncertainty
            action=action*accepted[:,None]
        self.stats['q_accepted_steps']+=int((accepted.numpy()&est.active).sum())
        self.stats['q_rejected_steps']+=int((~accepted.numpy()&est.active).sum())
        return self.execute(ep,est,self.decode(action.numpy()))
