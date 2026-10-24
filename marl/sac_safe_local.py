"""SAC navigation with the original reference and original non-navigation actions.

The experimental station-keeping loop and unconditional 0.96 reference cap
changed the untrained controller and caused severe curved-vessel regressions.
This executor preserves the parent exactly at zero residual and lets RL change
only active navigation. The original TPG/WallGuard/spacing rules still execute.
"""
import numpy as np
from marl.image_flow_sac import ImageFlowController


class SafeLocalSAC(ImageFlowController):
    def __init__(self,ep,checkpoint=None,actor=None,stochastic=False,config=None):
        cfg=dict(action_scales=[.2,.02,.02]);cfg.update(config or {});cfg['station_keeping']=False
        super().__init__(ep,checkpoint=checkpoint,actor=actor,stochastic=stochastic,config=cfg)
        self.stats['nonnavigation_override_steps']=0

    def observe(self,ep,est,tgt,rule,hold):
        obs=super().observe(ep,est,tgt,rule,hold)
        self.references=self.base.copy()
        for i,row in enumerate(obs):row['ctx'][19:22]=self.references[i]
        return obs

    def decode(self,action):
        if self.actor is None or self.actor.action_mode!='residual':return super().decode(action)
        latent=np.asarray(action);length=np.linalg.norm(self.references,axis=1)
        direction=self.references/np.maximum(length[:,None],1e-12)
        direction[length<1e-12]=[1.,0.,0.]
        scale=np.asarray(self.cfg['action_scales'])
        # Speed adaptation follows the already-selected safe route direction;
        # the two learned lateral adjustments have a smaller physical range.
        delta=scale[0]*latent[:,0,None]*direction
        delta[:,1:]+=scale[1:]*latent[:,1:]
        return self.references+delta

    def execute(self,ep,est,actions):
        out=super().execute(ep,est,actions)
        exact_zero=np.all(np.asarray(actions)==self.base,axis=1)
        idle=(self.targets<0)|self.hold|~est.active
        self.stats['nonnavigation_override_steps']+=int((idle&est.active&
            (np.linalg.norm(np.asarray(actions)-self.base,axis=1)>1e-5)).sum())
        # Restore the parent's already-guarded parking/recovery command; avoid
        # converting a finished task into an unvalidated feedback holding loop.
        out[idle|exact_zero]=self.base[idle|exact_zero]
        out[self.hold]=0.;out[~est.active]=0.
        return out

    def summary(self):
        result=super().summary();result['execution_revision']='exact parent reference; original hold/parking; SAC on active navigation'
        return result
