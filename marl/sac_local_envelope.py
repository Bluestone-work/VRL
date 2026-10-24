"""Scalar SAC with an observable distance-dependent speed action range.

Long upstream routes cannot tolerate the large braking range needed for clot
contact. Keep small speed adaptation in transit and allow stronger adaptation
near the mapped clot, using delayed-image position compensation only. The
envelope is an execution constraint, not a learned performance improvement.
"""
import numpy as np

from marl.sac_speed_control import SpeedOnlySAC


class EnvelopedSpeedSAC(SpeedOnlySAC):
    def __init__(self, ep, checkpoint=None, actor=None, stochastic=False, config=None):
        super().__init__(ep, checkpoint=checkpoint, actor=actor,
                         stochastic=stochastic, config=config)
        self.cfg.setdefault('transit_logit_gain', .08)
        self.cfg.setdefault('contact_logit_gain', 4.)
        self.cfg.setdefault('envelope_outer_mm', .35)
        self.cfg.setdefault('envelope_inner_mm', .12)

    def decode(self, action):
        latent = np.asarray(action, float).reshape(-1)
        distance = np.array([2. * row['ctx'][41] for row in self.last_obs])
        inner, outer = self.cfg['envelope_inner_mm'], self.cfg['envelope_outer_mm']
        weight = np.clip((outer - distance) / (outer - inner), 0., 1.)
        gain = (1. - weight) * self.cfg['transit_logit_gain'] + weight * self.cfg['contact_logit_gain']
        command = super().decode((latent * gain / self.cfg['speed_logit_gain'])[:, None])
        # Preserve an intentional stationary parent command. RL changes speed
        # where a reference direction exists, not arbitrary restart direction.
        command[np.linalg.norm(self.references, axis=1) < 1e-12] = 0.
        return command

    def summary(self):
        result = super().summary()
        result.update(action_envelope='observable distance-dependent log-odds range',
                      transit_logit_gain=self.cfg['transit_logit_gain'],
                      contact_logit_gain=self.cfg['contact_logit_gain'],
                      zero_reference_policy='preserve original stationary command')
        return result
