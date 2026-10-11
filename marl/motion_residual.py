"""SwitchSettle residual actor with observed-motion auxiliary supervision."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from marl.observed_motion import ObservedMotion
from marl.obstacle_control import TemporalPolicy


def action_dim(cfg):
    return 4 if cfg.get('lateral', False) else 2


def token_dim(cfg):
    return 40+14+3+action_dim(cfg)


class MotionPolicy(TemporalPolicy):
    def __init__(self, cfg):
        super().__init__(cfg['arch'], layers=cfg['layers'], window=cfg['window'], dim=token_dim(cfg))
        self.motion = nn.Sequential(nn.Linear(128+3, 128), nn.GELU(), nn.Linear(128, 3))
        # The predictive head conditions on the timestamp-aligned command.  The label is the displacement
        # between two received camera frames; no true flow/gain labels are used.
        nn.init.zeros_(self.motion[-1].weight); nn.init.zeros_(self.motion[-1].bias)
        self.pi = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, action_dim(cfg)))
        nn.init.zeros_(self.pi[-1].weight); nn.init.zeros_(self.pi[-1].bias)
        self.log_std = nn.Parameter(torch.full((action_dim(cfg),), cfg.get('init_log_std', -2.)))

    def predict_motion(self, h, command):
        # Predict only the command-to-motion mismatch around the identity response prior.
        return command + self.motion(torch.cat((h, command), dim=-1))


class MotionResidual:
    def __init__(self, ep, cfg):
        from scripts.benchmark_lysis import SwitchSettle
        self.cfg = cfg; self.base = SwitchSettle(ep, .15)
        self.motion = ObservedMotion(ep.n, ep.env.config.control_dt_s)
        self.last_a = np.zeros((ep.n, action_dim(cfg)))

    def observe(self, ep, est):
        if not hasattr(est, 'frame_time_s'):
            raise ValueError('MotionResidual requires acquisition timestamps')
        return self.motion.update(ep.env.elapsed_s, ep.sent_world, est.frame_time_s,
                                  est.pos, est.active, est.edge)

    def __call__(self, ep, est, targets, rule, hold, action):
        self.observe(ep, est)
        a = np.clip(np.asarray(action, float), -1., 1.)
        base = self.base(ep, est, targets, rule, hold)
        self.last_a = a.copy()
        if not np.any(a):
            return base
        F = ep.ctl.frames(est); world = np.einsum('nij,nj->ni', F.transpose(0, 2, 1), base)
        for i, target in enumerate(targets):
            if target < 0 or hold[i] or not est.active[i]:
                continue
            m = float(np.linalg.norm(world[i]))
            if m > 1e-8:
                desired = world[i]/m*float(np.clip(m+.3*a[i, 0], 0., 1.))
            else:
                desired = world[i].copy()
            gain = float(np.clip(self.motion.gain[i], .4, 1.6))
            inverse = (desired-self.motion.drift[i])/gain
            coeff = a[i, 1]*self.motion.confidence[i]
            proposal = desired+coeff*(inverse-desired)
            if a.shape[1] == 4:
                # Give the alternative arm cross-section authority only near the target.  Zero lateral action
                # is the original rule; the map wall executor is applied below to the full proposal.
                dist = float(np.linalg.norm(ep.env.clot_positions_mm[target]-est.pos[i]))
                gate = float(np.clip((2.-dist)/1.5, 0., 1.))
                proposal += F[i].T@np.array([0., .25*gate*a[i, 2], .25*gate*a[i, 3]])
            world[i] = proposal/max(float(np.linalg.norm(proposal)), 1.)
        local = np.einsum('nij,nj->ni', F, world)
        # Reuse the already-updated selected wall executor (do not call its replanner a second time).
        fixed = ep.env.elapsed_s-est.frame_time_s > .15
        guard = self.base.fixed if fixed else self.base.adaptive
        local = guard.exec.act(ep.env.elapsed_s, est, ep.ctl, local, hold)
        self.base.adaptive.prev_cmd = local.copy()
        return local


def motion_token(ep, est, targets, hold, controller, previous_action):
    from scripts.train_lysis_nav import token
    controller.observe(ep, est)
    T = token(ep, est, targets, hold, ep.sent_world.copy(), observable_history=True)
    F = ep.ctl.frames(est)
    # Put vector inputs into the same known-map local frame as the residual actions.
    for start in (0, 3, 6, 9, 15, 19, 23, 31, 35):
        T[:, start:start+3] = np.einsum('nij,nj->ni', F, T[:, start:start+3])
    m = controller.motion
    drift = np.einsum('nij,nj->ni', F, m.drift)
    cmd = np.einsum('nij,nj->ni', F, m.aligned_command)
    vel = np.einsum('nij,nj->ni', F, m.velocity)
    state = np.c_[m.gain/2., drift, m.confidence, m.rmse, cmd, vel,
                  np.full(ep.n, float(m.fresh)), np.full(ep.n, m.interval)]
    target_error = np.zeros((ep.n, 3))
    for i, c in enumerate(targets):
        if c >= 0:
            target_error[i] = np.clip(F[i]@(ep.env.clot_positions_mm[c]-est.pos[i]), -3., 3.)
    return np.concatenate((T, state, target_error, previous_action), axis=1).astype(np.float32)


class MotionController:
    def __init__(self, ep, checkpoint):
        from scripts.train_lysis_nav import History
        c = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.cfg = c['cfg']; self.net = MotionPolicy(self.cfg); self.net.load_state_dict(c['state']); self.net.eval()
        torch.set_num_threads(1)
        self.ctl = MotionResidual(ep, self.cfg); self.hist = History(ep.n, self.cfg['window'], token_dim(self.cfg))
        self.previous_action = np.zeros((ep.n, action_dim(self.cfg)))

    def __call__(self, ep, est, targets, rule, hold):
        T = motion_token(ep, est, targets, hold, self.ctl, self.previous_action)
        seq, mask = self.hist.push(T)
        with torch.no_grad():
            a = self.net.pi(self.net.backbone(torch.as_tensor(seq), torch.as_tensor(mask))).numpy()
        live = (np.asarray(targets) >= 0)&est.active&~hold
        a = np.clip(a, -1., 1.); a[~live] = 0.
        self.previous_action = a.copy()
        return self.ctl(ep, est, targets, rule, hold, a)
