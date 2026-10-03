"""Hand-written reactive controllers that read ONLY the fair observation (marl.partial_obs).

Same information as the learned policies. Each robot decides alone from its own vector.
  target   nearest alive clot in the observation (slot 0)
  bearing  move along the straight-line direction to the target (Turbo-style goal seeking)
  path     among the visible centreline paths, take the one whose far end best aligns with the
           target direction, avoiding dead ends that do not contain the target; go straight to the
           target once it is within `direct_mm`
Both add the same particle avoidance as the privileged teacher, recomputed from the observed
relative position/velocity (repulsion from the closest-approach point, gain 6) and its wait rule
(stop if forecast clearance < 0.3 safety margins within 0.5 s), then the 0.35 stop deadzone.
"""
from __future__ import annotations

import numpy as np

from marl.partial_obs import CLOT0, PATH0, PART0, PARTICLE_SLOTS, PATH_SLOTS, PartialObsConfig

ROBOT_R, PARTICLE_R, SAFETY = .08, .02, .15
HORIZON = 1.


def _particles(o, cfg):
    out = []
    for k in range(PARTICLE_SLOTS):
        s = PART0+7*k
        if o[s] > 0:
            out.append((o[s+1:s+4]*cfg.sensing_radius_mm, o[s+4:s+7]*1.))   # mm, mm/s (robot speed 1 mm/s)
    return out


def _closest(rel, relv):
    t = float(np.clip(-(rel@relv)/max(relv@relv, 1e-12), 0, HORIZON))
    closest = rel+t*relv
    clear = (np.linalg.norm(closest)-ROBOT_R-PARTICLE_R)/SAFETY
    return t, closest, clear


def fair_reactive_action(obs, mode='path', cfg=PartialObsConfig(), avoid_gain=6., wait_clearance=.3,
                         wait_horizon=.5, deadzone=.35, direct_mm=.7):
    n = len(obs)
    act = np.zeros((n, 3))
    for i, o in enumerate(obs):
        if o[CLOT0] <= 0:
            continue
        goal_dir, goal_mm = o[CLOT0+2:CLOT0+5].astype(np.float64), o[CLOT0+1]*10.
        direction = goal_dir
        if mode == 'path' and goal_mm > direct_mm:
            best, score = None, -np.inf
            for k in range(PATH_SLOTS):
                s = PATH0+10*k
                if o[s] <= 0:
                    continue
                far_dir = o[s+4:s+7]
                sc = float(far_dir@goal_dir) - (2. if o[s+8] > 0 else 0.)
                if sc > score:
                    best, score = o[s+1:s+4].astype(np.float64), sc
            if best is not None:
                direction = best
        push = np.zeros(3); wait = False
        for rel, relv in _particles(o, cfg):
            t, closest, clear = _closest(rel, relv)
            push += np.clip(1-clear, 0, 1)**2*(-closest/max(np.linalg.norm(closest), 1e-9))
            wait |= clear < wait_clearance and t < wait_horizon
        local = np.clip(direction+avoid_gain*push, -1, 1)
        local /= max(np.linalg.norm(local), 1.)
        if wait or np.linalg.norm(local) < deadzone:
            local = np.zeros(3)
        act[i] = local
    return act
