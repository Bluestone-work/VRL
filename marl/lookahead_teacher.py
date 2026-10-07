"""Privileged training-only option teacher for the deployable obstacle controller.

The teacher may inspect copied simulator outcomes while labelling demonstrations.  Its labels are
never consumed by the deployed controller; the student receives only ``token()`` observations.
"""
from __future__ import annotations

import copy
import numpy as np

from marl.obstacle_control import apf

OPTION_NAMES = (
    'pursuit', 'apf', 'wall_apf', 'left', 'right', 'up', 'down', 'wait', 'switch',
)
# v1 labels (8 options, per-step decisions, 1 s held horizon) used the first 8 entries only.
SWITCH_GAP = .3   # mm, detected surface gap below which 'switch' uses wall_apf instead of pursuit


def _basis(d):
    d = np.asarray(d, float)
    d = d / max(np.linalg.norm(d), 1e-9)
    h = np.eye(3)[int(np.argmin(np.abs(d)))]
    e1 = np.cross(d, h); e1 /= max(np.linalg.norm(e1), 1e-9)
    return e1, np.cross(d, e1)


def option_local(ep, est, rule, option):
    """Return a local command for an option, without reading simulator truth."""
    u = np.asarray(rule, float).copy()
    for i in range(ep.n):
        if not est.active[i]:
            u[i] = 0.; continue
        d = ep.ctl.nominal[i]
        if option == 0:
            u[i] = d
        elif option == 1:
            # Reuse the same measured detections as APFPursuit, but make the option explicit.
            F = ep.ctl.frames(est)[i]
            obs = [(F @ x, F @ v, r) for x, v, r in est.obstacles[i]]
            u[i] = apf(d, obs, ep.ctl.body)
        elif option == 2:
            F = ep.ctl.frames(est)[i]
            obs = [(F @ x, F @ v, r) for x, v, r in est.obstacles[i]]
            ax, radius, radial = ep.sensor.map_coordinates(est, i)
            off = ax-est.pos[i]
            wall_dir = off/max(np.linalg.norm(off), 1e-9)
            u[i] = apf(d, obs, ep.ctl.body, F @ wall_dir, radius-radial-ep.ctl.body)
        elif option == 8:
            gap = min((float(np.linalg.norm(x))-ep.ctl.body-r for x, _, r in est.obstacles[i]), default=np.inf)
            u[i] = option_local(ep, est, rule, 2 if gap < SWITCH_GAP else 0)[i]
        elif option in (3, 4, 5, 6):
            e1, e2 = _basis(d); side = (e1, -e1, e2, -e2)[option-3]
            u[i] = .75*d + .8*side
            u[i] /= max(np.linalg.norm(u[i]), 1e-9)
        else:
            u[i] = 0.
    return u


def _score(ep, before, out, after):
    """Lower is better; simulator-only quantities are confined to this function."""
    collisions = float(out['obs_events'])
    wall = float(np.sum(out['wall']))
    lost = float(np.sum(out['lost']))
    gain = float(out['removed'])
    progress = float(np.linalg.norm(after-before))
    return 100.*collisions + 20.*wall + 50.*lost - 10.*gain - .05*progress


def label(ep, est, rule, hold, horizon=10):
    """Evaluate all options on copied episodes and return ``(best, costs)``.

    ``horizon`` is measured in control steps.  The caller remains untouched.
    """
    costs = []
    before = ep.env.positions_mm[:ep.n].copy()
    for option in range(len(OPTION_NAMES)):
        trial = copy.deepcopy(ep)
        trial.prev_local = ep.prev_local.copy()
        trial_est, _, trial_rule, trial_hold = trial.observe()
        total = 0.
        for step in range(horizon):
            if step:
                trial_est, _, trial_rule, trial_hold = trial.observe()
            local = option_local(trial, trial_est, trial_rule, option)
            done, out = trial.step(trial_est, local, trial_hold)
            total += _score(trial, before, out, trial.env.positions_mm[:trial.n])
            if done: break
        trial.close()
        costs.append(total)
    costs = np.asarray(costs, float)
    return int(np.argmin(costs)), costs


def label_v2(ep, est, rule, hold, hold_steps=5, tail=25, base=8, options=None):
    """Teacher v2: each option is committed for ``hold_steps`` control steps (the student decides at the same
    period), then the copied episode continues under the deployable ``base`` option for ``tail`` steps, so
    the score sees whether the option set up a clean pass 2-3 s later (v1 saw only 1 s of the option itself)."""
    options = range(len(OPTION_NAMES)) if options is None else options
    before = ep.env.positions_mm[:ep.n].copy(); costs = np.full(len(OPTION_NAMES), np.inf)
    for option in options:
        trial = copy.deepcopy(ep); trial.prev_local = ep.prev_local.copy(); total = 0.
        for step in range(hold_steps+tail):
            t_est, _, t_rule, t_hold = trial.observe()
            done, out = trial.step(t_est, option_local(trial, t_est, t_rule, option if step < hold_steps else base), t_hold)
            total += _score(trial, before, out, trial.env.positions_mm[:trial.n])
            if done: break
        trial.close(); costs[option] = total
    return int(np.argmin(costs)), costs


# ---------------------------------------------------------------------------------------------------------------
# Primitive action set (v3). Every action is a motion primitive of the same level: a direction held for one
# decision period, defined in the local geometry (route direction d, nearest detected obstacle, map axis).
# No avoidance algorithm is embedded; heuristics (APF, rule_switch) are baselines only. The actuator runs
# nonzero commands at constant speed, so 'slow' is a 50 % duty cycle of 'advance'. No dead zone.
PRIMITIVE_NAMES = ('advance', 'pass_left', 'pass_right', 'away', 'center', 'slow', 'stop')
K_LAT = .8           # lateral component of the steering primitives (|lateral| / |d|)
OBS_ACTIVE_MM = 1.   # obstacle-relative primitives use the nearest detected obstacle within this surface gap


def _nearest(ep, est, i):
    best = None
    for rel, _, r in est.obstacles[i]:
        g = float(np.linalg.norm(rel))-ep.ctl.body-r
        if g < OBS_ACTIVE_MM and (best is None or g < best[0]):
            best = (g, rel)
    return best


def primitive_local(ep, est, rule, action):
    """Local-frame command of a primitive; reads only deployable estimates (detections, map, own estimate)."""
    u = np.zeros((ep.n, 3)); F = ep.ctl.frames(est)
    step = int(round(ep.env.elapsed_s/ep.env.config.control_dt_s))
    for i in range(ep.n):
        if not est.active[i]:
            continue
        d = np.asarray(ep.ctl.nominal[i], float); nd = np.linalg.norm(d)
        if nd < 1e-9:
            continue
        d = d/nd; name = PRIMITIVE_NAMES[action]; lat = np.zeros(3)
        if name in ('pass_left', 'pass_right', 'away'):
            nb = _nearest(ep, est, i)
            if nb is not None:
                o = F[i]@nb[1]; o = o-(o@d)*d; no = np.linalg.norm(o)
                if no < 1e-9:     # obstacle straight ahead: any lateral direction is 'away'
                    o = _basis(d)[0]; no = 1.
                o = o/no; b = np.cross(d, o)
                lat = {'pass_left': b, 'pass_right': -b, 'away': -o}[name]
        elif name == 'center':
            ax, _, _ = ep.sensor.map_coordinates(est, i); n = F[i]@(ax-est.pos[i]); n = n-(n@d)*d
            if np.linalg.norm(n) > 1e-9:
                lat = n/np.linalg.norm(n)
        if name == 'stop' or (name == 'slow' and step % 2):
            continue
        v = d+K_LAT*lat; u[i] = v/np.linalg.norm(v)
    return u


def label_primitives(ep, est, rule, hold, hold_steps=5, tail=25):
    """Teacher v2 over the primitive set: each primitive committed for hold_steps, then the copied episode
    continues under the deployable rule_switch for tail steps (value estimate only)."""
    before = ep.env.positions_mm[:ep.n].copy(); costs = np.full(len(PRIMITIVE_NAMES), np.inf)
    for a in range(len(PRIMITIVE_NAMES)):
        trial = copy.deepcopy(ep); trial.prev_local = ep.prev_local.copy(); total = 0.
        for step in range(hold_steps+tail):
            t_est, _, t_rule, t_hold = trial.observe()
            loc = primitive_local(trial, t_est, t_rule, a) if step < hold_steps else option_local(trial, t_est, t_rule, 8)
            done, out = trial.step(t_est, loc, t_hold)
            total += _score(trial, before, out, trial.env.positions_mm[:trial.n])
            if done: break
        trial.close(); costs[a] = total
    return int(np.argmin(costs)), costs


WALL_SAFE_MM = .35   # wall-aware primitives: below this map clearance, lateral steering may not point outward


def primitive_local_wa(ep, est, rule, action):
    """Wall-aware variant of primitive_local (same 7 actions). Near the wall (map clearance < WALL_SAFE_MM) the
    outward part of any lateral steering is removed and a pull toward the map axis is added, in proportion to
    how close the wall is. Uses the pre-operative map only (deployable). Motivation: under image sensing, plain
    route following has 4.7 s wall contact per scene, primitive policies 10-29 s (lateral steering into the wall)."""
    u = primitive_local(ep, est, rule, action); F = ep.ctl.frames(est)
    for i in range(ep.n):
        if not est.active[i] or not np.any(u[i]):
            continue
        d = np.asarray(ep.ctl.nominal[i], float); nd = np.linalg.norm(d)
        if nd < 1e-9:
            continue
        d = d/nd; ax, r_map, rad = ep.sensor.map_coordinates(est, i)
        gap = r_map-rad-ep.ctl.body; w = float(np.clip(1-gap/WALL_SAFE_MM, 0, 1))
        if w <= 0:
            continue
        n = F[i]@(ax-est.pos[i]); n = n-(n@d)*d
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n/np.linalg.norm(n); lat = u[i]-(u[i]@d)*d; out = min(float(lat@n), 0.)
        v = u[i]-w*out*n+w*K_LAT*.5*n
        u[i] = v/max(np.linalg.norm(v), 1e-9)
    return u
