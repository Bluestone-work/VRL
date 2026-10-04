"""Privileged training-time teacher: the route + avoid + wait controller.

Used only to label states (behaviour cloning, DAgger). It reads the known-map
route bearing, the sticky geodesic allocation and the predicted-clearance slots
of the routed observation, which the student never sees.

Teacher definition (frozen, = BASELINE_WAIT_PRIOR_ONLY, sealed 92.6% / 91.4% on MCA):
  configs/experiments/EXP_0044_WAIT_PRIOR_DYNAMICS.json with zero residual:
  own route bearing + 6 x observed particle repulsion, bounded in the Frenet frame,
  dropped to an exact stop when a slot forecasts clearance < 0.3 within 0.5 s, and commands
  shorter than 0.35 (stop deadzone, inherited from EXP43) also execute as a stop.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

TEACHER_CONFIG = 'configs/experiments/EXP_0044_WAIT_PRIOR_DYNAMICS.json'


def teacher_config(base):
    """Teacher prior settings on top of an environment config (observation must be routed_assigned_own)."""
    from environments.mca_physical_env import DynamicsConfig
    t = DynamicsConfig.from_json(TEACHER_CONFIG)
    return replace(base, action_prior=t.action_prior, action_avoid_gain=t.action_avoid_gain,
                   action_residual_scale=1., action_stop_deadzone=t.action_stop_deadzone,
                   action_wait_clearance=t.action_wait_clearance, action_wait_horizon_s=t.action_wait_horizon_s,
                   action_shield_clearance=0., action_shield_horizon_s=0.)


def teacher_label(env, teacher_cfg):
    """(local Frenet command [n,3], stop [n], subgoal clot index [n] or -1) for the env's current state."""
    saved = env.config
    try:
        env.config = teacher_cfg
        world = env._apply_action_prior(np.zeros((env.num_robots, 3)))
        subgoal = env._assigned_targets().copy()
    finally:
        env.config = saved
    frame = np.stack((env.tree.tangents[env.robot_stations], env.tree.normals[env.robot_stations],
                      env.tree.binormals[env.robot_stations]), axis=1)
    local = np.einsum('nij,nj->ni', frame, world)
    active = env.active[:env.num_robots]
    stop = (np.linalg.norm(world, axis=1) < 1e-12) & active
    subgoal = np.where(active & (subgoal >= 0) & (env.masses[subgoal.clip(0)] > 0), subgoal, -1)
    return local.astype(np.float32), stop, subgoal.astype(np.int64)


def execute_local(env, local):
    """World command for a Frenet-local action, float64 (same transform as the prior; inactive robots zero)."""
    frame = np.stack((env.tree.tangents[env.robot_stations], env.tree.normals[env.robot_stations],
                      env.tree.binormals[env.robot_stations]), axis=1).astype(np.float64)
    local = np.asarray(local, np.float64)
    local = np.clip(local, -1, 1); local /= np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1.)
    return np.einsum('nji,nj->ni', frame, local)*env.active[:env.num_robots, None]
