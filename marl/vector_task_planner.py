"""Vector-env task-allocator wiring for EXP_0013.

The connectivity allocator (marl/connectivity_allocator.py) was written and
validated against the SINGLE env's attribute layout (1-D clot/robot arrays).
The training loop uses BalancedVectorVascularEnv, whose children are
VectorVascularEnv instances with batched [n_envs, ...] arrays. This module
bridges the two without touching either side:

  * ``_RowEnv`` presents one row of a VectorVascularEnv through the single-env
    attribute names the allocator already reads (``clot_masses`` 1-D,
    ``robot_positions``/``robot_stations`` [R, 3]/[R], ``active_clots`` slot
    count). Everything else (``tree``, ``flow_speed``, ``max_speed``,
    ``lysis_saturation``, ``_route``...) is forwarded to the child env, so the
    allocator sees the same shared-tree route caches the env itself uses.
  * ``replan_allocations(env, mode)`` re-plans ONE child env at a time and
    pushes the result through ``set_task_assignments``, which is the env's
    own supported override path. Padding slots (mask False) are parked at -1
    so a padded robot is never assigned a clot.

Re-plan policy (deliberately simple and recorded in the run config):
re-plan EVERY step. At 64 envs the allocator costs ~25 ms against a ~28 ms
env step (~1.9x wall-clock on the rollout phase), which is the honest price
of a closed-loop planner and is charged to the treatment arm alone.

The allocator emits targets only. The executed 3-D action remains the
direct-local Frenet output of the policy; the assignment changes which clot
the observation's shaping/lookahead/reward-approach term points at, exactly
as ``set_task_assignments`` was designed to do.
"""
from __future__ import annotations

import numpy as np

from marl.connectivity_allocator import allocate


class _RowEnv:
    """Single-row view of a VectorVascularEnv row for the allocator."""

    def __init__(self, env, row: int):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_row", int(row))

    def __getattr__(self, name):
        # Forward everything that is NOT a per-row property to the child env.
        return getattr(self._env, name)

    @property
    def clot_masses(self):
        return self._env.clot_masses[self._row]

    @property
    def robot_positions(self):
        return self._env.robot_positions[self._row]

    @property
    def robot_stations(self):
        return self._env.robot_stations[self._row]

    @property
    def robot_velocities(self):
        return self._env.robot_velocities[self._row]

    @property
    def active_clots(self):
        # The vector env has no per-row active count; the allocator uses this
        # only to bound `_alive_clots`'s slice, so the slot dimension is the
        # correct width. Dead slots are filtered by mass > 0 anyway.
        return self._env.clot_masses.shape[1]


def replan_child(child, mode: str) -> None:
    """Re-plan and apply allocations for one VectorVascularEnv child."""
    if mode == "none":
        return
    assignments = np.zeros((child.n_envs, child.num_robots), np.int32)
    for row in range(child.n_envs):
        view = _RowEnv(child, row)
        # Per-row previous plan feeds the allocator's switching term. Padding
        # slots were parked at -1, which the allocator reads as "no previous
        # target", exactly right.
        previous = None
        if getattr(child, "task_assignments", None) is not None:
            previous = np.asarray(child.task_assignments[row], np.int32)
        result = allocate(view, mode, previous_assignments=previous)
        # Park padding slots at -1: the env ignores them and the allocator's
        # switching term should not see phantom continuity either.
        assignments[row] = np.where(
            child.agent_mask[row], result.assignments, -1
        )
    child.set_task_assignments(assignments)


def replan_allocations(env, mode: str) -> None:
    """Re-plan every child of a BalancedVectorVascularEnv."""
    if mode == "none":
        return
    for child in env.envs:
        replan_child(child, mode)
