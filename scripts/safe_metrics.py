"""Safe-navigation metrics (primary metric from 2026-10-03).

SafeSuccess = TaskSuccess AND total wall contact over all robots < 1.0 robot-second.
TaskSuccess is the previous raw definition (every clot removed); it is kept as a secondary
metric, and RawSuccess - SafeSuccess (Unsafe Success Gap) measures reliance on wall contact.

`WallTracker` accumulates per-step `info['wall_contact_s']` (robot-seconds per robot in that
control step); `episode_metrics` turns one finished episode into the reported fields and
`aggregate` into rates. Episodes stored before this module existed have no wall data and are
never counted as safe or unsafe (their safe fields are absent, not zero).
"""
from __future__ import annotations

import numpy as np

SAFE_WALL_CONTACT_S = 1.0


class WallTracker:
    def __init__(self, num_robots):
        self.total = np.zeros(num_robots)
        self.run = np.zeros(num_robots)
        self.max_run = np.zeros(num_robots)
        self.active_robot_s = 0.

    def update(self, info, active_before, dt):
        """`active_before`: robots active during the step; `dt`: step duration in s."""
        wall = np.asarray(info['wall_contact_s'], np.float64)
        self.total += wall
        touching = wall > 0
        self.run = np.where(touching, self.run+wall, 0.)
        self.max_run = np.maximum(self.max_run, self.run)
        self.active_robot_s += float(np.sum(active_before))*dt


def episode_metrics(info, tracker, initial_mass_sum):
    total = float(tracker.total.sum())
    task = bool(info['success'])
    return dict(
        task_success=task,
        safe_success=bool(task and total < SAFE_WALL_CONTACT_S),
        collision_free=bool(info['collision_free_success']),
        safe_collision_free=bool(info['collision_free_success'] and total < SAFE_WALL_CONTACT_S),
        wall_contact_s=total,
        wall_contact_ratio=total/max(tracker.active_robot_s, 1e-12),
        max_continuous_wall_contact_s=float(tracker.max_run.max()) if len(tracker.max_run) else 0.,
        timeout=info.get('termination_reason') == 'time_limit',
        removal=float(1-info['remaining_mass']/initial_mass_sum),
        elapsed_s=float(info['elapsed_s']),
        termination_reason=info.get('termination_reason'))


def aggregate(rows):
    """Rates over episodes that carry wall data."""
    rows = [r for r in rows if 'safe_success' in r]
    if not rows:
        return {}
    v = lambda k: np.array([float(r[k]) for r in rows])
    raw, safe = v('task_success').mean(), v('safe_success').mean()
    return dict(episodes=len(rows), raw_success=raw, safe_success=safe, unsafe_success_gap=raw-safe,
                collision_free=v('collision_free').mean(), safe_collision_free=v('safe_collision_free').mean(),
                mean_wall_contact_s=v('wall_contact_s').mean(), median_wall_contact_s=float(np.median(v('wall_contact_s'))),
                wall_contact_ratio=v('wall_contact_ratio').mean(),
                max_continuous_wall_contact_s=v('max_continuous_wall_contact_s').mean(),
                timeout_rate=v('timeout').mean(), removal=v('removal').mean(), elapsed_s=v('elapsed_s').mean())
