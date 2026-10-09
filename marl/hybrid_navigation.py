"""Measurement-gated semantic passing, preserving the reference around large obstacles."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl.hierarchical_navigation import NavigationConfig, SemanticAvoidance, WallRecoveryExecutor, unit


@dataclass(frozen=True)
class HybridConfig(NavigationConfig):
    small_obstacle_max_radius_mm: float = .24

    def __post_init__(self):
        super().__post_init__()
        if self.small_obstacle_max_radius_mm <= 0:
            raise ValueError('Obstacle-size gate must be positive')


class HybridNavigator:
    def __init__(self, sensor, count, body, cfg=HybridConfig()):
        self.cfg, self.body = cfg, float(body)
        self.semantic = SemanticAvoidance(count, body, cfg)
        self.executor = WallRecoveryExecutor(sensor, count, body, cfg)
        self.last_modes = ['reference']*count

    def trigger(self, time_s, robots):
        self.executor.trigger(time_s, robots)

    def act(self, time_s, estimate, controller, reference, hold):
        frames = controller.frames(estimate)
        semantic = self.semantic.act(time_s, estimate, controller)
        guarded = self.executor.act(time_s, estimate, controller, semantic, hold)
        local = np.asarray(reference, float).copy()
        for robot in range(len(local)):
            if hold[robot] or not estimate.active[robot]:
                local[robot] = 0.
                self.last_modes[robot] = 'hold'
                continue
            heading = unit(frames[robot].T@controller.nominal[robot])
            obstacles = [(float(np.linalg.norm(relative))-self.body-radius, radius)
                         for relative, _, radius in estimate.obstacles[robot]
                         if float(relative@heading) > -.05]
            nearest = min(obstacles, key=lambda entry: entry[0]) if obstacles else None
            if time_s < self.executor.recover_until[robot]:
                local[robot] = guarded[robot]
                self.last_modes[robot] = 'recover'
            elif nearest is not None and nearest[0] < self.cfg.avoidance_trigger_mm and nearest[1] <= self.cfg.small_obstacle_max_radius_mm:
                local[robot] = guarded[robot]
                self.last_modes[robot] = 'small_obstacle_pass'
            else:
                self.last_modes[robot] = 'reference'
        return local
