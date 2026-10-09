"""Route-frontier stall detection and committed obstacle-tangent local recovery.

Only measured positions/detections and the registered map enter this controller.
Healthy-map clearance is a planning proxy, not a safety certificate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl.hierarchical_navigation import tube_clearance, unit
from marl.hybrid_navigation import HybridConfig, HybridNavigator


@dataclass(frozen=True)
class ProgressConfig(HybridConfig):
    route_stall_s: float = 5.
    route_progress_mm: float = .3
    route_event_cooldown_s: float = 5.
    goal_exclusion_mm: float = 1.
    large_obstacle_gap_mm: float = 1.
    bypass_buffer_mm: float = .12
    bypass_buffer_fraction: float = .06
    bypass_radial_gain: float = 1.5
    bypass_probe_mm: float = .4
    bypass_wall_margin_mm: float = .12
    semantic_min_clearance_mm: float = .3

    def __post_init__(self):
        super().__post_init__()
        values = [self.route_stall_s, self.route_progress_mm, self.route_event_cooldown_s,
                  self.goal_exclusion_mm, self.large_obstacle_gap_mm, self.bypass_buffer_mm,
                  self.bypass_probe_mm, self.bypass_wall_margin_mm]
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError('Invalid route recovery configuration')
        if self.bypass_buffer_fraction < 0 or self.bypass_radial_gain <= 0:
            raise ValueError('Invalid bypass clearance controller')
        if self.semantic_min_clearance_mm <= 0:
            raise ValueError('Invalid semantic clearance gate')


def remaining_route_mm(controller, robot, position):
    route = controller.route[robot]
    if route is None or not len(route):
        return np.inf
    progress = min(int(controller.prog[robot]), len(route)-1)
    points = controller.pts[route[progress:]]
    return float(np.linalg.norm(position-points[0])+np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def route_subgoal(controller, robot, distance_mm):
    route = controller.route[robot]
    progress = min(int(controller.prog[robot]), len(route)-1)
    points = controller.pts[route[progress:]]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    index = min(int(np.searchsorted(arc, distance_mm)), len(points)-1)
    return points[index].copy()


class RouteProgressMonitor:
    """Track a target's best remaining route distance, even across station-index resets."""
    def __init__(self, count, cfg=ProgressConfig()):
        self.cfg = cfg
        self.goal = np.full(count, -1)
        self.best = np.full(count, np.inf)
        self.improved_at = np.zeros(count)
        self.last_event = np.full(count, -np.inf)
        self.events = []

    def update(self, time_s, estimate, targets, hold, controller, busy=None):
        busy = np.zeros(len(targets), bool) if busy is None else busy
        triggered = []
        for robot, target in enumerate(targets):
            target = int(target)
            remaining = remaining_route_mm(controller, robot, estimate.pos[robot])
            if target != self.goal[robot] or hold[robot] or not estimate.active[robot] or target < 0:
                self.goal[robot] = target
                self.best[robot] = remaining
                self.improved_at[robot] = time_s
            if target < 0 or hold[robot] or not estimate.active[robot] or not np.isfinite(remaining):
                continue
            if remaining < self.best[robot]-self.cfg.route_progress_mm:
                self.best[robot] = remaining
                self.improved_at[robot] = time_s
            if np.linalg.norm(controller.env.clot_positions_mm[target]-estimate.pos[robot]) < self.cfg.goal_exclusion_mm:
                self.improved_at[robot] = time_s
                continue
            if busy[robot] or time_s-self.improved_at[robot] < self.cfg.route_stall_s:
                continue
            if time_s-self.last_event[robot] < self.cfg.route_event_cooldown_s:
                continue
            large = [(float(np.linalg.norm(relative))-controller.body-radius, relative, radius)
                     for relative, _, radius in estimate.obstacles[robot]
                     if radius > self.cfg.small_obstacle_max_radius_mm]
            if not large or min(entry[0] for entry in large) > self.cfg.large_obstacle_gap_mm:
                continue
            self.last_event[robot] = time_s
            event = dict(time_s=float(time_s), robot=robot, target=target, reason='route_frontier_stall',
                         remaining_mm=remaining, best_remaining_mm=float(self.best[robot]))
            self.events.append(event)
            triggered.append(robot)
        return triggered


class ProgressNavigator(HybridNavigator):
    """Keep v6 until route progress stalls, then pass a measured large obstacle consistently."""
    def __init__(self, sensor, count, body, cfg=ProgressConfig()):
        super().__init__(sensor, count, body, cfg)
        self.center = np.zeros((count, 3))
        self.radius = np.zeros(count)
        self.subgoal = np.zeros((count, 3))
        self.until = np.full(count, -np.inf)
        self.tangent = np.zeros((count, 3))
        self.goal = np.full(count, -1)
        self.last_plan = []

    def active(self, time_s):
        return time_s < self.until

    def start(self, time_s, robots, estimate, targets, controller):
        started = []
        for robot in robots:
            obstacles = [(float(np.linalg.norm(relative))-self.body-radius, relative, radius)
                         for relative, _, radius in estimate.obstacles[robot]
                         if radius > self.cfg.small_obstacle_max_radius_mm]
            if not obstacles:
                continue
            _, relative, radius = min(obstacles, key=lambda entry: entry[0])
            self.center[robot] = estimate.pos[robot]+relative
            self.radius[robot] = radius
            self.subgoal[robot] = route_subgoal(controller, robot, max(2., 2.5*radius))
            self.until[robot] = time_s+min(max(5., 4.+2.5*radius), 25.)
            self.tangent[robot] = 0.
            self.goal[robot] = int(targets[robot])
            started.append(robot)
        return started

    def _command(self, robot, estimate):
        position = estimate.pos[robot]
        radial = unit(position-self.center[robot])
        distance = float(np.linalg.norm(position-self.center[robot]))
        desired = unit(self.subgoal[robot]-position)
        buffer = self.cfg.bypass_buffer_mm+self.cfg.bypass_buffer_fraction*self.radius[robot]
        safe_radius = self.radius[robot]+self.body+buffer
        reference = np.eye(3)[int(np.argmin(np.abs(radial)))]
        first = unit(np.cross(radial, reference))
        second = np.cross(radial, first)
        angles = np.arange(16)*np.pi/8
        tangents = np.cos(angles)[:, None]*first+np.sin(angles)[:, None]*second
        outward = self.cfg.bypass_radial_gain*float(np.clip((safe_radius-distance)/max(buffer, .01), 0., 1.))
        commands = tangents+outward*radial
        commands /= np.maximum(np.linalg.norm(commands, axis=1, keepdims=True), 1.)
        forward = desired-min(float(desired@radial), 0.)*radial+outward*radial
        commands = np.vstack([commands, unit(forward)])
        probe = np.linspace(.1, self.cfg.bypass_probe_mm, 4)
        positions = position+commands[:, None]*probe[None, :, None]
        wall = tube_clearance(positions, self.executor.sdf, self.body, int(estimate.edge[robot]))
        gap = np.linalg.norm(positions-self.center[robot], axis=-1)-self.body-self.radius[robot]
        cost = 100.*np.maximum(self.cfg.bypass_wall_margin_mm-wall, 0.).max(axis=1)**2
        cost += 100.*np.maximum(buffer-gap, 0.).max(axis=1)**2
        cost -= commands@desired
        if np.any(self.tangent[robot]):
            cost += .1*np.sum((commands-self.tangent[robot])**2, axis=1)
        for relative, _, radius in estimate.obstacles[robot]:
            center = position+relative
            if np.linalg.norm(center-self.center[robot]) < max(.3, .2*self.radius[robot]):
                continue
            other_gap = np.linalg.norm(positions-center, axis=-1)-self.body-radius
            cost += 40.*np.maximum(.08-other_gap, 0.).max(axis=1)**2
        selected = int(np.argmin(cost))
        self.tangent[robot] = commands[selected]
        self.last_plan.append(dict(robot=robot, subgoal=self.subgoal[robot].tolist(), obstacle_center=self.center[robot].tolist(),
                                   radius_mm=float(self.radius[robot]), predicted_wall_mm=float(wall[selected].min()),
                                   predicted_obstacle_mm=float(gap[selected].min()), cost=float(cost[selected])))
        passed = float(radial@desired) > .3 and distance > safe_radius+.2
        return commands[selected], passed

    def act_with_targets(self, time_s, estimate, targets, controller, reference, hold):
        local = super().act(time_s, estimate, controller, reference, hold)
        frames = controller.frames(estimate)
        self.last_plan = []
        for robot in range(len(local)):
            if int(targets[robot]) != self.goal[robot]:
                self.until[robot] = -np.inf
            if hold[robot] or not estimate.active[robot] or self.last_modes[robot] == 'recover':
                continue
            if time_s >= self.until[robot]:
                continue
            command, passed = self._command(robot, estimate)
            local[robot] = frames[robot]@command
            self.last_modes[robot] = 'large_obstacle_bypass'
            if passed or np.linalg.norm(self.subgoal[robot]-estimate.pos[robot]) < .5:
                self.until[robot] = -np.inf
        return local


class AllObstacleNavigator(HybridNavigator):
    """Use the measured semantic pass for every detected forward obstacle, then guard its lateral part."""
    def __init__(self, sensor, count, body, cfg=ProgressConfig()):
        super().__init__(sensor, count, body, cfg)
        self.last_plan = []

    def act_all(self, time_s, estimate, controller, reference, hold):
        self.last_plan = []
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
            forward = [d for d in estimate.obstacles[robot] if float(np.asarray(d[0])@heading) > -.05]
            clearance = self.executor.sdf(estimate.pos[robot], int(estimate.edge[robot]))[0]-self.body
            if forward and clearance >= self.cfg.semantic_min_clearance_mm:
                local[robot] = guarded[robot]
                self.last_modes[robot] = 'semantic_guard'
            elif forward:
                self.last_modes[robot] = 'clearance_reference'
            else:
                self.last_modes[robot] = 'reference'
        return local

    def act_all_no_gate(self, time_s, estimate, controller, reference, hold):
        """Ablation that keeps semantic passing and wallguard but removes the map-clearance switch."""
        self.last_plan = []
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
            forward = [d for d in estimate.obstacles[robot] if float(np.asarray(d[0])@heading) > -.05]
            if forward:
                local[robot] = guarded[robot]
                self.last_modes[robot] = 'semantic_guard_no_gate'
            else:
                self.last_modes[robot] = 'reference'
        return local

    def act_all_reference_guard(self, time_s, estimate, controller, reference, hold):
        """Semantic passing with the same measured wallguard also applied to route following.

        This is a separate candidate from ``act_all_no_gate``: the latter only invokes the
        executor while a forward obstacle is detected, while this variant guards the reference
        command whenever the deployable map estimate enters the recovery band.  It never uses
        simulator wall contact or obstacle truth.
        """
        self.last_plan = []
        frames = controller.frames(estimate)
        semantic = self.semantic.act(time_s, estimate, controller)
        local = np.asarray(reference, float).copy()
        selected = np.asarray(reference, float).copy()
        for robot in range(len(local)):
            if hold[robot] or not estimate.active[robot]:
                selected[robot] = 0.
                self.last_modes[robot] = 'hold'
                continue
            heading = unit(frames[robot].T@controller.nominal[robot])
            forward = any(float(np.asarray(d[0])@heading) > -.05 for d in estimate.obstacles[robot])
            if forward:
                selected[robot] = semantic[robot]
        guarded = self.executor.act(time_s, estimate, controller, selected, hold)
        for robot in range(len(local)):
            if hold[robot] or not estimate.active[robot]:
                local[robot] = 0.
                continue
            heading = unit(frames[robot].T@controller.nominal[robot])
            forward = any(float(np.asarray(d[0])@heading) > -.05 for d in estimate.obstacles[robot])
            local[robot] = guarded[robot]
            self.last_modes[robot] = 'semantic_reference_guard' if forward else 'reference_guard'
        return local
