"""Measurement-only local navigation and event-triggered route recovery.

This is a classical controller, not a learned-policy result or a safety certificate.
Inputs are registered map geometry, detected obstacle boxes and robot estimates.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from marl.vessel_sdf import VesselSDF, wall_safe_projection


@dataclass(frozen=True)
class NavigationConfig:
    horizon_s: float = .5
    samples: int = 5
    obstacle_margin_mm: float = .08
    wall_margin_mm: float = .08
    obstacle_weight: float = 80.
    wall_weight: float = 120.
    progress_weight: float = 1.
    switching_weight: float = .08
    reference_weight: float = .5
    maximum_wait_s: float = 1.
    commitment_s: float = .5
    stall_window_s: float = 2.
    stall_displacement_mm: float = .15
    replan_cooldown_s: float = 3.
    drift_alpha: float = .1
    recovery_margin_mm: float = .12
    recovery_gain: float = .4
    recovery_duration_s: float = .5
    avoidance_trigger_mm: float = .7

    def __post_init__(self):
        if self.horizon_s <= 0 or self.samples < 2 or self.commitment_s < 0:
            raise ValueError('Invalid prediction horizon or commitment')
        if self.stall_window_s <= 0 or self.replan_cooldown_s < 0:
            raise ValueError('Invalid replanning timing')
        if not 0 <= self.drift_alpha <= 1:
            raise ValueError('Invalid drift filter')


def unit(vector):
    vector = np.asarray(vector, float)
    return vector/max(float(np.linalg.norm(vector)), 1e-12)


def tube_clearance(points, sdf, body, edge):
    """Conservative hard union; unrelated spatially overlapping branches are excluded."""
    indices = sdf.neigh[int(edge)] if sdf.neigh is not None else slice(None)
    start, delta, length2 = sdf.a[indices], sdf.ab[indices], sdf.l2[indices]
    relative = np.asarray(points, float)[..., None, :]-start
    fraction = np.clip(np.sum(relative*delta, axis=-1)/length2, 0., 1.)
    radial = np.linalg.norm(relative-fraction[..., None]*delta, axis=-1)
    radius = sdf.r0[indices]+fraction*(sdf.r1[indices]-sdf.r0[indices])
    return np.max(radius-radial, axis=-1)-body


def trajectory_cost(commands, direction, position, drift, speed, obstacles, sdf, body, edge, cfg):
    """Compare bounded commands using estimated short trajectories, never simulator copies."""
    times = np.linspace(cfg.horizon_s/cfg.samples, cfg.horizon_s, cfg.samples)
    velocities = speed*np.asarray(commands)+drift
    displacement = velocities[:, None, :]*times[None, :, None]
    points = position+displacement
    wall = tube_clearance(points, sdf, body, edge)
    wall_risk = np.mean(np.maximum(cfg.wall_margin_mm-wall, 0.)**2, axis=1)
    obstacle_risk = np.zeros(len(commands))
    minimum_gap = np.full(len(commands), np.inf)
    for relative, absolute_velocity, radius in obstacles:
        future = np.asarray(relative)+times[:, None]*absolute_velocity
        gap = np.linalg.norm(future[None]-displacement, axis=-1)-body-radius
        minimum_gap = np.minimum(minimum_gap, gap.min(axis=1))
        obstacle_risk += np.mean(np.maximum(cfg.obstacle_margin_mm-gap, 0.)**2, axis=1)
    progress = displacement[:, -1]@direction
    costs = cfg.wall_weight*wall_risk+cfg.obstacle_weight*obstacle_risk-cfg.progress_weight*progress
    return costs, wall.min(axis=1), minimum_gap


class EventReplanner:
    """Replan only on measured stall, with a cooldown and intentional-hold exclusion."""
    def __init__(self, count, cfg=NavigationConfig()):
        self.cfg = cfg
        self.history = [deque() for _ in range(count)]
        self.last = np.full(count, -np.inf)
        self.goal = np.full(count, -1)
        self.events = []

    def update(self, time_s, estimate, targets, previous_world, hold, controller):
        triggered = []
        for robot, history in enumerate(self.history):
            target = int(targets[robot])
            if target != self.goal[robot] or hold[robot] or not estimate.active[robot] or target < 0:
                history.clear()
                self.goal[robot] = target
            if hold[robot] or target < 0 or not estimate.active[robot]:
                continue
            if np.linalg.norm(previous_world[robot]) < .2:
                history.clear()
                continue
            history.append((float(time_s), estimate.pos[robot].copy()))
            while len(history) > 1 and history[1][0] <= time_s-self.cfg.stall_window_s:
                history.popleft()
            if time_s-history[0][0] < self.cfg.stall_window_s-1e-8:
                continue
            positions = np.stack([entry[1] for entry in history])
            displacement = float(np.linalg.norm(positions[-1]-positions[0]))
            excursion = float(np.max(np.linalg.norm(positions-positions[0], axis=1)))
            if max(displacement, excursion) >= self.cfg.stall_displacement_mm:
                continue
            if time_s-self.last[robot] < self.cfg.replan_cooldown_s:
                continue
            controller._plan(robot, estimate.pos[robot], target)
            self.last[robot] = time_s
            history.clear()
            event = dict(time_s=float(time_s), robot=robot, reason='commanded_stall', target=target)
            self.events.append(event)
            triggered.append(robot)
        return triggered


class LocalNavigator:
    """Global route supplies the heading; local geometry selects a safe motion correction."""
    def __init__(self, sensor, count, speed, body, cfg=NavigationConfig()):
        self.cfg, self.speed, self.body = cfg, float(speed), float(body)
        self.sensor, self.sdf = sensor, VesselSDF(sensor)
        self.drift = np.zeros((count, 3))
        self.previous = np.zeros((count, 3))
        self.previous_position = [None]*count
        self.previous_time = np.full(count, np.nan)
        self.selected = [None]*count
        self.selected_time = np.full(count, -np.inf)
        self.wait_start = np.full(count, np.nan)
        self.last_audit = []

    def candidates(self, estimate, robot, direction, rule):
        axis, _, _ = self.sensor.map_coordinates(estimate, robot)
        center = axis-estimate.pos[robot]
        center -= (center@direction)*direction
        reference = np.eye(3)[int(np.argmin(np.abs(direction)))]
        side = unit(np.cross(direction, reference))
        other = np.cross(direction, side)
        vectors = [direction, rule, unit(direction+.8*unit(center)), -self.drift[robot]/max(self.speed, 1e-12), .5*direction]
        names = ['advance', 'reference', 'center', 'wait', 'slow']
        for angle in np.arange(8)*np.pi/4:
            lateral = np.cos(angle)*side+np.sin(angle)*other
            vectors.append(unit(direction+.8*lateral))
            names.append(f'avoid_{int(round(angle*4/np.pi))}')
        if estimate.obstacles[robot]:
            nearest = min(estimate.obstacles[robot], key=lambda detection: np.linalg.norm(detection[0])-detection[2])
            away = -np.asarray(nearest[0], float)
            away -= (away@direction)*direction
            vectors.append(unit(direction+.8*unit(away)))
            names.append('avoid_away')
        world = np.asarray(vectors)
        for index in range(len(world)):
            world[index] = wall_safe_projection(world[index], estimate.pos[robot], self.sdf, self.body, edge=int(estimate.edge[robot]))
        return names, world

    def act(self, time_s, estimate, controller, rule_local, hold):
        frames = controller.frames(estimate)
        local = np.zeros_like(rule_local)
        self.last_audit = []
        for robot in range(len(local)):
            position = estimate.pos[robot]
            if not estimate.active[robot] or hold[robot] or not np.any(controller.nominal[robot]):
                self.previous[robot] = 0.
                self.previous_position[robot] = None
                continue
            if self.previous_position[robot] is not None:
                dt = float(time_s-self.previous_time[robot])
                if dt > 1e-8:
                    observed = (position-self.previous_position[robot])/dt
                    residual = np.clip(observed-self.speed*self.previous[robot], -.5, .5)
                    alpha = self.cfg.drift_alpha
                    self.drift[robot] = (1-alpha)*self.drift[robot]+alpha*residual
            direction = unit(frames[robot].T@controller.nominal[robot])
            rule = frames[robot].T@rule_local[robot]
            names, commands = self.candidates(estimate, robot, direction, rule)
            obstacles = [(relative, np.clip(relative_velocity+estimate.vel[robot], -1., 1.), radius)
                         for relative, relative_velocity, radius in estimate.obstacles[robot]]
            costs, wall, gap = trajectory_cost(commands, direction, position, self.drift[robot], self.speed,
                                               obstacles, self.sdf, self.body, int(estimate.edge[robot]), self.cfg)
            costs += self.cfg.switching_weight*np.sum((commands-self.previous[robot])**2, axis=1)
            costs += self.cfg.reference_weight*np.sum((commands-rule)**2, axis=1)
            if np.isfinite(self.wait_start[robot]) and time_s-self.wait_start[robot] >= self.cfg.maximum_wait_s:
                costs[names.index('wait')] = 1e6
            selected = int(np.argmin(costs))
            previous_name = self.selected[robot]
            if previous_name in names and time_s-self.selected_time[robot] < self.cfg.commitment_s:
                old = names.index(previous_name)
                if wall[old] >= 0 and gap[old] >= 0 and costs[old] <= costs[selected]+.05:
                    selected = old
            if names[selected] != previous_name:
                self.selected_time[robot] = time_s
            self.selected[robot] = names[selected]
            if names[selected] == 'wait':
                if not np.isfinite(self.wait_start[robot]):
                    self.wait_start[robot] = time_s
            else:
                self.wait_start[robot] = np.nan
            command = commands[selected]
            local[robot] = frames[robot]@command
            self.previous[robot] = command
            self.previous_position[robot] = position.copy()
            self.previous_time[robot] = time_s
            self.last_audit.append(dict(robot=robot, primitive=names[selected], costs=costs.tolist(),
                                        candidates=names, predicted_wall_mm=float(wall[selected]),
                                        predicted_obstacle_mm=float(gap[selected]) if np.isfinite(gap[selected]) else None,
                                        drift_mm_s=self.drift[robot].tolist(), proposed_world=command.tolist()))
        return local


class WallRecoveryExecutor:
    """Conservative reference execution; map-normal recovery only inside a small guard band."""
    def __init__(self, sensor, count, body, cfg=NavigationConfig()):
        self.sdf, self.body, self.cfg = VesselSDF(sensor), float(body), cfg
        self.recover_until = np.full(count, -np.inf)
        self.last_audit = []

    def trigger(self, time_s, robots):
        self.recover_until[robots] = time_s+self.cfg.recovery_duration_s

    def act(self, time_s, estimate, controller, reference, hold):
        frames = controller.frames(estimate)
        local = np.asarray(reference, float).copy()
        self.last_audit = []
        for robot in range(len(local)):
            if not estimate.active[robot] or hold[robot]:
                local[robot] = 0.
                continue
            command = frames[robot].T@local[robot]
            distance, gradient = self.sdf(estimate.pos[robot], int(estimate.edge[robot]))
            clearance = distance-self.body
            normal = unit(gradient)
            heading = unit(frames[robot].T@controller.nominal[robot])
            normal = unit(normal-(normal@heading)*heading)
            recovering = time_s < self.recover_until[robot]
            mode = 'reference'
            if recovering and np.any(controller.nominal[robot]):
                command = -.5*heading+.25*normal
                mode = 'recover'
            elif np.any(command) and clearance < self.cfg.recovery_margin_mm:
                scale = float(np.clip(1-clearance/self.cfg.recovery_margin_mm, 0., 1.))
                forward = float(command@heading)*heading
                lateral = command-forward
                outward = min(float(lateral@normal), 0.)
                command = command-scale*outward*normal
                if np.linalg.norm(lateral) > 1e-6:
                    command += self.cfg.recovery_gain*scale*normal
                command /= max(float(np.linalg.norm(command)), 1.)
                mode = 'wall_guard'
            local[robot] = frames[robot]@command
            self.last_audit.append(dict(robot=robot, primitive=mode, map_clearance_mm=float(clearance),
                                        proposed_world=command.tolist()))
        return local


class SemanticAvoidance:
    """Advance unless a forward detected obstacle requires a committed lateral pass."""
    def __init__(self, count, body, cfg=NavigationConfig()):
        self.body, self.cfg = float(body), cfg
        self.side = np.zeros((count, 3))
        self.until = np.full(count, -np.inf)
        self.modes = ['advance']*count

    def act(self, time_s, estimate, controller):
        frames = controller.frames(estimate)
        local = np.asarray(controller.nominal, float).copy()
        for robot in range(len(local)):
            heading = unit(frames[robot].T@local[robot])
            if not estimate.active[robot] or not np.any(heading):
                local[robot] = 0.
                continue
            detected = [(float(np.linalg.norm(relative))-self.body-radius, relative)
                        for relative, _, radius in estimate.obstacles[robot]
                        if float(relative@heading) > -.05]
            nearest = min(detected, key=lambda entry: entry[0]) if detected else None
            avoiding = nearest is not None and nearest[0] < self.cfg.avoidance_trigger_mm
            if avoiding:
                if time_s >= self.until[robot]:
                    lateral = -np.asarray(nearest[1], float)
                    lateral -= (lateral@heading)*heading
                    if np.linalg.norm(lateral) < 1e-8:
                        reference = np.eye(3)[int(np.argmin(np.abs(heading)))]
                        lateral = np.cross(heading, reference)
                    self.side[robot] = unit(lateral)
                    self.until[robot] = time_s+self.cfg.commitment_s
                lateral = self.side[robot]-(self.side[robot]@heading)*heading
                local[robot] = frames[robot]@unit(heading+.8*unit(lateral))
                self.modes[robot] = 'avoid'
            else:
                self.until[robot] = -np.inf
                self.modes[robot] = 'advance'
        return local


def relaxed_success(row):
    """Previously registered relaxed metric; dynamic contacts remain separately reported."""
    return bool(row['task_success'] and row['lost'] == 0 and row['obstacle_events_static'] == 0
                and row['wall_contact_s'] < 5. and row.get('robot_pair_contact_s', 0.) <= 1e-12
                and row.get('spacing', {}).get('spacing_compliant', True))
