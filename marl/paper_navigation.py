"""Deployable CBF-QP and sampling-based MPC baselines.

Both controllers consume only the image-derived estimate, registered map SDF and
detected obstacle boxes. They never access simulator truth or copied episodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from marl.hierarchical_navigation import trajectory_cost, tube_clearance, unit
from marl.vessel_sdf import VesselSDF


@dataclass(frozen=True)
class CBFConfig:
    speed_mm_s: float = 1.0
    wall_margin_mm: float = .08
    obstacle_margin_mm: float = .08
    alpha: float = .5
    max_obstacles: int = 6
    active_gap_mm: float = 1.5
    projection_iterations: int = 8

    def __post_init__(self):
        values = (self.speed_mm_s, self.wall_margin_mm, self.obstacle_margin_mm, self.alpha,
                  self.active_gap_mm)
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError('Invalid CBF configuration')
        if self.max_obstacles < 1 or self.projection_iterations < 1:
            raise ValueError('Invalid CBF limits')


def _project_halfspaces(nominal, matrix, bounds, iterations=8):
    """Project a 3-D nominal vector onto measured linear CBF half-spaces.

    The active-set enumeration is the exact Euclidean projection for the small
    unconstrained QP when a feasible active set of at most three constraints
    exists. Sequential projection is retained only as a deterministic fallback
    for redundant or infeasible noisy constraints.
    """
    nominal = np.asarray(nominal, float)
    matrix = np.asarray(matrix, float).reshape(-1, 3)
    bounds = np.asarray(bounds, float)
    if not len(matrix):
        return nominal.copy(), 0.

    def feasible(vector):
        return float(np.max(bounds-matrix@vector)) <= 1e-8

    candidates = []
    if feasible(nominal):
        candidates.append(nominal.copy())
    count = len(matrix)
    for size in range(1, min(3, count)+1):
        for indices in combinations(range(count), size):
            active = matrix[list(indices)]
            gram = active@active.T
            try:
                multipliers = np.linalg.solve(gram, bounds[list(indices)]-active@nominal)
            except np.linalg.LinAlgError:
                continue
            if np.min(multipliers) < -1e-8:
                continue
            candidate = nominal+active.T@multipliers
            if feasible(candidate):
                candidates.append(candidate)
    if candidates:
        selected = min(candidates, key=lambda vector: float(np.sum((vector-nominal)**2)))
        return selected, float(np.max(bounds-matrix@selected))

    selected = nominal.copy()
    for _ in range(iterations):
        for row, bound in zip(matrix, bounds):
            violation = float(bound-row@selected)
            if violation > 0:
                selected += violation*row/max(float(row@row), 1e-12)
    return selected, float(max(np.max(bounds-matrix@selected), 0.))


class CBFQPFilter:
    """Measured linearized CBF-QP action filter around a nominal command."""
    def __init__(self, sensor, count, body, cfg=CBFConfig()):
        self.sensor, self.sdf = sensor, VesselSDF(sensor)
        self.body, self.cfg = float(body), cfg
        self.drift = np.zeros((count, 3))
        self.previous = np.zeros((count, 3))
        self.previous_time = np.full(count, np.nan)
        self.last_audit = []

    def _update_drift(self, time_s, estimate, robot):
        if np.any(self.previous[robot]) and np.isfinite(self.previous_time[robot]):
            dt = max(float(time_s-self.previous_time[robot]), 1e-6)
            observed = np.asarray(estimate.vel[robot], float)
            residual = np.clip(observed-self.cfg.speed_mm_s*self.previous[robot], -2., 2.)
            self.drift[robot] = .9*self.drift[robot]+.1*residual
        self.previous_time[robot] = float(time_s)

    def _constraints(self, estimate, robot):
        position = np.asarray(estimate.pos[robot], float)
        edge = int(estimate.edge[robot])
        clearance, gradient = self.sdf(position, edge)
        h = clearance-self.body-self.cfg.wall_margin_mm
        rows = [self.cfg.speed_mm_s*np.asarray(gradient, float)]
        bounds = [-self.cfg.alpha*h-float(np.asarray(gradient)@self.drift[robot])]
        detections = sorted(estimate.obstacles[robot], key=lambda item: np.linalg.norm(item[0])-item[2])
        for relative, relative_velocity, radius in detections[:self.cfg.max_obstacles]:
            relative = np.asarray(relative, float)
            distance = float(np.linalg.norm(relative))
            gap = distance-self.body-float(radius)
            if distance < 1e-8 or gap > self.cfg.active_gap_mm:
                continue
            normal = relative/distance
            obstacle_velocity = np.asarray(relative_velocity, float)+np.asarray(estimate.vel[robot], float)
            h_obstacle = gap-self.cfg.obstacle_margin_mm
            rows.append(-self.cfg.speed_mm_s*normal)
            bounds.append(-self.cfg.alpha*h_obstacle-float(normal@obstacle_velocity)
                          +float(normal@self.drift[robot]))
        return np.asarray(rows), np.asarray(bounds), float(clearance-self.body)

    def act(self, time_s, estimate, controller, nominal_world, hold):
        frames = controller.frames(estimate)
        output = np.asarray(nominal_world, float).copy()
        self.last_audit = []
        for robot in range(len(output)):
            if hold[robot] or not estimate.active[robot]:
                output[robot] = 0.
                continue
            self._update_drift(time_s, estimate, robot)
            nominal = frames[robot].T@output[robot]
            if not np.any(nominal):
                continue
            nominal = unit(nominal)
            matrix, bounds, clearance = self._constraints(estimate, robot)
            command, residual = _project_halfspaces(nominal, matrix, bounds, self.cfg.projection_iterations)
            command = command/max(float(np.linalg.norm(command)), 1e-9)
            output[robot] = frames[robot]@command
            self.previous[robot] = command
            self.last_audit.append(dict(robot=robot, map_clearance_mm=clearance,
                                       constraints=int(len(matrix)), residual_mm_s=residual,
                                       nominal=nominal.tolist(), command=command.tolist()))
        return output


@dataclass(frozen=True)
class MPPIConfig:
    speed_mm_s: float = 1.0
    horizon_steps: int = 8
    samples: int = 64
    noise_sigma: float = .35
    temperature: float = 1.
    wall_margin_mm: float = .08
    obstacle_margin_mm: float = .08
    wall_weight: float = 120.
    obstacle_weight: float = 80.
    progress_weight: float = 1.
    control_weight: float = .1
    seed: int = 20261007

    def __post_init__(self):
        if self.horizon_steps < 2 or self.samples < 8:
            raise ValueError('Invalid MPPI horizon or sample count')
        values = (self.speed_mm_s, self.noise_sigma, self.temperature, self.wall_margin_mm,
                  self.obstacle_margin_mm, self.wall_weight, self.obstacle_weight, self.progress_weight,
                  self.control_weight)
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError('Invalid MPPI configuration')


class MPPIController:
    """Sampling-based MPC with a measured kinematic model and registered map geometry."""
    def __init__(self, sensor, count, body, cfg=MPPIConfig()):
        self.sensor, self.sdf = sensor, VesselSDF(sensor)
        self.body, self.cfg = float(body), cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.drift = np.zeros((count, 3))
        self.previous = np.zeros((count, 3))
        self.last_audit = []

    def act(self, time_s, estimate, controller, nominal_world, hold):
        frames = controller.frames(estimate)
        output = np.asarray(nominal_world, float).copy()
        self.last_audit = []
        dt = float(controller.env.config.control_dt_s)
        for robot in range(len(output)):
            if hold[robot] or not estimate.active[robot]:
                output[robot] = 0.
                continue
            nominal = frames[robot].T@output[robot]
            if not np.any(nominal):
                continue
            nominal = unit(nominal)
            direction = nominal.copy()
            noise = self.rng.normal(0., self.cfg.noise_sigma,
                                    (self.cfg.samples, self.cfg.horizon_steps, 3))
            raw = nominal[None, None, :]+noise
            norms = np.linalg.norm(raw, axis=-1, keepdims=True)
            commands = raw/np.maximum(norms, 1e-9)
            positions = np.repeat(np.asarray(estimate.pos[robot], float)[None, :], self.cfg.samples, axis=0)
            costs = np.zeros(self.cfg.samples)
            detections = list(estimate.obstacles[robot])
            for step in range(self.cfg.horizon_steps):
                velocity = self.cfg.speed_mm_s*commands[:, step, :]+self.drift[robot]
                positions = positions+dt*velocity
                wall = np.asarray([self.sdf(position, int(estimate.edge[robot]))[0]-self.body
                                   for position in positions])
                costs += self.cfg.wall_weight*np.maximum(self.cfg.wall_margin_mm-wall, 0.)**2
                for relative, relative_velocity, radius in detections:
                    obstacle_velocity = np.asarray(relative_velocity, float)+np.asarray(estimate.vel[robot], float)
                    centers = np.asarray(estimate.pos[robot], float)+np.asarray(relative, float)+step*dt*obstacle_velocity
                    gap = np.linalg.norm(centers-positions, axis=1)-self.body-float(radius)
                    costs += self.cfg.obstacle_weight*np.maximum(self.cfg.obstacle_margin_mm-gap, 0.)**2
                costs -= self.cfg.progress_weight*dt*(velocity@direction)
                costs += self.cfg.control_weight*np.sum(noise[:, step, :]**2, axis=1)
            weights = np.exp(-(costs-costs.min())/self.cfg.temperature)
            weights /= max(float(weights.sum()), 1e-12)
            command = np.sum(weights[:, None]*commands[:, 0, :], axis=0)
            command = unit(command)
            output[robot] = frames[robot]@command
            self.previous[robot] = command
            self.last_audit.append(dict(robot=robot, selected_cost=float(costs@weights),
                                       minimum_cost=float(costs.min()), effective_samples=float(1./np.sum(weights**2)),
                                       command=command.tolist()))
        return output
