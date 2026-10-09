"""Measured wall-risk residual with hysteretic local recovery.

The controller uses only deployable estimates and the registered pre-operative
map. It does not read simulator wall contact, true lumen geometry, or obstacle
labels. The risk mode is deliberately a residual over the semantic navigator:
the semantic pass remains responsible for detected obstacles and this module
only suppresses outward motion when the measured short-term state is risky.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl.hierarchical_navigation import unit
from marl.progress_navigation import AllObstacleNavigator, ProgressConfig, remaining_route_mm


@dataclass(frozen=True)
class RiskConfig(ProgressConfig):
    risk_enter_score: float = .55
    risk_exit_score: float = .30
    risk_enter_offset_fraction: float = .90
    risk_offset_full_fraction: float = 1.05
    risk_outward_velocity_mm_s: float = .15
    risk_outward_command: float = .20
    risk_stall_speed_mm_s: float = .20
    risk_filter_alpha: float = .25
    risk_enter_confirm_steps: int = 3
    risk_exit_confirm_steps: int = 5
    risk_min_hold_s: float = 1.0
    risk_rearm_offset_fraction: float = .80
    risk_guard_gain: float = .65

    def __post_init__(self):
        super().__post_init__()
        positive = (self.risk_outward_velocity_mm_s, self.risk_outward_command,
                    self.risk_stall_speed_mm_s, self.risk_min_hold_s,
                    self.risk_guard_gain)
        if any(not np.isfinite(value) or value <= 0 for value in positive):
            raise ValueError('Invalid wall-risk thresholds')
        if not 0 <= self.risk_exit_score < self.risk_enter_score <= 1:
            raise ValueError('Risk hysteresis scores must satisfy 0 <= exit < enter <= 1')
        if not 0 < self.risk_enter_offset_fraction < self.risk_offset_full_fraction:
            raise ValueError('Invalid wall-risk offset fractions')
        if not 0 < self.risk_rearm_offset_fraction < self.risk_enter_offset_fraction:
            raise ValueError('Invalid wall-risk rearm fraction')
        if not 0 < self.risk_filter_alpha <= 1:
            raise ValueError('Invalid wall-risk filter')
        if self.risk_enter_confirm_steps < 1 or self.risk_exit_confirm_steps < 1:
            raise ValueError('Risk confirmation steps must be positive')


class RiskHysteresisNavigator(AllObstacleNavigator):
    """Semantic passing plus measured, hysteretic suppression of outward drift."""

    def __init__(self, sensor, count, body, cfg=RiskConfig()):
        super().__init__(sensor, count, body, cfg)
        self.sensor = sensor
        self.cfg = cfg
        self.risk_active = np.zeros(count, dtype=bool)
        self.armed = np.ones(count, dtype=bool)
        self.enter_count = np.zeros(count, dtype=int)
        self.exit_count = np.zeros(count, dtype=int)
        self.last_enter = np.full(count, -np.inf)
        self.filtered_score = np.zeros(count)
        self.previous_offset_fraction = np.full(count, np.nan)
        self.previous_time = np.full(count, np.nan)
        self.last_risk = []
        self.risk_events = []

    def _reset(self, robot):
        self.risk_active[robot] = False
        self.armed[robot] = True
        self.enter_count[robot] = 0
        self.exit_count[robot] = 0
        self.filtered_score[robot] = 0.
        self.previous_offset_fraction[robot] = np.nan
        self.previous_time[robot] = np.nan

    def _score(self, robot, estimate, controller, command_world):
        axis, map_radius, radial_offset = self.sensor.map_coordinates(estimate, robot)
        offset_vector = estimate.pos[robot]-axis
        outward = unit(offset_vector) if radial_offset > 1e-8 else np.zeros(3)
        available = max(float(map_radius)-self.body, 1e-6)
        offset_fraction = float(radial_offset)/available
        offset_term = np.clip((offset_fraction-self.cfg.risk_enter_offset_fraction) /
                              (self.cfg.risk_offset_full_fraction-self.cfg.risk_enter_offset_fraction), 0., 1.)
        outward_velocity = max(float(np.asarray(estimate.vel[robot])@outward), 0.)
        velocity_term = np.clip(outward_velocity/self.cfg.risk_outward_velocity_mm_s, 0., 1.)
        now = getattr(controller.env, 'elapsed_s', np.nan)
        if np.isfinite(self.previous_time[robot]) and np.isfinite(now):
            dt = max(float(now)-self.previous_time[robot], 1e-6)
            outward_rate = max((offset_fraction-self.previous_offset_fraction[robot])/dt, 0.)
        else:
            outward_rate = 0.
        rate_term = np.clip(outward_rate/.50, 0., 1.)
        self.previous_offset_fraction[robot] = offset_fraction
        self.previous_time[robot] = now
        command_norm = float(np.linalg.norm(command_world))
        outward_command = max(float(np.asarray(command_world)@outward), 0.)/max(command_norm, 1e-8)
        command_term = np.clip(outward_command/self.cfg.risk_outward_command, 0., 1.)
        observed_speed = float(np.linalg.norm(estimate.vel[robot]))
        contact_term = float(offset_fraction >= self.cfg.risk_enter_offset_fraction and
                             observed_speed <= self.cfg.risk_stall_speed_mm_s and
                             outward_command >= self.cfg.risk_outward_command)
        dynamic = .55*velocity_term+.30*rate_term+.15*command_term*max(velocity_term, rate_term)
        raw = max(dynamic, .70*offset_term if contact_term else 0.)
        alpha = self.cfg.risk_filter_alpha
        self.filtered_score[robot] = (1-alpha)*self.filtered_score[robot]+alpha*raw
        return dict(score=float(self.filtered_score[robot]), raw_score=float(raw),
                    offset_fraction=float(offset_fraction), radial_offset_mm=float(radial_offset),
                    map_radius_mm=float(map_radius), outward_velocity_mm_s=float(outward_velocity),
                    outward_rate=float(outward_rate), outward_command=float(outward_command),
                    observed_speed_mm_s=observed_speed, stalled=bool(contact_term), axis=axis.tolist())

    def _update_mode(self, time_s, robot, risk):
        score = risk['score']
        if self.risk_active[robot]:
            self.enter_count[robot] = 0
            if time_s-self.last_enter[robot] < self.cfg.risk_min_hold_s:
                self.exit_count[robot] = 0
            elif score <= self.cfg.risk_exit_score:
                self.exit_count[robot] += 1
            else:
                self.exit_count[robot] = 0
            if self.exit_count[robot] >= self.cfg.risk_exit_confirm_steps:
                self.risk_active[robot] = False
                self.exit_count[robot] = 0
                self.risk_events.append(dict(time_s=float(time_s), robot=robot,
                                             event='exit', score=float(score)))
        else:
            self.exit_count[robot] = 0
            if not self.armed[robot]:
                if risk['offset_fraction'] <= self.cfg.risk_rearm_offset_fraction:
                    self.armed[robot] = True
                else:
                    self.enter_count[robot] = 0
                    return
            if score >= self.cfg.risk_enter_score:
                self.enter_count[robot] += 1
            else:
                self.enter_count[robot] = 0
            if self.enter_count[robot] >= self.cfg.risk_enter_confirm_steps:
                self.risk_active[robot] = True
                self.armed[robot] = False
                self.last_enter[robot] = time_s
                self.enter_count[robot] = 0
                self.risk_events.append(dict(time_s=float(time_s), robot=robot,
                                             event='enter', score=float(score)))

    def _guard(self, robot, estimate, controller, command_world, risk):
        frames = controller.frames(estimate)
        local = frames[robot].T@command_world
        axis, _, radial_offset = self.sensor.map_coordinates(estimate, robot)
        inward_world = axis-estimate.pos[robot]
        inward = unit(frames[robot].T@inward_world) if radial_offset > 1e-8 else np.zeros(3)
        heading = unit(frames[robot].T@controller.nominal[robot])
        outward = -inward
        outward_component = max(float(local@outward), 0.)
        tangent = local-outward_component*outward
        if np.linalg.norm(tangent) < 1e-8:
            tangent = heading
        severity = float(np.clip((risk['score']-self.cfg.risk_exit_score) /
                                 max(self.cfg.risk_enter_score-self.cfg.risk_exit_score, 1e-8), 0., 1.))
        corrected = local-severity*outward_component*outward
        if np.linalg.norm(corrected) < 1e-8:
            corrected = tangent
        guarded = corrected+self.cfg.risk_guard_gain*severity*inward
        return frames[robot]@unit(guarded)

    def act_risk(self, time_s, estimate, controller, reference, hold):
        self.last_plan = []
        self.last_risk = []
        frames = controller.frames(estimate)
        semantic = self.semantic.act(time_s, estimate, controller)
        guarded = self.executor.act(time_s, estimate, controller, semantic, hold)
        local = np.asarray(reference, float).copy()
        for robot in range(len(local)):
            if hold[robot] or not estimate.active[robot]:
                local[robot] = 0.
                self._reset(robot)
                self.last_modes[robot] = 'hold'
                self.last_risk.append(dict(robot=robot, score=0., active=False, reason='inactive'))
                continue
            heading = unit(frames[robot].T@controller.nominal[robot])
            forward = [d for d in estimate.obstacles[robot] if float(np.asarray(d[0])@heading) > -.05]
            base = guarded[robot] if forward else local[robot]
            risk = self._score(robot, estimate, controller, base)
            self._update_mode(time_s, robot, risk)
            if self.risk_active[robot] and not forward:
                local[robot] = self._guard(robot, estimate, controller, base, risk)
                self.last_modes[robot] = 'risk_guard'
            elif forward:
                local[robot] = guarded[robot]
                self.last_modes[robot] = 'semantic_guard' if not self.risk_active[robot] else 'semantic_guard_risk_hold'
            else:
                local[robot] = reference[robot]
                self.last_modes[robot] = 'reference'
            risk.update(robot=robot, active=bool(self.risk_active[robot]), forward_obstacle=bool(forward),
                        mode=self.last_modes[robot])
            self.last_risk.append(risk)
        return local


@dataclass(frozen=True)
class ProgressBoundRiskConfig(RiskConfig):
    progress_stall_s: float = 2.0
    progress_commit_s: float = 2.0
    progress_release_mm: float = .3

    def __post_init__(self):
        super().__post_init__()
        if any(not np.isfinite(value) or value <= 0 for value in
               (self.progress_stall_s, self.progress_commit_s, self.progress_release_mm)):
            raise ValueError('Invalid progress-bound risk timing')


class ProgressBoundRiskNavigator(RiskHysteresisNavigator):
    """Only allow risk recovery when a measured route frontier has stalled."""

    def __init__(self, sensor, count, body, cfg=ProgressBoundRiskConfig()):
        super().__init__(sensor, count, body, cfg)
        self.cfg = cfg
        self.progress_goal = np.full(count, -1, dtype=int)
        self.progress_best = np.full(count, np.inf)
        self.progress_improved_at = np.zeros(count)
        self.progress_guard_until = np.full(count, -np.inf)
        self.progress_guard_entry_best = np.full(count, np.inf)

    def _progress_state(self, time_s, robot, estimate, targets, controller, hold):
        target = int(targets[robot])
        remaining = remaining_route_mm(controller, robot, estimate.pos[robot])
        reset = (target != self.progress_goal[robot] or hold[robot] or
                 not estimate.active[robot] or target < 0 or not np.isfinite(remaining))
        if reset:
            self.progress_goal[robot] = target
            self.progress_best[robot] = remaining
            self.progress_improved_at[robot] = time_s
        elif remaining < self.progress_best[robot]-self.cfg.progress_release_mm:
            self.progress_best[robot] = remaining
            self.progress_improved_at[robot] = time_s
        stalled = bool(np.isfinite(remaining) and not hold[robot] and estimate.active[robot] and
                       target >= 0 and time_s-self.progress_improved_at[robot] >= self.cfg.progress_stall_s)
        improved = bool(np.isfinite(self.progress_guard_entry_best[robot]) and
                        np.isfinite(remaining) and
                        remaining < self.progress_guard_entry_best[robot]-self.cfg.progress_release_mm)
        return dict(target=target, remaining_mm=float(remaining), best_remaining_mm=float(self.progress_best[robot]),
                    stalled=stalled, improved=improved)

    def _force_release(self, time_s, robot, reason):
        if self.risk_active[robot]:
            self.risk_active[robot] = False
            self.exit_count[robot] = 0
            self.risk_events.append(dict(time_s=float(time_s), robot=robot,
                                         event='release', reason=reason))

    def act_progress_bound(self, time_s, estimate, controller, reference, hold, targets):
        self.last_plan = []
        self.last_risk = []
        frames = controller.frames(estimate)
        semantic = self.semantic.act(time_s, estimate, controller)
        guarded = self.executor.act(time_s, estimate, controller, semantic, hold)
        local = np.asarray(reference, float).copy()
        for robot in range(len(local)):
            if hold[robot] or not estimate.active[robot]:
                local[robot] = 0.
                self._reset(robot)
                self.progress_guard_until[robot] = -np.inf
                self.progress_guard_entry_best[robot] = np.inf
                self.last_modes[robot] = 'hold'
                self.last_risk.append(dict(robot=robot, score=0., active=False, reason='inactive'))
                continue
            progress = self._progress_state(time_s, robot, estimate, targets, controller, hold)
            heading = unit(frames[robot].T@controller.nominal[robot])
            forward = [d for d in estimate.obstacles[robot] if float(np.asarray(d[0])@heading) > -.05]
            base = guarded[robot] if forward else local[robot]
            risk = self._score(robot, estimate, controller, base)
            risk.update(progress)
            eligible = bool(risk['score'] >= self.cfg.risk_enter_score and progress['stalled'] and not forward)
            if not self.risk_active[robot] and not eligible:
                self.enter_count[robot] = 0
            self._update_mode(time_s, robot, risk)
            if self.risk_active[robot] and self.progress_guard_until[robot] == -np.inf:
                self.progress_guard_until[robot] = time_s+self.cfg.progress_commit_s
                self.progress_guard_entry_best[robot] = progress['best_remaining_mm']
                self.risk_events.append(dict(time_s=float(time_s), robot=robot,
                                             event='progress_guard_start',
                                             remaining_mm=progress['remaining_mm']))
            if self.risk_active[robot] and (progress['improved'] or
                                            time_s >= self.progress_guard_until[robot]):
                self._force_release(time_s, robot, 'progress' if progress['improved'] else 'timeout')
                self.progress_guard_until[robot] = -np.inf
                self.progress_guard_entry_best[robot] = np.inf
            if self.risk_active[robot] and not forward:
                local[robot] = self._guard(robot, estimate, controller, base, risk)
                self.last_modes[robot] = 'progress_risk_guard'
            elif forward:
                local[robot] = guarded[robot]
                self.last_modes[robot] = 'semantic_guard'
            else:
                local[robot] = reference[robot]
                self.last_modes[robot] = 'reference'
            risk.update(robot=robot, active=bool(self.risk_active[robot]), forward_obstacle=bool(forward),
                        mode=self.last_modes[robot])
            self.last_risk.append(risk)
        return local
