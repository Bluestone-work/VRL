"""The four-metric evaluator shared by every V3 experiment.

Primary metrics (one per evaluation dimension)
  1 Navigation safety       safe_success: task success AND zero wall contact AND zero
                            dynamic-obstacle contact AND no lost cluster (contact-free).
  2 Navigation success      success: every target delivered (its assigned cluster within
                            the arrival tolerance) before the deadline.
  3 Delivery efficiency     T_last: deadline-scored last target arrival, measured from
                            catheter insertion (deployment and waiting included);
                            failures score the deadline.  Arrival time is a delivery
                            proxy: release and dose are not modelled.
  4 Interaction conflict    conflict: at least one control step with two released,
                            active clusters closer than d_safe (true centre distance).

Every record also separates
  A planned potential conflicts (from the configuration, reported by the planner),
  B executed spacing violations (this module, ground truth, never visible to a controller),
  C safety-layer interventions (shield rewrites, wall projection rewrites, TPG holds,
    counted at every step, not only the last),
  D physical magnetic interference: NOT modelled; coupling_exposure is a distance proxy.

Sampling: quantities are evaluated at the end of each 0.1 s control step on true simulator
state, except spacing_violation_pair_s_substep which comes from the substep-resolved
SpacingTracker of the simulator.
"""
from __future__ import annotations

import numpy as np

METRIC_VERSION = 'v3_four_metrics_1'


class FourMetricRecorder:
    def __init__(self, n, n_targets, *, d_safe, body_radius, deadline, tol, target_owner):
        self.n, self.m = int(n), int(n_targets)
        self.d_safe, self.body, self.deadline, self.tol = float(d_safe), float(body_radius), float(deadline), float(tol)
        self.owner = list(target_owner)                       # cluster assigned to each target
        self.arrival = np.full(self.m, np.nan)
        self.release = np.full(self.n, np.nan)
        self.lost_time = np.full(self.n, np.nan)
        self.wall_s = np.zeros(self.n); self.wall_after_s = np.zeros(self.n)
        self.obstacle_contact_s = np.zeros(self.n); self.obstacle_events = np.zeros(self.n, int)
        self.min_clearance = np.full(self.n, np.inf)
        self.pair_exposed_s = 0.; self.pair_violation_s = 0.
        self.min_pair = np.inf; self.events = 0; self.event_durations = []
        self._in = np.zeros((self.n, self.n), bool); self._start = np.zeros((self.n, self.n))
        self.coupling = 0.
        self.shield_steps = np.zeros(self.n, int); self.wall_rewrite_steps = np.zeros(self.n, int)
        self.hold_s = np.zeros(self.n); self.hold_run = np.zeros(self.n); self.max_hold_run = np.zeros(self.n)
        self.cmd_speed_sum = 0.; self.act_speed_sum = 0.; self.motion_samples = 0
        self.t = 0.

    def mark_release(self, i, t):
        self.release[i] = t

    def mark_arrival(self, target, t):
        if not np.isfinite(self.arrival[target]):
            self.arrival[target] = t

    def update(self, t, dt, *, positions, active, wall_s, obstacle_contact_s, obstacle_events, lost_now,
               clearance, hold, shield_changed, wall_rewritten, cmd_world, actual_velocity, delivered_mask):
        self.t = float(t)
        active = np.asarray(active, bool)
        self.wall_s += np.asarray(wall_s, float)
        self.wall_after_s += np.asarray(wall_s, float) * delivered_mask
        self.obstacle_contact_s += np.asarray(obstacle_contact_s, float)
        self.obstacle_events += np.asarray(obstacle_events, int)
        for i in np.flatnonzero(lost_now):
            if not np.isfinite(self.lost_time[i]):
                self.lost_time[i] = t
        c = np.asarray(clearance, float)
        self.min_clearance[active] = np.minimum(self.min_clearance[active], c[active])
        h = np.asarray(hold, bool) & active
        self.hold_s += h * dt
        self.hold_run = np.where(h, self.hold_run + dt, 0.)
        self.max_hold_run = np.maximum(self.max_hold_run, self.hold_run)
        self.shield_steps += np.asarray(shield_changed, bool) & active
        self.wall_rewrite_steps += np.asarray(wall_rewritten, bool) & active
        if active.any():
            cs = np.linalg.norm(np.asarray(cmd_world)[active], axis=1); vs = np.linalg.norm(np.asarray(actual_velocity)[active], axis=1)
            self.cmd_speed_sum += float(cs.sum()); self.act_speed_sum += float(vs.sum()); self.motion_samples += int(active.sum())
        P = np.asarray(positions, float)
        for a in range(self.n):
            for b in range(a + 1, self.n):
                if not (active[a] and active[b]):
                    if self._in[a, b]:
                        self.event_durations.append(t - self._start[a, b]); self._in[a, b] = False
                    continue
                d = float(np.linalg.norm(P[a] - P[b]))
                self.min_pair = min(self.min_pair, d)
                self.pair_exposed_s += dt
                if d < 2 * self.d_safe:
                    self.coupling += min((self.d_safe / max(d, 1e-6)) ** 3, 1e3) * dt
                v = d < self.d_safe
                if v:
                    self.pair_violation_s += dt
                    if not self._in[a, b]:
                        self.events += 1; self._start[a, b] = t - dt
                elif self._in[a, b]:
                    self.event_durations.append(t - self._start[a, b])
                self._in[a, b] = v

    def summary(self, substep_violation_pair_s=None):
        for a in range(self.n):
            for b in range(a + 1, self.n):
                if self._in[a, b]:
                    self.event_durations.append(self.t - self._start[a, b])
        delivered = np.isfinite(self.arrival) & (self.arrival <= self.deadline + 1e-9)
        success = bool(delivered.all())
        lost = int(np.isfinite(self.lost_time).sum())
        wall_total = float(self.wall_s.sum()); obst = float(self.obstacle_contact_s.sum())
        safe = bool(success and wall_total <= 1e-12 and obst <= 1e-12 and int(self.obstacle_events.sum()) == 0 and lost == 0)
        arr = np.where(delivered, self.arrival, self.deadline)
        rel = self.release[np.isfinite(self.release)]
        finish = []
        for i in range(self.n):
            mine = [j for j in range(self.m) if self.owner[j] == i]
            end = max([arr[j] for j in mine] or [self.release[i] if np.isfinite(self.release[i]) else 0.])
            if np.isfinite(self.lost_time[i]):
                end = max(end, self.lost_time[i]) if not mine else self.deadline
            start = self.release[i] if np.isfinite(self.release[i]) else self.deadline
            finish.append(max(0., end - start))
        conflict = bool(self.pair_violation_s > 0)
        return dict(
            metric_version=METRIC_VERSION,
            # 1 safety
            safe_success=safe,
            low_contact_success=bool(success and wall_total < 1. and lost == 0),
            wall_contact_robot_s=wall_total, wall_contact_after_delivery_robot_s=float(self.wall_after_s.sum()),
            obstacle_contact_robot_s=obst, obstacle_collision_events=int(self.obstacle_events.sum()),
            obstacle_collision=bool(self.obstacle_events.sum() > 0),
            lost_clusters=lost, lost_any=bool(lost > 0),
            min_wall_clearance_mm=float(np.min(self.min_clearance)) if np.isfinite(self.min_clearance).any() else None,
            # 2 success
            success=success, targets_delivered=int(delivered.sum()), target_coverage=float(delivered.mean()),
            per_target_arrived=delivered.astype(int).tolist(), timeout=bool(not success and lost == 0),
            failure_reason=None if success else ('lost_cluster' if lost else 'timeout'),
            # 3 efficiency
            T_last_deadline_scored_s=float(arr.max()),
            T_last_success_s=float(arr.max()) if success else None,
            mean_target_arrival_s=float(arr.mean()),
            arrival_span_s=float(arr.max() - arr.min()) if success else None,
            deployment_time_s=float(rel.max()) if len(rel) else None,
            total_cluster_time_s=float(np.sum(finish)),
            arrival_times_s=[None if not np.isfinite(x) else float(x) for x in self.arrival],
            release_times_s=[None if not np.isfinite(x) else float(x) for x in self.release],
            # 4 interaction
            conflict=conflict,
            spacing_violation_pair_s=float(self.pair_violation_s),
            spacing_violation_pair_s_substep=substep_violation_pair_s,
            exposed_pair_s=float(self.pair_exposed_s),
            violation_per_exposed_pair_s=(float(self.pair_violation_s / self.pair_exposed_s)
                                          if self.pair_exposed_s > 0 else None),
            min_inter_cluster_distance_mm=float(self.min_pair) if np.isfinite(self.min_pair) else None,
            conflict_events=int(self.events),
            mean_conflict_duration_s=float(np.mean(self.event_durations)) if self.event_durations else 0.,
            coupling_exposure_proxy=float(self.coupling),
            deadlock=bool(np.any(self.max_hold_run >= 60.)),
            # C: interventions
            tpg_hold_robot_s=float(self.hold_s.sum()), max_continuous_hold_s=float(self.max_hold_run.max()),
            shield_intervention_steps=int(self.shield_steps.sum()),
            wall_projection_rewrite_steps=int(self.wall_rewrite_steps.sum()),
            # motion bookkeeping
            mean_command_speed_mm_s=self.cmd_speed_sum / max(self.motion_samples, 1),
            mean_actual_speed_mm_s=self.act_speed_sum / max(self.motion_samples, 1),
        )
