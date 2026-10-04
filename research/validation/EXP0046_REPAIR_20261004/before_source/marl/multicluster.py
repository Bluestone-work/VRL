"""Cluster-level controllers for the next VRL experiment.

The physical environment already treats each controlled body as an independently
commandable magnetic entity.  This module gives those entities a cluster-level
protocol without changing the vessel, flow, clot or lysis model:

* ``single_sequential`` is the method-1 baseline.  One cluster receives a
  pre-registered clot order and clears one target at a time.
* ``multi_parallel`` is method 2.  Each cluster follows its own nearest clot
  from the fair local observation.  A short-horizon right-of-way shield stops
  the lower-priority cluster when two predicted centroids would violate
  ``min_spacing_mm``.  Persistent mutual waiting is resolved by alternating
  priority, and is reported as a deadlock event.

The controller does not read route tables, geodesic assignments, or future
states in ``multi_parallel``.  The spacing shield uses only current cluster
positions and the local flow velocity already available to the actuator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from marl.fair_reactive import fair_reactive_action
from marl.partial_obs import PartialObserver, PartialObsConfig, CLOT0, CLOT_SLOTS
from marl.teacher import execute_local, teacher_label


@dataclass(frozen=True)
class MultiClusterConfig:
    method: str = "multi_parallel"
    clusters: int = 2
    min_spacing_mm: float = 2.0
    deadlock_window_steps: int = 10
    wait_horizon_s: float = 0.5
    observation_noise: float = 0.025

    def __post_init__(self):
        if self.method not in ("single_sequential", "multi_parallel"):
            raise ValueError("method must be single_sequential or multi_parallel")
        if (not isinstance(self.clusters, int) or isinstance(self.clusters, bool)
                or self.clusters < 1):
            raise ValueError("clusters must be a positive integer")
        if self.min_spacing_mm < 0 or not np.isfinite(self.min_spacing_mm):
            raise ValueError("min_spacing_mm must be finite and non-negative")
        if (not isinstance(self.deadlock_window_steps, int)
                or isinstance(self.deadlock_window_steps, bool)
                or self.deadlock_window_steps < 1):
            raise ValueError("deadlock_window_steps must be positive")
        if not np.isfinite(self.wait_horizon_s) or self.wait_horizon_s < 0:
            raise ValueError("wait_horizon_s must be finite and non-negative")


def balanced_cluster_ids(num_entities: int, clusters: int) -> np.ndarray:
    """Deterministic balanced partition; one entity is one magnetic cluster."""
    if clusters > num_entities:
        raise ValueError("clusters cannot exceed the number of controlled entities")
    result = np.empty(num_entities, dtype=np.int32)
    for cluster, indices in enumerate(np.array_split(np.arange(num_entities), clusters)):
        result[indices] = cluster
    return result


def _cluster_centroids(env, cluster_ids: np.ndarray) -> np.ndarray:
    centres = np.full((int(cluster_ids.max()) + 1, 3), np.nan, dtype=np.float64)
    for cluster in range(len(centres)):
        members = np.flatnonzero((cluster_ids == cluster) & env.active[:env.num_robots])
        if len(members):
            centres[cluster] = env.positions_mm[members].mean(axis=0)
    return centres


def _target_slot_from_global_position(obs_row: np.ndarray, target: np.ndarray) -> int:
    """Match a pre-planned target to a local clot slot without exposing a map."""
    best, error = -1, np.inf
    target_distance = float(np.linalg.norm(target))
    for slot in range(CLOT_SLOTS):
        start = CLOT0 + 6 * slot
        if obs_row[start] <= 0:
            continue
        # Slot distance and bearing are the only quantities used.  The target
        # position is supplied by the registered sequential baseline only.
        observed_distance = float(obs_row[start + 1] * 10.0)
        score = abs(observed_distance - target_distance) / max(target_distance, 1e-6)
        if score < error:
            best, error = slot, score
    return best


class MultiClusterController:
    """Fair local controller with a cluster-level spacing protocol."""

    def __init__(self, env, config: MultiClusterConfig):
        if env.num_robots != config.clusters:
            raise ValueError("the experiment defines one controlled entity per cluster")
        self.env = env
        self.config = config
        self.cluster_ids = balanced_cluster_ids(env.num_robots, config.clusters)
        self.observer = PartialObserver(
            env, PartialObsConfig(noise=config.observation_noise), seed=0
        )
        self.step_index = 0
        self.wait_steps = np.zeros(config.clusters, dtype=np.int32)
        self.deadlock_events = 0
        self.yield_events = 0
        self.spacing_violations = 0
        self.min_spacing_seen_mm = np.inf
        self._right_of_way = 0
        self._planned_target_order = None
        self._previous_edges = None
        self._previous_previous_edges = None
        self._edge_flip_count = np.zeros(config.clusters, dtype=np.int32)
        self._edge_same_count = np.zeros(config.clusters, dtype=np.int32)

    def reset(self, seed: int):
        self.observer.reset(seed)
        self.step_index = 0
        self.wait_steps.fill(0)
        self.deadlock_events = 0
        self.yield_events = 0
        self.spacing_violations = 0
        self.min_spacing_seen_mm = np.inf
        self._right_of_way = 0
        self._previous_edges = self.env.edges[:self.env.num_robots].copy()
        self._previous_previous_edges = self._previous_edges.copy()
        self._edge_flip_count.fill(0)
        self._edge_same_count.fill(0)
        # The baseline may use a fixed pre-operative order.  This is never
        # consulted by the parallel controller.
        if self.config.method == "single_sequential":
            self._planned_target_order = np.argsort(
                self.env.tree.arclength[self.env.clot_stations]
            ).astype(np.int32)
        else:
            self._planned_target_order = None

    def _sequential_slots(self, obs: np.ndarray) -> np.ndarray:
        live = np.flatnonzero(self.env.masses > 0)
        slots = np.full(self.env.num_robots, -1, dtype=np.int32)
        if not len(live):
            return slots
        current = next((int(j) for j in self._planned_target_order if self.env.masses[j] > 0), int(live[0]))
        target = self.env.clot_positions_mm[current]
        for i in range(self.env.num_robots):
            slots[i] = _target_slot_from_global_position(
                obs[i], target - self.env.positions_mm[i]
            )
        return slots

    def _spacing_shield(self, world: np.ndarray, centres: np.ndarray) -> np.ndarray:
        """Apply right-of-way stopping from current and predicted spacing."""
        commands = world.copy()
        if self.config.clusters <= 1 or self.config.min_spacing_mm <= 0:
            return commands
        flow = self.env.transport.velocity_mm_s(
            self.env.positions_mm[:self.env.num_robots],
            self.env.edges[:self.env.num_robots], self.env.solution,
        )
        predicted = centres + self.config.wait_horizon_s * (
            flow + world * self.env.config.robot_speed_mm_s
        )
        blocked = np.zeros(self.config.clusters, dtype=bool)
        for a in range(self.config.clusters):
            for b in range(a + 1, self.config.clusters):
                if not np.isfinite(centres[[a, b]]).all():
                    continue
                now = float(np.linalg.norm(centres[a] - centres[b]))
                future = float(np.linalg.norm(predicted[a] - predicted[b]))
                self.min_spacing_seen_mm = min(self.min_spacing_seen_mm, now, future)
                violating = min(now, future) < self.config.min_spacing_mm
                if violating:
                    # Preserve one right-of-way command.  If the bodies are
                    # already inside the exclusion radius, stopping both can
                    # never restore separation in a flow field; the loser
                    # yields while the winner is allowed to leave the region.
                    winner = self._right_of_way if self._right_of_way in (a, b) else a
                    loser = b if winner == a else a
                    blocked[loser] = True
                    self.wait_steps[loser] += 1
                    self.yield_events += 1
                    if self.wait_steps[loser] >= self.config.deadlock_window_steps:
                        self.deadlock_events += 1
                        self._right_of_way = loser
                        self.wait_steps[[a, b]] = 0
                        blocked[winner] = True
                        blocked[loser] = False
                else:
                    self.wait_steps[[a, b]] = 0
        for cluster in np.flatnonzero(blocked):
            commands[self.cluster_ids == cluster] = 0.
        # Count violating control steps, rather than counting every step after
        # the historical minimum has crossed the threshold.
        if any(
            np.isfinite(centres[a]).all() and np.isfinite(centres[b]).all()
            and np.linalg.norm(centres[a] - centres[b]) < self.config.min_spacing_mm
            for a in range(self.config.clusters)
            for b in range(a + 1, self.config.clusters)
        ):
            self.spacing_violations += 1
        return commands

    def act(self) -> tuple[np.ndarray, dict]:
        current_edges = self.env.edges[:self.env.num_robots].copy()
        if self._previous_edges is None:
            self._previous_edges = current_edges.copy()
            self._previous_previous_edges = current_edges.copy()
        changed = current_edges != self._previous_edges
        self._edge_same_count[changed] = 0
        self._edge_same_count[~changed] += 1
        # A local path controller can select opposite sides of a junction on
        # successive frames.  Holding that entity for one control interval
        # lets passive flow move it out of the junction and avoids feeding an
        # ill-conditioned sequence of edge crossings to the integrator.
        returned = changed & (current_edges == self._previous_previous_edges)
        self._edge_flip_count[returned] += 1
        self._edge_flip_count[changed & ~returned] = 0
        self._edge_flip_count[~changed] = 0
        obs = self.observer.observe()
        if self.config.method == "single_sequential":
            # Method 1 is the traditional pre-operative baseline: one field
            # follows a fixed clot order with the known route and wait rule.
            live = np.flatnonzero(self.env.masses > 0)
            if len(live):
                target = next((int(j) for j in self._planned_target_order
                               if self.env.masses[j] > 0), int(live[0]))
                self.env._assignment = np.full(self.env.num_robots, target, dtype=np.int64)
                self.env._assignment_alive = live.tobytes()
            local, _stop, _subgoal = teacher_label(self.env, self.env.config)
        else:
            # Slot 0 is the nearest visible clot in each local observation;
            # no global route, assignment or future target is read here.
            local = fair_reactive_action(obs, mode="path")
        world = execute_local(self.env, local)
        world[(self._edge_flip_count >= 2) | (self._edge_same_count >= 25)] = 0.
        centres = _cluster_centroids(self.env, self.cluster_ids)
        world = self._spacing_shield(world, centres)
        self.observer.record_action(local)
        self._previous_previous_edges = self._previous_edges
        self._previous_edges = current_edges
        self.step_index += 1
        return world, {
            "clusters": int(self.config.clusters),
            "method": self.config.method,
            "yield_events": int(self.yield_events),
            "deadlock_events": int(self.deadlock_events),
            "spacing_violations": int(self.spacing_violations),
            "locally_held_for_stall": int(np.sum(self._edge_same_count >= 25)),
            "min_spacing_mm": None if not np.isfinite(self.min_spacing_seen_mm) else float(self.min_spacing_seen_mm),
        }

    def summary(self) -> dict:
        return {
            "method": self.config.method,
            "clusters": int(self.config.clusters),
            "min_spacing_threshold_mm": float(self.config.min_spacing_mm),
            "yield_events": int(self.yield_events),
            "deadlock_events": int(self.deadlock_events),
            "spacing_violation_steps": int(self.spacing_violations),
            "minimum_spacing_mm": None if not np.isfinite(self.min_spacing_seen_mm) else float(self.min_spacing_seen_mm),
        }
