"""Observation-only cluster controllers for EXP0046 revision 2.

The policy receives arrays from an explicit sensor adapter. It has no environment
handle, route table, true flow field, physical edge id, or ground-truth safety
metric. The velocity filter is a heuristic under measurement/dynamics error;
its predictions are never reported as measured spacing or a safety certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from marl.fair_reactive import fair_reactive_action
from marl.partial_obs import CLOT0, CLOT_SLOTS, PartialObsConfig


@dataclass(frozen=True)
class MultiClusterConfig:
    method: str = 'multi_parallel'
    clusters: int = 2
    min_spacing_mm: float = 2.
    deadlock_window_steps: int = 10
    wait_horizon_s: float = .5
    observation_noise: float = .025
    peer_sensing_radius_mm: float = 6.
    position_noise_mm: float = .02
    spacing_buffer_mm: float = .1
    control_seed: int = 42

    def __post_init__(self):
        if self.method not in ('single_sequential', 'single_route', 'multi_parallel', 'multi_unshielded'):
            raise ValueError('Unknown method')
        for name in ('clusters', 'deadlock_window_steps'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.method.startswith('single_') and self.clusters != 1:
            raise ValueError('Single-cluster baselines require clusters=1')
        if self.method.startswith('multi_') and self.clusters < 2:
            raise ValueError('Parallel methods require clusters>=2')
        for name in ('min_spacing_mm', 'observation_noise', 'position_noise_mm', 'spacing_buffer_mm'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('wait_horizon_s', 'peer_sensing_radius_mm'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.peer_sensing_radius_mm <= self.min_spacing_mm + self.spacing_buffer_mm:
            raise ValueError('Peer sensing must extend beyond the exclusion radius and buffer')


@dataclass(frozen=True)
class ClusterObservation:
    navigation: np.ndarray       # legacy fair 111-vector, in each ego frame
    clot_ids: np.ndarray         # tracked clot identities aligned with the slots
    peer_relative_mm: np.ndarray # [ego, peer, xyz], ego frame, zero outside sensor range
    peer_relative_velocity_mm_s: np.ndarray
    peer_visible: np.ndarray
    active: np.ndarray


def balanced_cluster_ids(num_entities: int, clusters: int) -> np.ndarray:
    for v in (num_entities, clusters):
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError('Counts must be positive integers')
    if clusters > num_entities:
        raise ValueError('clusters cannot exceed the number of controlled entities')
    result = np.empty(num_entities, np.int32)
    for cluster, indices in enumerate(np.array_split(np.arange(num_entities), clusters)):
        result[indices] = cluster
    return result


class MultiClusterController:
    """Stateless sensor interface plus per-cluster target/yield memory."""

    def __init__(self, config: MultiClusterConfig, *, robot_speed_mm_s=1., control_dt_s=.1):
        self.config = config
        self.speed = float(robot_speed_mm_s)
        self.dt = float(control_dt_s)
        if min(self.speed, self.dt) <= 0 or not np.isfinite([self.speed, self.dt]).all():
            raise ValueError('Invalid actuator/control time calibration')
        if config.method == 'single_route':
            raise ValueError('The privileged route baseline is separate from the local policy')
        if config.method == 'multi_parallel' and config.peer_sensing_radius_mm < (
            config.min_spacing_mm + config.spacing_buffer_mm + 2*self.speed*config.wait_horizon_s
        ):
            raise ValueError('Peer sensing insufficient for the configured command lookahead')
        self.reset(config.control_seed)

    def reset(self, seed=None):
        n = self.config.clusters
        self.step_index = 0
        self.current_target = np.full(n, -1, np.int32)
        self.previous_yielding = np.zeros(n, bool)
        self.wait_steps = np.zeros(n, np.int32)
        self.yield_events = self.yield_agent_steps = self.persistent_yield_events = 0
        self.predicted_conflict_agent_steps = 0
        self.ranks = np.random.default_rng(self.config.control_seed if seed is None else seed).permutation(n)

    def _sequential_slots(self, observation):
        slots = np.zeros(self.config.clusters, np.int32)
        for i, ids in enumerate(observation.clot_ids):
            live = [k for k in range(CLOT_SLOTS) if observation.navigation[i, CLOT0+6*k] > 0]
            previous = [k for k in live if ids[k] == self.current_target[i]]
            slots[i] = previous[0] if previous else (live[0] if live else -1)
            self.current_target[i] = ids[slots[i]] if slots[i] >= 0 else -1
        return slots

    def _spacing_filter(self, preferred, observation):
        """Project commands onto local linear clearance constraints.

        Velocity is measured from image tracking. Last actuator command is
        re-expressed in the current ego frame by the adapter. Neighbours are
        assumed to retain measured velocity over the lookahead; this assumption
        is approximate, so realized separation is independently measured.
        """
        out = preferred.copy()
        yielding = np.zeros(self.config.clusters, bool)
        for i in np.flatnonzero(observation.active):
            constraints = []
            old = observation.navigation[i, :3]
            for j in np.flatnonzero(observation.peer_visible[i]):
                rel = observation.peer_relative_mm[i, j]
                relv = observation.peer_relative_velocity_mm_s[i, j]
                distance = float(np.linalg.norm(rel))
                # IDs only break the exactly-coincident tracking degeneracy.
                normal = rel/max(distance, 1e-12) if distance > 1e-9 else np.array([1. if j > i else -1., 0., 0.])
                gap = distance - self.config.min_spacing_mm - self.config.spacing_buffer_mm
                bound = float(normal@old + (normal@relv + .5*gap/self.config.wait_horizon_s)/self.speed)
                constraints.append((normal, bound))
                closing = relv + self.speed*(old-preferred[i])
                t = float(np.clip(-rel@closing/max(closing@closing, 1e-12), 0, self.config.wait_horizon_s))
                risk = np.linalg.norm(rel+t*closing) < self.config.min_spacing_mm+self.config.spacing_buffer_mm
                if risk:
                    self.predicted_conflict_agent_steps += 1
                    if self.ranks[i] > self.ranks[j]:
                        yielding[i] = True
            if yielding[i]:
                out[i] = 0.
            for _ in range(8):
                for normal, bound in constraints:
                    out[i] -= max(float(normal@out[i])-bound, 0.)*normal
                out[i] /= max(float(np.linalg.norm(out[i])), 1.)
        return out, yielding

    def act(self, observation: ClusterObservation):
        n = self.config.clusters
        if observation.navigation.shape != (n, 111) or not np.isfinite(observation.navigation).all():
            raise ValueError('Expected finite [clusters,111] navigation observations')
        slots = self._sequential_slots(observation) if self.config.method == 'single_sequential' else None
        local = fair_reactive_action(observation.navigation, mode='path',
                                     cfg=PartialObsConfig(), target_slots=slots)
        yielding = np.zeros(n, bool)
        if self.config.method == 'multi_parallel' and self.config.min_spacing_mm > 0:
            local, yielding = self._spacing_filter(local, observation)
        local[~observation.active] = 0.
        self.yield_events += int(np.sum(yielding & ~self.previous_yielding))
        self.yield_agent_steps += int(yielding.sum())
        # This is a detectable prolonged-yield proxy, not a proof of deadlock.
        stalled = yielding & (np.linalg.norm(observation.navigation[:, 3:6], axis=1)*self.speed < .05)
        self.wait_steps = np.where(stalled, self.wait_steps+1, 0)
        prolonged = self.wait_steps == self.config.deadlock_window_steps
        self.persistent_yield_events += int(prolonged.sum())
        if np.any(prolonged):
            self.ranks = (self.ranks+1) % n
        self.previous_yielding = yielding
        self.step_index += 1
        return local, self.summary()

    def summary(self):
        return dict(method=self.config.method, clusters=self.config.clusters,
                    yield_events=self.yield_events, yield_agent_steps=self.yield_agent_steps,
                    persistent_yield_events=self.persistent_yield_events,
                    predicted_conflict_agent_steps=self.predicted_conflict_agent_steps,
                    safety_certificate=False, controller_input='measured_observation_only')
