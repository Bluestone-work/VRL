"""EXP0051 measured options over frozen physical and measurement interfaces."""
import hashlib
import json
import numpy as np

from scripts.tracked_learning_episode import TrackedLearningEpisode, tracking_hashes, ROOT
from marl.measured_options import (option_observation, latch_targets, target_packet,
    joint_measured_projection, measured_assignment, priority_options, N_OPTIONS)
from marl.fair_reactive import fair_reactive_action

PROTOCOL = ROOT/'configs/experiments/EXP_0051_MEASURED_OPTIONS.json'


def option_hashes():
    hashes = tracking_hashes()
    for name in ('marl/measured_options.py', 'scripts/option_learning_episode.py',
                 'scripts/run_option_learning.py', 'scripts/run_option_study.py',
                 'configs/experiments/EXP_0051_MEASURED_OPTIONS.json'):
        hashes[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return hashes


class OptionEpisode(TrackedLearningEpisode):
    def __init__(self, scene_seed, *, clusters=3, option_steps=10,
                 supervisor='joint', coupling=0., duration=180., control_seed=42):
        self.protocol = json.loads(PROTOCOL.read_text())
        self.previous_options = np.zeros(clusters, np.int64)
        self.target_ids = np.full(clusters, -1, np.int32)
        self.option_steps = option_steps
        self.supervisor = supervisor
        self.coupling = float(coupling)
        self.option_counts = np.zeros(N_OPTIONS, np.int64)
        self.macro_steps = self.supervisor_changes = self.supervisor_infeasible = 0
        self.missing_track_steps = 0
        self.max_supervisor_residual = 0.
        if option_steps < 1 or coupling < 0 or coupling > 1:
            raise ValueError('Invalid option duration or synthetic coupling')
        super().__init__(scene_seed, clusters=clusters, control_seed=control_seed, duration=duration)
        from scripts.multicluster_protocol import digest
        self.initial_snapshot = dict(positions_mm=self.env.positions_mm.tolist(),
            masses=self.env.masses.tolist(), active=self.env.active.tolist(),
            edges=self.env.edges.tolist(), clot_positions_mm=self.env.clot_positions_mm.tolist(),
            initial_mass=self.env.initial_mass.tolist(),
            driving_pressure=float(self.env.flow_model.driving_pressure),
            flow_multiplier=float(self.env.episode_flow_multiplier))
        self.initial_snapshot_hash = digest(self.initial_snapshot)
        self._sensor_execute = self.sensor.execute
        if coupling:
            self.sensor.execute = self._synthetic_actuator_execute

    def _synthetic_actuator_execute(self, commands):
        # Physics-only stress. The actor knows requested commands, not the true
        # spatial response matrix. This is NOT a calibrated magnetic model.
        requested = self._sensor_execute(commands)
        n = self.cfg.clusters
        points = self.env.positions_mm[:n]
        distance = np.linalg.norm(points[:, None]-points[None], axis=-1)
        length = self.protocol['actuator_stress']['length_scale_mm']
        response = self.coupling*np.exp(-distance/length)
        np.fill_diagonal(response, 1.)
        output = response@requested
        return output/np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1.)

    def prepare(self):
        self.packet = self.sensor.observe()
        self.high_features, self.high_valid = option_observation(self.packet, self.previous_options, self.target_ids)

    def conventional_options(self, policy):
        if policy in ('memory', 'legacy_memory', 'untrained'):
            return np.zeros(self.cfg.clusters, np.int64)
        if policy == 'balanced':
            return measured_assignment(self.packet, self.target_ids)
        if policy == 'priority':
            return priority_options(self.packet, self.target_ids)
        raise ValueError('Unknown scheduler')

    def step_option(self, options, *, maximum_steps=None):
        options = np.asarray(options, np.int64)
        n = self.cfg.clusters
        if options.shape != (n,) or not np.all(self.high_valid[np.arange(n), options]):
            raise ValueError('Invalid measured option')
        self.previous_options = options.copy()
        self.target_ids = latch_targets(self.packet, options)
        self.option_counts += np.bincount(options[self.packet.active], minlength=N_OPTIONS)
        reward, duration = 0., 0
        limit = min(self.option_steps, maximum_steps or self.option_steps)
        for k in range(limit):
            self.packet = target_packet(self.packet, self.target_ids)
            self.features, self.candidates, self.valid, self.details = self.library.prepare(self.packet)
            choice = self.library.conventional_choice(self.packet, self.valid, self.details, memory=True)
            command = self.candidates[np.arange(n), choice].copy()
            held = options == 5
            command[held] = self.packet.navigation[held, :3]-self.packet.navigation[held, 3:6]
            retreat = options == 6
            if retreat.any():
                command[retreat] = -fair_reactive_action(self.packet.navigation, mode='path')[retreat]
            command /= np.maximum(np.linalg.norm(command, axis=1, keepdims=True), 1.)
            if self.supervisor == 'joint':
                command, checks = joint_measured_projection(command, self.packet,
                    self.protocol['supervisor'], self.library.speed)
                self.supervisor_changes += int(checks['changed'])
                self.supervisor_infeasible += int(checks['residual'] > 1e-5)
                self.max_supervisor_residual = max(self.max_supervisor_residual, checks['residual'])
                self.missing_track_steps += checks['missing_tracks']
                self.details['projection_residual'][np.arange(n), choice] = np.maximum(
                    self.details['projection_residual'][np.arange(n), choice], checks['residual'])
            elif self.supervisor != 'legacy':
                raise ValueError('Unknown supervisor')
            self.candidates[np.arange(n), choice] = command
            removal_before = self.previous_removal
            spacing_before = self.spacing.pair_violation_s
            physical_before = self.env.active[:n].copy()  # reward instrumentation only
            _, done, _, _ = super().step(choice)
            dt = float(self.info['step_duration_s'])
            horizon = self.env.config.episode_duration_s
            rc = self.protocol['reward']
            step_reward = rc['removed_fraction']*(self.previous_removal-removal_before)
            step_reward += rc['removal_auc_fraction']*.5*(self.previous_removal+removal_before)*dt/horizon
            step_reward += rc['wall_fraction']*float(np.sum(self.info['wall_contact_s']))/(n*horizon)
            step_reward += rc['spacing_pair_fraction']*(self.spacing.pair_violation_s-spacing_before)/(max(n*(n-1)/2, 1)*horizon)
            step_reward += rc['particle_fraction']*float(np.sum(self.info['particle_contact_s']))/(n*horizon)
            step_reward += rc['lost_fraction']*float(np.sum(physical_before & ~self.env.active[:n]))/n
            step_reward += rc['step_cost_per_horizon']*dt/horizon
            reward += self.protocol['gamma_per_control_step']**k*step_reward
            duration += 1
            if done:
                break
            # Only a still-tracked robot can trigger a measured target event.
            # Missing detections are not mislabeled as verified target clearance.
            target_event = any(self.target_ids[i] >= 0 and self.packet.active[i]
                and self.target_ids[i] not in self.packet.clot_ids[i] for i in range(n))
            if target_event:
                break
        self.macro_steps += 1
        return float(reward), self.done, duration

    def result(self, policy):
        result = super().result(policy)
        result.update(experiment='EXP_0051_MEASURED_OPTIONS', source_hashes=option_hashes(),
            scheduler=policy, option_steps=self.option_steps, macro_steps=self.macro_steps,
            physical_control_steps=self.steps,
            mean_option_control_steps=self.steps/max(self.macro_steps, 1),
            option_counts=self.option_counts.tolist(), supervisor=self.supervisor,
            memory_option_fraction=float(self.option_counts[0]/max(self.option_counts.sum(), 1)),
            supervisor_changed_steps=self.supervisor_changes,
            supervisor_infeasible_steps=self.supervisor_infeasible,
            supervisor_max_residual=self.max_supervisor_residual,
            missing_track_agent_steps=self.missing_track_steps,
            supervisor_settings=self.protocol['supervisor'],
            actuator_coupling=self.coupling, actuator_hardware_calibrated=False,
            actuator_stress_model=self.protocol['actuator_stress'],
            actual_initial_snapshot=self.initial_snapshot,
            actual_initial_snapshot_hash=self.initial_snapshot_hash,
            learning_information='measured local packets shared centrally; no simulator actor inputs',
            safety_certificate=False)
        return result
