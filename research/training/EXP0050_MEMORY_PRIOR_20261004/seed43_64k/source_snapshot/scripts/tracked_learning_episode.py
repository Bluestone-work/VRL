"""EXP0048 episode uses the same measured-frame pipeline for every policy."""
from dataclasses import replace
import hashlib
import json
import numpy as np

from environments.mca_physical_env import DynamicsConfig
from marl.multicluster import MultiClusterConfig
from marl.local_maneuver_learning import ManeuverLibrary
from marl.tracked_sensors import TrackedSensorAdapter, TrackingSpec
from scripts.local_learning_episode import LearningEpisode, ROOT, learning_hashes
from scripts.evaluate_multicluster import DEFAULT_CONFIG
from scripts.multicluster_protocol import paired_environment, SpacingTracker, attach_spacing_monitor
from scripts.safe_metrics import WallTracker

PROTOCOL = ROOT/'configs/experiments/EXP_0048_TRACKED_LEARNING.json'


def tracking_hashes():
    hashes = learning_hashes()
    for file in ('marl/tracked_sensors.py', 'scripts/tracked_learning_episode.py',
                 'scripts/run_tracked_learning.py', 'configs/experiments/EXP_0048_TRACKED_LEARNING.json'):
        hashes[file] = hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    return hashes


class TrackingManeuverLibrary(ManeuverLibrary):
    def prepare(self, packet):
        features, candidates, valid, details = super().prepare(packet)
        # Binary visual clot state provides no mass-progress estimate. Do not
        # mistake a legitimate near-target treatment dwell for navigation stall.
        near_target = (packet.navigation[:, 11] > 0) & (packet.navigation[:, 12]*10 < .3)
        details['stuck'][near_target] = False
        features[near_target, 127] = 0.
        return features, candidates, valid, details


class TrackedLearningEpisode(LearningEpisode):
    def __init__(self, scene_seed, *, control_seed=42, clusters=3, duration=180., reward_config=None):
        base = replace(DynamicsConfig.from_json(DEFAULT_CONFIG), num_robots=clusters,
                       episode_duration_s=duration)
        self.cfg = MultiClusterConfig(method='multi_parallel' if clusters > 1 else 'single_sequential',
                                     clusters=clusters, min_spacing_mm=2., control_seed=control_seed)
        self.env, self.manifest = paired_environment(base, clusters, scene_seed, budget='fixed_total')
        settings = json.loads(PROTOCOL.read_text())['sensor_settings']
        self.sensor = TrackedSensorAdapter(self.env, self.cfg, TrackingSpec(**settings))
        self.sensor.reset(int(np.random.SeedSequence([scene_seed, control_seed]).generate_state(1)[0]))
        self.library = TrackingManeuverLibrary(self.cfg, speed=base.robot_speed_mm_s, dt=base.control_dt_s)
        self.walls = WallTracker(clusters)
        self.spacing = SpacingTracker(self.env.positions_mm[:clusters], self.env.active[:clusters], 2.)
        attach_spacing_monitor(self.env, self.spacing)
        self.initial_mass = float(self.env.initial_mass.sum())
        self.previous_removal = self.auc = self.pair_contact = self.commands = 0.
        self.milestones = {str(k): None for k in (50, 90, 100)}
        self.reward_config = reward_config or json.loads(PROTOCOL.read_text())['reward']
        self.scene_seed, self.control_seed = scene_seed, control_seed
        self.steps = 0; self.done = False
        self.choices = np.zeros(10, np.int64)
        self.prepare()

    def step(self, choices, *, legacy=False):
        # Ground truth is instrumentation only: missing image tracks must not
        # shrink the wall-contact denominator. Navigation still uses packet.active.
        physical_active = self.env.active[:self.cfg.clusters].copy()
        observed_active = self.packet.active.copy()
        reward, native_done, active, lost = super().step(choices, legacy=legacy)
        dt = float(self.info['step_duration_s'])
        self.walls.active_robot_s += float(physical_active.sum()-observed_active.sum())*dt
        # Simulator all-clear is an outcome label, not an oracle stop signal.
        # Keep collecting delayed images and simulating motion until the same
        # measurement processor used by every policy confirms completion.
        if native_done:
            self.prepare()
        confirmed = bool(not self.sensor.processor.alive.any())
        horizon = self.env.elapsed_s >= self.env.config.episode_duration_s-1e-12
        all_exited = not self.env.active[:self.cfg.clusters].any()
        self.done = bool(confirmed or horizon or all_exited)
        self.env._done = self.done
        self.info['termination_reason'] = (
            'observed_all_clots_cleared' if confirmed else
            'all_robots_exited' if all_exited else 'time_limit' if horizon else None)
        return reward, self.done, active, lost

    def result(self, policy):
        from dataclasses import asdict
        result = super().result(policy)
        result.update(experiment='EXP_0048_TRACKED_LEARNING', source_hashes=tracking_hashes(),
            observation_contract='tracked_local_v1', sensor_assumptions=asdict(self.sensor.spec),
            policy_frame='fixed_calibrated_camera_actuator_axes',
            exact_mass_input=False, true_velocity_input=False, true_frenet_input=False,
            geometry_reconstruction_hardware_verified=False,
            sensor_confirmed_all_cleared=bool(not self.sensor.processor.alive.any()),
            completion_uses_observed_confirmation=True,
            false_visual_completion=bool(not self.sensor.processor.alive.any() and not self.info['success']))
        result['physical_cluster_safe_success'] = result['cluster_safe_success']
        result['cluster_safe_success'] = bool(result['cluster_safe_success'] and result['sensor_confirmed_all_cleared'])
        return result
