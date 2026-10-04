"""Shared EXP0047 environment, metrics and observation-only execution."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np

from environments.mca_physical_env import DynamicsConfig
from marl.multicluster import MultiClusterConfig
from marl.multicluster_observation import ClusterSensorAdapter
from marl.local_maneuver_learning import ManeuverLibrary
from scripts.evaluate_multicluster import DEFAULT_CONFIG, source_hashes
from scripts.multicluster_protocol import paired_environment, SpacingTracker, attach_spacing_monitor, digest
from scripts.safe_metrics import WallTracker, episode_metrics

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'configs/experiments/EXP_0047_LOCAL_LEARNING.json'


def learning_hashes():
    result = source_hashes()
    for file in ('marl/local_maneuver_learning.py', 'scripts/local_learning_episode.py',
                 'scripts/run_local_learning.py', 'configs/experiments/EXP_0047_LOCAL_LEARNING.json'):
        result[file] = hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    return result


class LearningEpisode:
    def __init__(self, scene_seed, *, control_seed=42, clusters=3, duration=180., reward_config=None):
        base = replace(DynamicsConfig.from_json(DEFAULT_CONFIG), num_robots=clusters,
                       episode_duration_s=duration)
        self.cfg = MultiClusterConfig(method='multi_parallel' if clusters > 1 else 'single_sequential',
                                     clusters=clusters, min_spacing_mm=2., control_seed=control_seed)
        self.env, self.manifest = paired_environment(base, clusters, scene_seed, budget='fixed_total')
        self.sensor = ClusterSensorAdapter(self.env, self.cfg)
        self.sensor.reset(int(np.random.SeedSequence([scene_seed, control_seed]).generate_state(1)[0]))
        self.library = ManeuverLibrary(self.cfg, speed=base.robot_speed_mm_s, dt=base.control_dt_s)
        self.walls = WallTracker(clusters)
        self.spacing = SpacingTracker(self.env.positions_mm[:clusters], self.env.active[:clusters], 2.)
        attach_spacing_monitor(self.env, self.spacing)
        self.initial_mass = float(self.env.initial_mass.sum())
        self.previous_removal = self.auc = self.pair_contact = self.commands = 0.
        self.milestones = {str(k): None for k in (50, 90, 100)}
        self.reward_config = reward_config or json.loads(PROTOCOL.read_text())['reward']
        self.scene_seed, self.control_seed = scene_seed, control_seed
        self.steps = 0
        self.done = False
        self.choices = np.zeros(10, np.int64)
        self.prepare()

    def prepare(self):
        self.packet = self.sensor.observe()
        self.features, self.candidates, self.valid, self.details = self.library.prepare(self.packet)

    def step(self, choices, *, legacy=False):
        n = self.cfg.clusters
        active = self.packet.active.copy()
        choices = np.asarray(choices, np.int64)
        assert choices.shape == (n,) and np.all(self.valid[np.arange(n), choices])
        local = self.details['legacy'] if legacy else self.candidates[np.arange(n), choices]
        self.library.commit(choices, self.details)
        self.choices += np.bincount(choices[active], minlength=10)
        self.spacing.begin_step()
        before_spacing = self.spacing.pair_violation_s
        world = self.sensor.execute(local)
        _, _, terminal, truncated, info = self.env.step(world)
        self.spacing.end_step()
        dt = float(info['step_duration_s'])
        self.walls.update(info, active, dt)
        self.commands += float(np.square(world).sum())*dt
        self.pair_contact += float(info['robot_pair_contact_s'])
        removal = 1-float(info['remaining_mass'])/self.initial_mass
        self.auc += .5*(self.previous_removal+removal)*dt
        self.previous_removal = removal
        for k in self.milestones:
            if self.milestones[k] is None and removal >= int(k)/100-1e-12:
                self.milestones[k] = float(info['elapsed_s'])
        rc = self.reward_config
        # Team removal/spacing outcome and own contact costs. Labels are used
        # only for training reward; no simulator metric enters actor features.
        reward = np.full(n, rc['removed_mass']*float(info['removed_mass'])/n
            +rc['spacing_pair_s']*(self.spacing.pair_violation_s-before_spacing)/n
            +rc['step_cost'], dtype=float)
        reward += rc['wall_contact_s']*np.asarray(info['wall_contact_s'])
        reward += rc['particle_contact_s']*np.asarray(info['particle_contact_s']).reshape(n, -1).sum(axis=1)
        lost = active & ~self.env.active[:n]
        reward += rc['lost_cluster']*lost
        reward[~active] = 0.
        self.info = info
        self.done = bool(terminal or truncated)
        self.steps += 1
        if not self.done:
            self.prepare()
        return reward.astype(np.float32), self.done, active, lost

    def result(self, policy):
        assert self.done
        metrics = episode_metrics(self.info, self.walls, self.initial_mass)
        spacing = self.spacing.summary()
        safe = bool(metrics['safe_collision_free'] and spacing['spacing_compliant'] and self.pair_contact <= 1e-12)
        elapsed = float(self.info['elapsed_s'])
        return dict(experiment='EXP_0047_LOCAL_LEARNING', scene_seed=self.scene_seed,
            control_seed=self.control_seed, clusters=self.cfg.clusters, method=policy,
            status='completed', **metrics, **spacing, cluster_safe_success=safe,
            removal_auc_s=self.auc, removal_auc_180=(self.auc+(self.env.config.episode_duration_s-elapsed)*metrics['removal'])/self.env.config.episode_duration_s,
            time_to_removal_s=self.milestones, particle_contact_s=float(self.info['episode_particle_contact_s']),
            robot_pair_contact_s=self.pair_contact, lost_clusters=int(self.info['lost_robots']),
            active_cluster_s=self.spacing.active_cluster_s, command_squared_s=self.commands,
            path_mm=float(np.sum(self.info['robot_path_mm'])), maneuver_counts=self.choices.tolist(),
            projection_infeasible_agent_steps=self.library.infeasible_agent_steps,
            scenario_hash=self.manifest['scenario_hash'], reset_info=self.manifest,
            final_state_hash=digest(dict(positions=self.env.positions_mm.tolist(), masses=self.env.masses.tolist(),
                active=self.env.active.tolist(), elapsed_s=self.env.elapsed_s)),
            source_hashes=learning_hashes(), sealed_test_used=False, safety_certificate=False)

    def close(self):
        self.env.close()
