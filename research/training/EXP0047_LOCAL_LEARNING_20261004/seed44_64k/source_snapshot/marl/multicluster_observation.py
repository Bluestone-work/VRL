"""Simulation sensor adapter; explicitly separate from observation-only policies.

Assumptions: pre-operative clot positions/IDs; imaging estimates of local lumen,
ego frame, active tracked bodies, and nearby peers' velocities. Exact simulator
geometry remains a sensor proxy, not a demonstrated hardware reconstruction.
"""
from __future__ import annotations
import numpy as np
from marl.partial_obs import PartialObserver, PartialObsConfig, CLOT_SLOTS, CLOT0
from marl.multicluster import ClusterObservation


class ClusterSensorAdapter:
    def __init__(self, env, config):
        self.env, self.config = env, config
        self.base = PartialObserver(env, PartialObsConfig(noise=config.observation_noise,
                                                         position_noise_mm=0.))
        self.reset(0)

    def reset(self, seed):
        self.base.reset(seed)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 461]))
        self.previous_world = np.zeros((self.env.num_robots, 3))

    def frames(self):
        e = self.env
        return np.stack((e.tree.tangents[e.robot_stations], e.tree.normals[e.robot_stations],
                         e.tree.binormals[e.robot_stations]), axis=1).astype(float)

    def observe(self):
        env, c = self.env, self.config
        n = env.num_robots
        frames = self.frames()
        nav = self.base.observe()
        nav[:, :3] = np.einsum('nij,nj->ni', frames, self.previous_world)
        positions = env.positions_mm[:n] + self.rng.normal(0., c.position_noise_mm, (n, 3))
        velocities = env.velocity_mm_s[:n] * self.rng.normal(1., c.observation_noise, (n, 3))
        nav[:, 3:6] = np.einsum('nij,nj->ni', frames, velocities)/env.config.robot_speed_mm_s
        rel = positions[None, :, :] - positions[:, None, :]
        relv = velocities[None, :, :] - velocities[:, None, :]
        visible = np.linalg.norm(rel, axis=2) <= c.peer_sensing_radius_mm
        visible &= env.active[:n, None] & env.active[None, :n]
        np.fill_diagonal(visible, False)
        local_rel = np.einsum('iab,ijb->ija', frames, rel)
        local_relv = np.einsum('iab,ijb->ija', frames, relv)
        local_rel[~visible] = 0.; local_relv[~visible] = 0.
        clot_ids = np.full((n, CLOT_SLOTS), -1, np.int32)
        # Rebuild goal slots from the same noisy tracking positions, retaining
        # only Euclidean target vectors and observable completion status.
        nav[:, CLOT0:CLOT0+6*CLOT_SLOTS] = 0.
        for i in np.flatnonzero(env.active[:n]):
            targets = env.clot_positions_mm - positions[i]
            distances = np.linalg.norm(targets, axis=1)
            order = np.argsort(np.where(env.masses > 0, distances, np.inf))[:CLOT_SLOTS]
            for k, j in enumerate(order):
                if env.masses[j] <= 0:
                    continue
                s = CLOT0+6*k
                clot_ids[i, k] = j
                nav[i, s] = 1.
                nav[i, s+1] = max(0., distances[j]*self.rng.normal(1., c.observation_noise))/10.
                nav[i, s+2:s+5] = frames[i]@targets[j]/max(distances[j], 1e-9)
                nav[i, s+5] = env.masses[j]/env.initial_mass[j]
        return ClusterObservation(nav, clot_ids, local_rel, local_relv, visible, env.active[:n].copy())

    def execute(self, local):
        frames = self.frames()
        world = np.einsum('nji,nj->ni', frames, local)*self.env.active[:self.env.num_robots, None]
        self.previous_world = world.copy()
        self.base.record_action(local)
        return world
