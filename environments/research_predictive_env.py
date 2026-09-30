"""Isolated v2 protocol for EXP16--20. Historical running jobs are unchanged."""
from __future__ import annotations

import time
import numpy as np

from environments.vector_env import VectorVascularEnv
from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from environments.balanced_vector_env import BalancedVectorVascularEnv
from environments.vessel_geometry import resolve_pool
from marl.predictive_research import Features


class ResearchVector(VectorVascularEnv):
    def __init__(self, *args, predictor=None, predictor_device='cpu', **kwargs):
        self.features = Features(predictor, predictor_device)
        super().__init__(*args, **kwargs)

    def _reset_envs(self, idx):
        super()._reset_envs(idx)
        self.features.reset(idx)
        if self.particles is not None:
            # Independent fresh episode obstacles; preserve unfinished rows.
            old_p, old_v = self.particles.positions.copy(), self.particles.velocities.copy()
            self.particles.reset(self.tree)
            keep = np.ones(self.n_envs, bool)
            keep[idx] = False
            self.particles.positions[keep] = old_p[keep]
            self.particles.velocities[keep] = old_v[keep]

    def _observe(self):
        return self.features.augment(self, super()._observe())

    def step(self, action):
        before = self.robot_positions.copy()
        previous_length = self.robot_path_length.copy()
        obs, reward, term, trunc, info = super().step(action)
        done = term | trunc
        after = self.robot_positions.copy()
        if 'final_context' in info:
            after[done] = info['final_context']['positions'][done]
        length = previous_length + np.linalg.norm(after-before, axis=-1)*self.agent_mask
        info['robot_path_length'], info['path_length'] = length, length.sum(-1)
        self.robot_path_length[~done] = length[~done]
        self.path_length[~done] = length.sum(-1)[~done]
        diff = after[:, :, None]-after[:, None]
        overlap = np.linalg.norm(diff, axis=-1) < self.collision_distance
        info['robot_collisions'] = np.triu(overlap, 1).sum((1, 2))
        self.last_info = info
        return obs, reward, term, trunc, info


class ResearchSingle(Vascular3DMARLEnv):
    def __init__(self, *args, predictor=None, predictor_device='cpu', **kwargs):
        self.features = Features(predictor, predictor_device)
        super().__init__(*args, **kwargs)

    def reset(self, **kwargs):
        self.features.reset()
        self.robot_path_length = np.zeros(self.num_robots, np.float64)
        return super().reset(**kwargs)

    def _build_observation(self):
        return self.features.augment(self, super()._build_observation(), vector=False)

    def step(self, action):
        before, previous_length = self.robot_positions.copy(), self.robot_path_length.copy()
        obs, reward, term, trunc, info = super().step(action)
        self.robot_path_length = previous_length+np.linalg.norm(self.robot_positions-before, axis=-1)
        self.path_length = float(self.robot_path_length.sum())
        info.update(path_length=self.path_length, robot_path_length=self.robot_path_length.copy())
        return obs, reward, term, trunc, info


class ResearchBalanced(BalancedVectorVascularEnv):
    def __init__(self, n_envs=56, seed=42, predictor=None, predictor_device='cpu',
                 scenario_pool='anatomical', **kwargs):
        self.scenarios = tuple(resolve_pool(scenario_pool))
        if n_envs < len(self.scenarios):
            raise ValueError('Need at least one env per scenario')
        self.n_envs, self.num_robots = n_envs, int(kwargs.get('num_robots', 5))
        self.num_clots, self.horizon = int(kwargs.get('num_clots', 3)), int(kwargs.get('horizon', 300))
        counts = np.full(len(self.scenarios), n_envs//len(self.scenarios))
        counts[:n_envs % len(counts)] += 1
        self.group_sizes = tuple(map(int, counts))
        self.envs = [ResearchVector(
            n_envs=int(count), scenario=scenario, scenario_pool=[scenario],
            randomize_scenario=False, seed=seed+1009*i,
            particle_seed=seed+300001+1009*i, predictor=predictor,
            predictor_device=predictor_device, **kwargs)
            for i, (scenario, count) in enumerate(zip(self.scenarios, counts))]
        self.scenario_ids = np.repeat(np.arange(len(counts), dtype=np.int16), counts)
        self.scenario_names = np.asarray([self.scenarios[i] for i in self.scenario_ids])

    def step(self, action):
        # Parent aggregation omits obstacle events. Preserve these per child
        # through a cached info record without changing the historical parent.
        obs, reward, term, trunc, info = super().step(action)
        for key in ('particle_collisions', 'particle_clearance', 'robot_collisions'):
            info[key] = np.concatenate([child.last_info[key] for child in self.envs])
        return obs, reward, term, trunc, info


def assignments(env, row=None):
    """Greedy workload/ETA allocation; identical across v2 policy arms.

    Known map, current mass and current local flow. No particle-risk term.
    Coordinates and route indices remain distinct. No simulator prediction.
    """
    masses = env.clot_masses if row is None else env.clot_masses[row]
    stations = env.robot_stations if row is None else env.robot_stations[row]
    positions = env.robot_positions if row is None else env.robot_positions[row]
    clot_stations = env.clot_stations if row is None else env.clot_stations[row]
    prev = env.task_assignments
    if prev is not None and row is not None:
        prev = prev[row]
    active = env.num_robots if row is None else env.active_robots
    live = np.flatnonzero(masses > 0)
    result = np.full(env.num_robots, -1, np.int32)
    if not len(live):
        return result
    distance, eta = [], []
    occluded = env._occluded_radius(env.robot_stations)
    flow = env.tree.flow(positions, stations, env.flow_speed, env.tube_radius,
                         radius_override=occluded if row is None else occluded[row])
    for c in live:
        dist, hop = (env._route(int(c)) if row is None else env._route(int(clot_stations[c])))
        direction = env.tree.points[hop[stations]]-env.tree.points[stations]
        direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
        distance.append(dist[stations])
        speed = np.clip(env.max_speed+np.sum(flow*direction, -1), env.max_speed*.2, env.max_speed*2)
        eta.append(dist[stations]/speed)
    distance, eta = np.stack(distance, -1), np.stack(eta, -1)
    loads = np.zeros(len(live))
    for r in range(active):
        service = masses[live]/(env.lysis_rate*np.minimum(loads+1, env.lysis_saturation))
        # Greedy marginal workload improvement plus travel and commitment.
        marginal = -service/(loads+1)
        cost = eta[r]+marginal+2*loads
        if prev is not None and prev[r] in live:
            cost += 15*(live != prev[r])
        choice = int(cost.argmin())
        result[r] = live[choice]
        loads[choice] += 1
    return result


def replan(env):
    start = time.perf_counter()
    switches = 0
    if hasattr(env, 'n_envs'):
        for child in (env.envs if hasattr(env, 'envs') else [env]):
            old = child.task_assignments
            new = np.stack([assignments(child, i) if old is None or
                child.steps[i] % 5 == 0 or np.any(child.clot_masses[i, np.maximum(old[i], 0)] <= 0)
                else old[i].copy() for i in range(child.n_envs)])
            if old is not None:
                switches += int(((old >= 0) & (new != old)).sum())
            child.set_task_assignments(new)
        obs = env.observe() if hasattr(env, 'envs') else env._observe()
    else:
        old = env.task_assignments
        new = assignments(env) if old is None or env.steps % 5 == 0 or np.any(
            env.clot_masses[np.maximum(old, 0)] <= 0) else old.copy()
        if env.task_assignments is not None:
            switches = int(((env.task_assignments >= 0) & (new != env.task_assignments)).sum())
        env.set_task_assignments(new)
        obs = env._build_observation()
    return obs, {'switches': switches, 'planner_seconds': time.perf_counter()-start}
