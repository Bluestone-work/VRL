"""EXP_0021 isolated protocol: tail-repair via no-progress early truncation.

Base (all validated, stacked): variable-N padding (8 slots / 5 active,
max_agents=10) + dynamic particles 16 in training + GRU six-frame motion
prediction (arm 17 observation stack) + greedy 5-step scheduler with the
v2 shared fixes. Single registered variable vs the matched base arm:
no-progress early truncation — a row that goes K=50 consecutive steps
with zero total mass removal and is not otherwise done is truncated
(truncated, NOT terminated: the learner bootstraps value, matching the
horizon semantics) and reset immediately, converting dead time into new
episodes.

Historical EXP16-20 runs and files are untouched; this module is new code.
"""
from __future__ import annotations

import time

import numpy as np

from environments.research_predictive_env import (
    ResearchBalanced, ResearchSingle, ResearchVector, replan,
)


STALL_LIMIT = 50  # K: consecutive steps with zero removed mass -> truncate


class TailTruncationVector(ResearchVector):
    """ResearchVector + per-row no-progress stall truncation.

    The stall counter counts steps since the last positive total clot-mass
    decrease. A success (terminated) or horizon (truncated) row never has
    its semantics altered. When the counter reaches K the row is truncated
    and auto-reset exactly like a horizon cut: `final_observation` /
    `final_context` are provided so PPO bootstraps the terminal value.
    """

    def __init__(self, *args, stall_limit: int = STALL_LIMIT, **kwargs):
        # Base __init__ calls reset_all -> _reset_envs; set counters first.
        self.stall_limit = int(stall_limit)
        self._stall = None
        self._prev_total_mass = None
        self.stall_truncated = None
        super().__init__(*args, **kwargs)
        self._stall = np.zeros(self.n_envs, np.int64)
        self._prev_total_mass = self.clot_masses.sum(axis=1).copy()
        self.stall_truncated = np.zeros(self.n_envs, bool)

    def _reset_envs(self, idx):
        super()._reset_envs(idx)
        if self._stall is not None:
            self._stall[idx] = 0
            self._prev_total_mass[idx] = self.clot_masses[idx].sum(axis=1)
            self.stall_truncated[idx] = False

    def step(self, action):
        before_pos = self.robot_positions.copy()
        previous_length = self.robot_path_length.copy()
        obs, reward, terminated, truncated, info = super().step(action)
        done = terminated | truncated
        if 'final_context' in info:
            before_pos[done] = info['final_context']['positions'][done]

        total = self.clot_masses.sum(axis=1)
        # Rows still mid-episode: a strictly positive decrease resets the
        # counter; zero removal advances it. Rows that just finished keep
        # their counters until `_reset_envs` clears them (harmless: they
        # reset on the same call that restarts the episode).
        live = ~done
        decreased = total < self._prev_total_mass - 1e-12
        self._stall[live & decreased] = 0
        self._stall[live & ~decreased] += 1
        self._prev_total_mass[live] = total[live]

        stalled = live & (self._stall >= self.stall_limit)
        if not np.any(stalled):
            info['stall_truncated'] = self.stall_truncated & done
            info['stall_steps'] = self._stall.copy()
            return obs, reward, terminated, truncated, info

        # Cut semantics mirror the horizon cut in the parent step(): the
        # learner bootstraps from the final observation, so this is a
        # truncation, never a termination.
        truncated = truncated | stalled
        done = terminated | truncated
        info['final_observation'] = obs.copy()
        final_ctx_positions = before_pos.copy()
        final_ctx_positions[~done] = self.robot_positions[~done]
        info['final_context'] = {
            'positions': final_ctx_positions,
            'velocities': self.robot_velocities.copy(),
        }
        self.stall_truncated[stalled] = True
        info['stall_truncated'] = self.stall_truncated & done
        info['stall_steps'] = self._stall.copy()
        self._reset_envs(np.flatnonzero(done))
        return self._observe(), reward, terminated, truncated, info


class TailTruncationSingle(ResearchSingle):
    """Single-env twin used for deterministic evaluation.

    Evaluation uses the SAME truncation rule as training: the preregistered
    comparison is "base vs base+early-truncation" under one protocol, and
    reporting both raw horizon outcomes and truncation counts keeps the
    change auditable. Success requires clearing every clot within the
    horizon, exactly as before; the cut only removes dead time.
    """

    def __init__(self, *args, stall_limit: int = STALL_LIMIT, **kwargs):
        super().__init__(*args, **kwargs)
        self.stall_limit = int(stall_limit)
        self._stall = 0
        self._prev_total_mass = float(self.clot_masses.sum())
        self.stall_truncated = False

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._stall = 0
        self._prev_total_mass = float(self.clot_masses.sum())
        self.stall_truncated = False
        return obs, info

    def step(self, action):
        before, previous_length = self.robot_positions.copy(), self.robot_path_length.copy()
        obs, reward, terminated, truncated, info = super().step(action)
        total = float(self.clot_masses.sum())
        if not (terminated or truncated):
            if total < self._prev_total_mass - 1e-12:
                self._stall = 0
            else:
                self._stall += 1
            self._prev_total_mass = total
            if self._stall >= self.stall_limit:
                truncated = True
                self.stall_truncated = True
        info['stall_truncated'] = bool(self.stall_truncated)
        info['stall_steps'] = int(self._stall)
        return obs, reward, terminated, truncated, info


def make_training_env(n_envs, seed, predictor, predictor_device,
                      num_robots=8, active_robots=5, stall_limit=STALL_LIMIT):
    """Balanced 14-scenario training stack with the EXP_0021 base config."""
    envs = []
    from environments.vessel_geometry import resolve_pool
    scenarios = tuple(resolve_pool('anatomical'))
    counts = np.full(len(scenarios), n_envs // len(scenarios))
    counts[:n_envs % len(counts)] += 1
    for i, (scenario, count) in enumerate(zip(scenarios, counts)):
        envs.append(TailTruncationVector(
            n_envs=int(count), scenario=scenario, scenario_pool=[scenario],
            randomize_scenario=False, seed=seed + 1009 * i,
            particle_seed=seed + 300001 + 1009 * i,
            predictor=predictor, predictor_device=predictor_device,
            num_robots=num_robots, active_robots=active_robots, num_clots=3,
            horizon=300, robot_radius=.0011, obs_mode='geometric_predictive',
            contact_mode='geodesic', dynamic_intravascular_particles=True,
            particle_count=16, particle_radius_ratio=1.6,
            particle_lateral_drift=.15, tree_resample_interval=900,
            stall_limit=stall_limit))
    return _BalancedShell(envs, counts)


class _BalancedShell(ResearchBalanced):
    """ResearchBalanced constructed from prebuilt TailTruncation children.

    Subclassing keeps the parent's aggregation (scenario grouping, info
    merging, particles bookkeeping) while replacing the child class; the
    stall bookkeeping keys are merged alongside the particle keys.
    """

    def __init__(self, envs, counts):
        self.envs = list(envs)
        from environments.vessel_geometry import resolve_pool
        self.scenarios = tuple(resolve_pool('anatomical'))
        self.n_envs = sum(int(c) for c in counts)
        self.num_robots = self.envs[0].num_robots
        self.num_clots = self.envs[0].num_clots
        self.horizon = self.envs[0].horizon
        self.group_sizes = tuple(map(int, counts))
        self.scenario_ids = np.repeat(
            np.arange(len(counts), dtype=np.int16), counts)
        self.scenario_names = np.asarray(
            [self.scenarios[i] for i in self.scenario_ids])

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        for key in ('stall_truncated', 'stall_steps'):
            info[key] = np.concatenate(
                [np.asarray(child.last_info[key]) for child in self.envs])
        return obs, reward, term, trunc, info


def make_eval_env(scenario, episode_seed, particle_seed, predictor,
                  predictor_device, stall_limit=STALL_LIMIT):
    """Deterministic evaluation env: 5 real robots (no padding at eval, as in
    the sealed-protocol evaluations) with the same truncation rule."""
    return TailTruncationSingle(
        scenario=scenario, scenario_pool=[scenario],
        randomize_scenario=False, seed=episode_seed, num_robots=5,
        num_clots=3, horizon=300, robot_radius=.0011,
        obs_mode='geometric_predictive', contact_mode='geodesic',
        dynamic_intravascular_particles=True, particle_count=24,
        particle_seed=particle_seed, predictor=predictor,
        predictor_device=predictor_device, stall_limit=stall_limit)
