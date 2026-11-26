"""Parameter-shared IPPO for local blood-flow navigation.

The physical simulator still advances all clusters synchronously.  A thin
VecEnv adapter exposes one logical environment per cluster, so SB3 PPO learns
one shared local actor/critic and emits one 3-D command for each cluster.  The
critic is local (IPPO), while rewards are the deployable per-cluster
navigation decomposition returned by ``CurriculumVascularEnv``.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.policies import ActorCriticPolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv
from torch import nn


# The action is a complete deployable local command, rather than a residual
# around a hand-written action.  These slots deliberately cover the strongest
# classical behaviours (route pursuit and near-target settling) plus the small
# set of lateral manoeuvres needed for moving-obstacle avoidance.
SEMANTIC_CANDIDATE_COUNT = 18
ATOMIC_CANDIDATE_COUNT = 15
# Keep actions that were selected by at least one trained policy, plus the
# low-amplitude lateral primitives needed for dynamic-obstacle avoidance.
SEMANTIC_REDUCED_INDICES = (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
SEMANTIC_REDUCED_COUNT = len(SEMANTIC_REDUCED_INDICES)
# Measured over three converged seeds (action_stats.csv): six of the thirteen
# reduced slots were selected 0.00% of the time -- both lateral pairs, reverse
# and target_full -- despite being valid 76.6% of steps. Five slots cover 98.9%
# of selections. This set keeps the seven that are ever used, so the categorical
# head is not spending capacity on actions the policy never takes. Enlarging the
# set was already tested and lost 9.2pp (K18), so the direction is to shrink.
SEMANTIC_MINIMAL_INDICES = (0, 1, 2, 4, 6, 7, 8)
SEMANTIC_MINIMAL_COUNT = len(SEMANTIC_MINIMAL_INDICES)


def _unit(v, fallback=None):
    v = np.asarray(v, np.float32)
    n = float(np.linalg.norm(v))
    if n > 1e-8:
        return v / n
    return np.zeros_like(v) if fallback is None else np.asarray(fallback, np.float32)


def atomic_candidates(physical):
    """Strict primitive local velocity pulses.

    The first local axis is the route-frame tangent.  Its sign is selected
    from the measured route command, while the other two axes are the local
    normal/binormal.  Each action is one pulse; obstacle avoidance therefore
    comes from a sequence of small lateral pulses, not a macro target action.
    """
    est, targets, hold, frames = physical.current
    ep = physical.ep
    rule = np.asarray(getattr(physical, "current_rule", np.zeros((physical.clusters, 3))), np.float32)
    n = physical.clusters
    commands = np.zeros((n, ATOMIC_CANDIDATE_COUNT, 3), np.float32)
    valid = np.ones((n, ATOMIC_CANDIDATE_COUNT), np.float32)
    for i in range(n):
        tangent_sign = 1. if float(rule[i, 0]) >= 0. else -1.
        fwd = np.array([tangent_sign, 0., 0.], np.float32)
        back = -fwd
        commands[i] = np.asarray([
            [0., 0., 0.],
            .35*fwd, .70*fwd, fwd,
            .35*back, .70*back, back,
            [0., .35, 0.], [0., -.35, 0.],
            [0., .70, 0.], [0., -.70, 0.],
            [0., 0., .35], [0., 0., -.35],
            [0., 0., .70], [0., 0., -.70],
        ], np.float32)
        if (not bool(est.active[i])) or bool(hold[i]) or int(targets[i]) < 0:
            commands[i, 1:] = 0.
        # No duplicate slots are retained in the primitive set; inactive
        # clusters intentionally expose only the zero command.
        seen = {}
        for k, c in enumerate(commands[i]):
            key = tuple(np.round(c, 6))
            if key in seen:
                valid[i, k] = 0.
            else:
                seen[key] = k
    return commands, valid


def direct_candidates(physical, action_set="semantic"):
    """Build deployable direct commands from the current measured/map state.

    The route and clot geometry are already part of the permitted task context;
    the actor only receives this candidate matrix and selects its index.  The
    candidate generator never reads simulator flow, response gain, or true
    future position.
    """
    if action_set == "atomic":
        return atomic_candidates(physical)
    if action_set == "semantic_reduced":
        indices = SEMANTIC_REDUCED_INDICES
    elif action_set == "semantic_minimal":
        indices = SEMANTIC_MINIMAL_INDICES
    elif action_set == "semantic":
        indices = range(SEMANTIC_CANDIDATE_COUNT)
    else:
        raise ValueError(f"unknown action_set: {action_set}")
    est, targets, hold, frames = physical.current
    ep = physical.ep
    rule = np.asarray(getattr(physical, "current_rule", np.zeros((physical.clusters, 3))), np.float32)
    n = physical.clusters
    indices = np.asarray(indices, dtype=np.intp)
    commands = np.zeros((n, len(indices), 3), np.float32)
    valid = np.ones((n, len(indices)), np.float32)
    for i in range(n):
        r = np.clip(rule[i], -1., 1.)
        nominal = np.clip(np.asarray(ep.ctl.nominal[i], np.float32), -1., 1.)
        route = _unit(nominal, _unit(r))
        target = np.zeros(3, np.float32)
        t = int(targets[i])
        if 0 <= t < len(ep.env.clot_positions_mm):
            target = _unit(frames[i] @ (ep.env.clot_positions_mm[t] - est.pos[i]))
        # Fixed/adaptive settle family: direct approach at full, moderate and
        # distance-scaled speed.  The distance-scaled slot is the analogue of
        # the best classical Settle controller, while the actor can release it
        # by selecting a full approach command when flow stalls.
        d = float(np.linalg.norm(ep.env.clot_positions_mm[t] - est.pos[i])) if t >= 0 else np.inf
        settle = target * float(np.clip(d / .3, 0., 1.))
        # [zero, classical route command, route at three speeds, direct target
        # at three speeds, settle, reverse, +/- normal and +/- binormal]
        tangent = route
        normal = np.array([0., 1., 0.], np.float32)
        binormal = np.array([0., 0., 1.], np.float32)
        commands[i] = np.asarray([
            np.zeros(3), r, tangent, .75*tangent, .5*tangent,
            target, .5*target, .25*target, settle, -tangent,
            .35*normal, -.35*normal, .35*binormal, -.35*binormal,
            .7*normal, -.7*normal, .7*binormal, -.7*binormal,
        ], np.float32)[indices]
        # If no action is executable, keep zero as the only meaningful slot;
        # safety/TPG still owns the final hold decision.
        if (not bool(est.active[i])) or bool(hold[i]) or t < 0:
            commands[i, 1:] = 0.
        commands[i] = np.clip(commands[i], -1., 1.)
        # Duplicate commands are merged into the first slot by marking later
        # slots invalid in the diagnostic mask.  The fixed categorical policy
        # still has a valid no-op fallback for inactive agents.
        seen = {}
        for k, c in enumerate(commands[i]):
            key = tuple(np.round(c, 6))
            if key in seen:
                valid[i, k] = 0.
            else:
                seen[key] = k
    return commands, valid

from scripts.train_sb3_curriculum import CurriculumVascularEnv
from scripts.navigation_dataset import MANIFEST, iter_rows


class IPPOAgentVecEnv(VecEnv):
    """Expose the synchronized physical episode as ``clusters`` local envs."""

    def __init__(self, physical_env: CurriculumVascularEnv, action_set="semantic",
                 failure_log="integration_failures.json", flow_feature=False,
                 safety_parity=False, settle_radius=.3, exit_mask=False,
                 centralized_critic=False, critic_identity=False, record_trace=False):
        self.physical = physical_env
        if action_set not in ("semantic", "semantic_reduced", "semantic_minimal", "atomic", "continuous"):
            raise ValueError("action_set must be semantic, semantic_reduced, semantic_minimal, atomic, or continuous")
        self.action_set = action_set
        self.integration_failures = 0
        self._failure_log = failure_log
        # Module B of the WAND-style design: expose an online flow estimate to the
        # policy. ObservedMotion identifies v = gain*command + drift from the
        # timestamp-aligned command log and image velocities, so it is deployable
        # and costs one least-squares fit per step rather than a frozen ensemble
        # forward pass. It also reports a confidence, which matters because the
        # measured direction accuracy of a flow estimate is only ~0.71 cosine --
        # the policy should be able to discount a low-confidence estimate.
        self.flow_feature = bool(flow_feature)
        self._om = None
        # Safety parity (see --safety-parity). The classical SettleGuard is
        # WallGuard + an unconditional d/radius scaling inside the reach radius;
        # the RL path had neither, so an action chosen near the target could carry
        # the cluster across the boundary node and deactivate it. Applying the
        # same guards after the policy output is what makes the two comparable.
        self.safety_parity = bool(safety_parity)
        self.settle_radius = float(settle_radius)
        self.exit_mask = bool(exit_mask)
        self.centralized_critic = bool(centralized_critic)
        self.critic_identity = bool(critic_identity)
        self.exit_masked_total = 0
        self._wall_guard = None
        # Continuous actors still receive the compact atomic basis as context,
        # but their output is a free 3-D local velocity rather than an index.
        self.candidate_count = (ATOMIC_CANDIDATE_COUNT
                                if action_set in ("atomic", "continuous")
                                else SEMANTIC_CANDIDATE_COUNT)
        if action_set == "semantic_reduced":
            self.candidate_count = SEMANTIC_REDUCED_COUNT
        elif action_set == "semantic_minimal":
            self.candidate_count = SEMANTIC_MINIMAL_COUNT
        self.n_agents = physical_env.clusters
        n = self.n_agents
        self._pending_actions = None
        self._obs = None
        self._candidate_commands = None
        self._candidate_valid = None
        self._last_reset_info = None
        self.trace = []
        self.record_trace = record_trace
        physical_env.record_execution = record_trace
        observation_space = spaces.Dict({
            "hist": spaces.Box(-np.inf, np.inf, (24, 11), np.float32),
            "hmask": spaces.Box(0., 1., (24,), np.float32),
            "ctx": spaces.Box(-np.inf, np.inf, (32,), np.float32),
            "image": spaces.Box(-.3, 2., (4, 32, 32), np.float16),
            "obstacles": spaces.Box(-10., 10., (8, 25), np.float32),
            "obstacle_valid": spaces.Box(0., 1., (8,), np.float32),
            "peers": spaces.Box(-10., 10., (n, 8), np.float32),
            "peer_valid": spaces.Box(0., 1., (n,), np.float32),
            "candidates": spaces.Box(-1., 1., (self.candidate_count, 3), np.float32),
            "candidate_valid": spaces.Box(0., 1., (self.candidate_count,), np.float32),
            **({"flow": spaces.Box(-5., 5., (6,), np.float32)} if self.flow_feature else {}),
            **({"central_state": spaces.Box(-10., 10., (self._central_state_width() + (self.n_agents if self.critic_identity else 0),), np.float32)}
               if getattr(self, "centralized_critic", False) else {}),
        })
        action_space = (spaces.Box(-1., 1., (3,), np.float32)
                        if action_set == "continuous" else
                        spaces.Discrete(self.candidate_count))
        super().__init__(n, observation_space, action_space)
        self.reset_infos = [{} for _ in range(n)]

    def _central_state_width(self):
        return self.n_agents * (11 + 32 + self.candidate_count * 4 + self.n_agents * 8 + 8 * 25)

    def _split(self, obs):
        rows = []
        for i in range(self.n_agents):
            row = {}
            for k, v in obs.items():
                array = np.asarray(v)
                row[k] = array.copy() if k == "central_state" and array.ndim == 1 else np.asarray(array[i]).copy()
            rows.append(row)
        return rows

    def _stack(self, rows):
        return {k: np.stack([r[k] for r in rows], axis=0) for k in rows[0]}

    def _with_candidates(self, obs):
        feature_set = "atomic" if self.action_set == "continuous" else self.action_set
        commands, valid = direct_candidates(self.physical, feature_set)
        self._candidate_commands = commands
        self._candidate_valid = valid
        enriched = dict(obs)
        if self.flow_feature or self.exit_mask:
            flow = self._flow_features()
            if self.flow_feature:
                enriched["flow"] = flow
        if self.exit_mask:
            from marl.exit_mask import apply_exit_mask
            est = self.physical.current[0]
            valid, n_masked = apply_exit_mask(self.physical.ep, est, commands, valid, drift=self._om.drift)
            self.exit_masked_total += n_masked
            self._candidate_valid = valid
        enriched["candidates"] = commands
        enriched["candidate_valid"] = valid
        if getattr(self, "centralized_critic", False):
            rows = []
            for i in range(self.n_agents):
                rows.append(np.concatenate([
                    np.asarray(obs["hist"][i, -1], np.float32),
                    np.asarray(obs["ctx"][i], np.float32),
                    commands[i].reshape(-1), valid[i],
                    np.asarray(obs["peers"][i], np.float32).reshape(-1),
                    np.asarray(obs["obstacles"][i], np.float32).reshape(-1),
                ]))
            joint = np.concatenate(rows).astype(np.float32)
            if self.critic_identity:
                identity = np.eye(self.n_agents, dtype=np.float32)
                enriched["central_state"] = np.concatenate(
                    [np.concatenate([joint, identity[i]]) for i in range(self.n_agents)]
                ).reshape(self.n_agents, -1)
            else:
                enriched["central_state"] = joint
        return enriched

    def _apply_guards(self, joint):
        """SettleGuard's two mechanisms, applied to the policy's chosen command.

        Order follows the classical controller: WallGuard first (it may rewrite the
        command for wall recovery or after an event re-plan), then the unconditional
        settle scaling inside the reach radius. Both act on the local-frame command,
        ahead of the TPG hold and spacing shield that LysisEpisode.step applies.
        """
        n = self.n_agents
        ep = getattr(self.physical, 'ep', None)
        if ep is None:
            return joint
        cmd = np.asarray(joint, float).reshape(n, 3)
        from scripts.benchmark_lysis import SettleGuard
        est, targets, hold, _frames = self.physical.current
        if self._wall_guard is None:
            self._wall_guard = SettleGuard(ep, radius=self.settle_radius)
        cmd = self._wall_guard(ep, est, targets, cmd, hold)
        return cmd.reshape(-1)

    def _flow_features(self):
        """[n, 6] = drift in the local route frame (3), gain, fit rmse, confidence.

        The drift is rotated into the same local frame the candidate commands use,
        so a policy can compare "where the flow pushes me" against "where this
        candidate sends me" without an implicit frame conversion.
        """
        n = self.n_agents
        out = np.zeros((n, 6), np.float32)
        ep = getattr(self.physical, 'ep', None)
        if ep is None:
            return out
        if self._om is None:
            from marl.observed_motion import ObservedMotion
            self._om = ObservedMotion(n, dt=float(ep.env.config.control_dt_s))
        est, _targets, _hold, frames = self.physical.current
        self._om.update(ep.env.elapsed_s, ep.sent_world, est.frame_time_s,
                        est.pos[:n], est.active[:n], est.edge[:n])
        F = np.asarray(frames, float).reshape(n, 3, 3)
        drift_local = np.einsum('nab,nb->na', F, np.asarray(self._om.drift, float)[:n])
        out[:, 0:3] = np.clip(drift_local, -5., 5.)
        out[:, 3] = np.clip(np.asarray(self._om.gain, float)[:n], 0., 5.)
        out[:, 4] = np.clip(np.asarray(self._om.rmse, float)[:n], 0., 5.)
        out[:, 5] = np.clip(np.asarray(self._om.confidence, float)[:n], 0., 1.)
        return out

    def seed(self, seed=None):
        # All logical agents share one physical simulator and one scene RNG.
        shared = super().seed(seed)[0]
        self._seeds = [shared] * self.n_agents
        return self._seeds

    def set_options(self, options=None):
        if isinstance(options, list):
            if len(options) != self.n_agents or any(opt != options[0] for opt in options[1:]):
                raise ValueError("all agents in the physical episode must share reset options")
        super().set_options(options)

    def reset(self, options=None):
        self._pending_actions = None
        self._om = None
        self._wall_guard = None
        self.trace = []
        obs, info = self.physical.reset(seed=self._seeds[0], options=self._options[0] if options is None else options)
        self._reset_seeds()
        self._reset_options()
        obs = self._with_candidates(obs)
        self._obs = obs
        self._last_reset_info = info
        self.reset_infos = [dict(info) for _ in range(self.n_agents)]
        return self._stack(self._split(obs))

    def step_async(self, actions):
        if self.action_set == "continuous":
            a = np.asarray(actions, dtype=np.float32)
            if a.shape != (self.n_agents, 3) or not np.isfinite(a).all():
                raise ValueError(f"expected {(self.n_agents, 3)} finite continuous actions, got {a.shape}")
            a = np.clip(a, -1., 1.)
            # Unit-ball projection preserves low-speed commands and bounds
            # the physical local velocity, including diagonal actions.
            a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1.)
            self._pending_actions = a.copy()
        else:
            a = np.asarray(actions, dtype=np.int64).reshape(-1)
            if a.shape != (self.n_agents,):
                raise ValueError(f"expected {(self.n_agents,)} candidate indices, got {a.shape}")
            self._pending_actions = np.clip(a, 0, self.candidate_count - 1)

    def step_wait(self):
        if self._pending_actions is None:
            raise RuntimeError("step_async must be called before step_wait")
        if self._candidate_commands is None:
            raise RuntimeError("candidate cache is not initialized")
        if self.action_set == "continuous":
            joint = self._pending_actions.reshape(-1)
            chosen = None
        else:
            chosen = self._pending_actions.copy()
        # Categorical SB3 has no built-in action-mask distribution.  When a
        # duplicated/inactive slot is sampled, canonicalise it to the first
        # equivalent valid command before execution; the actor still sees the
        # validity mask as part of its observation.
        if chosen is not None:
            for i in range(self.n_agents):
                k = int(chosen[i])
                if self._candidate_valid[i, k] < .5:
                    same = np.all(np.isclose(self._candidate_commands[i],
                                             self._candidate_commands[i, k], atol=1e-6), axis=1)
                    valid_same = np.flatnonzero(same & (self._candidate_valid[i] > .5))
                    if len(valid_same):
                        chosen[i] = valid_same[0]
                    else:
                        chosen[i] = np.flatnonzero(self._candidate_valid[i] > .5)[0]
            joint = self._candidate_commands[np.arange(self.n_agents), chosen].reshape(-1)
        if self.record_trace:
            requested_local = joint.reshape(self.n_agents, 3).copy()
        if self.safety_parity:
            joint = self._apply_guards(joint)
        if self.exit_mask and self.action_set == "continuous":
            from marl.exit_mask import compensate_exit_command
            est = self.physical.current[0]
            joint = compensate_exit_command(
                self.physical.ep, est, joint.reshape(self.n_agents, 3), drift=self._om.drift
            ).reshape(-1)
        try:
            obs, shared_reward, terminated, truncated, info = self.physical.step(joint)
        except RuntimeError as exc:
            # The adaptive integrator refuses to cap flow silently and raises when
            # a step would need more substeps than its budget. That is one unusable
            # scene, not a reason to lose the whole run: end the episode as a
            # failure, count it, and let the rollout continue. The count is written
            # to the run directory so a quietly degraded run stays visible.
            if 'integration budget' not in str(exc):
                raise
            self.integration_failures += 1
            failure_log = Path(self._failure_log)
            failure_log.parent.mkdir(parents=True, exist_ok=True)
            failure_log.write_text(json.dumps(dict(integration_failures=self.integration_failures,
                                                   last_error=str(exc)[:200])) + '\n')
            obs = self._obs
            shared_reward, terminated, truncated = 0., True, False
            physical = self.physical
            ep = physical.ep
            # The physical integrator rolled back the rejected interval. Log the
            # completed prefix once so the success-rate denominator includes the failure.
            metrics = dict(physical.scene, integration_failure=True, termination_reason="integration_budget",
                           termination="integration_budget", collision_free_success=False,
                           navigation_success=False, strict_success=False, task_success=False,
                           navigation_reached=int(physical.nav_reached.sum()), lost=int(physical.nav_lost.sum()),
                           elapsed_s=float(ep.env.elapsed_s), environment_steps=ep.steps,
                           wall_contact_s=float(ep.walls.total.sum()),
                           wall_after_reach_s=float(physical.wall_after_reach.sum()),
                           particle_collision_events=int(ep.env.episode_particle_collision_events),
                           particle_contact_s=float(ep.env.episode_particle_contact_s),
                           navigation_return=physical.nav_return, return_sum=physical.return_sum,
                           agent_returns=physical.agent_returns.tolist(),
                           mean_agent_return=float(physical.agent_returns.mean()))
            physical._record_navigation_episode(metrics)
            info = dict(integration_failure=True, navigation_success=False, is_success=False,
                        navigation_reached=metrics["navigation_reached"], episode_metrics=metrics)
        if self.record_trace and not info.get("integration_failure"):
            step = info.pop("executed_step")
            step.update(requested_local_velocity=requested_local.tolist(),
                        action_indices=None if chosen is None else self._pending_actions.tolist(),
                        executed_candidate_indices=None if chosen is None else chosen.tolist(),
                        candidate_commands_local=self._candidate_commands.tolist(),
                        navigation_reached=int(info["navigation_reached"]),
                        navigation_success=bool(info["navigation_success"]))
            self.trace.append(step)
        done = bool(terminated or truncated)
        rewards = np.asarray(info.get("agent_rewards", np.full(self.n_agents, shared_reward)), dtype=np.float32).reshape(-1)
        if rewards.size == 1:
            rewards = np.full(self.n_agents, float(rewards[0]), np.float32)
        if rewards.size != self.n_agents:
            rewards = np.full(self.n_agents, float(shared_reward), np.float32)
        infos = [dict(info) for _ in range(self.n_agents)]
        # A rejected integration step has no new sensor frame or executed command.
        if not info.get("integration_failure"):
            obs = self._with_candidates(obs)
        next_rows = self._split(obs)
        if done:
            # Preserve the terminal local observation for SB3 diagnostics and
            # immediately reset the one physical episode for the next rollout.
            for i in range(self.n_agents):
                infos[i]["terminal_observation"] = next_rows[i]
                infos[i]["TimeLimit.truncated"] = bool(truncated)
            if self.exit_mask:
                infos[0]["exit_masked_total"] = self.exit_masked_total
            if self.record_trace:
                for agent_info in infos:
                    agent_info["execution_trace"] = self.trace
                self.trace = []
            self._om = None          # one estimator per episode; excitation history must not carry over
            self._wall_guard = None  # WallGuard carries per-episode replanner state
            reset_obs, reset_info = self.physical.reset()
            reset_obs = self._with_candidates(reset_obs)
            self._obs = reset_obs
            self.reset_infos = [dict(reset_info) for _ in range(self.n_agents)]
            next_rows = self._split(reset_obs)
        else:
            self._obs = obs
        self._pending_actions = None
        return self._stack(next_rows), rewards, np.full(self.n_agents, done, dtype=bool), infos

    def close(self):
        self.physical.close()

    def get_attr(self, attr_name, indices=None):
        value = getattr(self.physical, attr_name)
        ids = range(self.n_agents) if indices is None else np.atleast_1d(indices)
        return [value for _ in ids]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.physical, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        result = getattr(self.physical, method_name)(*method_args, **method_kwargs)
        ids = range(self.n_agents) if indices is None else np.atleast_1d(indices)
        return [result for _ in ids]

    def env_is_wrapped(self, wrapper_class, indices=None):
        ids = range(self.n_agents) if indices is None else np.atleast_1d(indices)
        return [False for _ in ids]

    def get_images(self):
        return None

    def render(self, mode=None):
        return self.physical.render() if hasattr(self.physical, "render") else None


class IPPOExtractor(BaseFeaturesExtractor):
    """Compact local encoder with masked obstacle and peer pooling."""

    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(16, 24, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(96, 32), nn.Tanh())
        self.gru = nn.GRU(11, 32, batch_first=True)
        self.ctx = nn.Sequential(nn.Linear(32, 32), nn.LayerNorm(32), nn.GELU())
        self.obstacle = nn.Sequential(nn.Linear(25, 32), nn.GELU())
        self.peer = nn.Sequential(nn.Linear(8, 32), nn.GELU())
        candidate_count = int(observation_space["candidates"].shape[0])
        self.candidate = nn.Sequential(nn.Linear(candidate_count * 3, 32), nn.GELU())
        # Flow adapter (module B). Residual fusion into the trunk output with a
        # zero-initialised last layer, so an untrained flow branch contributes
        # exactly nothing at step 0 and the arm starts from the no-flow policy.
        # Same trick the predictive actor already uses on its own output head.
        self.flow_adapter = None
        if "flow" in observation_space.spaces:
            self.flow_adapter = nn.Sequential(
                nn.Linear(int(observation_space["flow"].shape[0]), 32), nn.GELU(),
                nn.Linear(32, 128))
            nn.init.zeros_(self.flow_adapter[-1].weight)
            nn.init.zeros_(self.flow_adapter[-1].bias)
        self.trunk = nn.Sequential(nn.Linear(32 + 32 + 32 + 32 + 32 + 32, 128),
                                   nn.LayerNorm(128), nn.GELU())

    @staticmethod
    def _masked_mean(x, valid):
        w = valid.to(dtype=x.dtype).unsqueeze(-1)
        return (x * w).sum(1) / w.sum(1).clamp_min(1.)

    def forward(self, obs):
        hmask = obs["hmask"].bool()
        h = self.gru(obs["hist"].masked_fill(hmask[..., None], 0.))[0][:, -1]
        image = self.cnn(obs["image"].float())
        ctx = self.ctx(obs["ctx"])
        obstacle = self._masked_mean(self.obstacle(obs["obstacles"]), obs["obstacle_valid"].bool())
        peer = self._masked_mean(self.peer(obs["peers"]), obs["peer_valid"].bool())
        candidate = self.candidate((obs["candidates"] * obs["candidate_valid"][..., None]).reshape(obs["candidates"].shape[0], -1))
        z = self.trunk(torch.cat([h, image, ctx, obstacle, peer, candidate], dim=-1))
        if self.flow_adapter is not None:
            z = z + self.flow_adapter(obs["flow"].float())
        return z


class CentralizedValueExtractor(BaseFeaturesExtractor):
    """Encode the training-only joint state for a MAPPO value function."""

    def __init__(self, observation_space):
        central = observation_space.spaces.get("central_state")
        if central is None:
            raise ValueError("MAPPO requires a central_state observation")
        super().__init__(observation_space, features_dim=128)
        width = int(np.prod(central.shape))
        self.net = nn.Sequential(
            nn.Linear(width, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
        )

    def forward(self, observations):
        state = observations["central_state"].float()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self.net(state.flatten(start_dim=1))


class MAPPOMultiInputActorCriticPolicy(MultiInputActorCriticPolicy):
    """Shared local actor with a centralized training-only value encoder."""

    def __init__(self, *args, **kwargs):
        kwargs["share_features_extractor"] = False
        kwargs["features_extractor_class"] = IPPOExtractor
        super().__init__(*args, **kwargs)
        old_optimizer = self.optimizer
        self.vf_features_extractor = CentralizedValueExtractor(self.observation_space)
        self.optimizer = self.optimizer_class(self.parameters(),
                                               lr=old_optimizer.param_groups[0]["lr"],
                                               **self.optimizer_kwargs)

    def get_distribution(self, obs):
        actor = self.mlp_extractor.forward_actor(self.pi_features_extractor(obs))
        return self._get_action_dist_from_latent(actor)

    def extract_features(self, obs):
        """Route actor and critic through their distinct observation paths."""
        pi_features = self.pi_features_extractor(obs)
        vf_features = self.vf_features_extractor(obs)
        return pi_features, vf_features


class MaskedMAPPOMultiInputActorCriticPolicy(MAPPOMultiInputActorCriticPolicy):
    """MAPPO policy using the same legal-candidate mask as the IPPO actor."""

    def _latent(self, obs):
        pi_features, vf_features = self.extract_features(obs)
        return (self.mlp_extractor.forward_actor(pi_features),
                self.mlp_extractor.forward_critic(vf_features))

    def _masked_distribution(self, obs, latent_pi):
        logits = self.action_net(latent_pi)
        mask = obs.get("candidate_valid")
        if mask is not None and logits.shape[-1] == mask.shape[-1]:
            mask = mask.to(dtype=torch.bool).clone()
            mask[:, 0] |= ~mask.any(dim=1)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic=False):
        latent_pi, latent_vf = self._latent(obs)
        values = self.value_net(latent_vf)
        distribution = self._masked_distribution(obs, latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    def evaluate_actions(self, obs, actions):
        latent_pi, latent_vf = self._latent(obs)
        distribution = self._masked_distribution(obs, latent_pi)
        return self.value_net(latent_vf), distribution.log_prob(actions), distribution.entropy()

    def _predict(self, observation, deterministic=False):
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def get_distribution(self, obs):
        latent_pi = self.mlp_extractor.forward_actor(self.pi_features_extractor(obs))
        return self._masked_distribution(obs, latent_pi)


class MAPPOMultiInputActorCriticPolicyV2(MAPPOMultiInputActorCriticPolicy):
    """Versioned MAPPO critic with per-cluster identity in the joint state."""


class MaskedMAPPOMultiInputActorCriticPolicyV2(MaskedMAPPOMultiInputActorCriticPolicy):
    """Masked V2 MAPPO critic, kept separate from the legacy checkpoint class."""


class MaskedMultiInputActorCriticPolicy(MultiInputActorCriticPolicy):
    """Multi-input PPO policy that masks unavailable semantic candidates.

    The original adapter canonicalised an invalid categorical sample after PPO
    had already computed its log-probability.  That wastes probability mass and
    gives the learner a different action than the one used for its update.  The
    mask is part of the deployable observation, so applying it to the logits in
    rollout and training keeps the categorical objective aligned with the
    executed command.
    """

    def _latent(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        pi_features, vf_features = features
        return (self.mlp_extractor.forward_actor(pi_features),
                self.mlp_extractor.forward_critic(vf_features))

    def _masked_distribution(self, obs, latent_pi):
        logits = self.action_net(latent_pi)
        mask = obs.get("candidate_valid")
        if mask is not None and logits.shape[-1] == mask.shape[-1]:
            mask = mask.to(dtype=torch.bool)
            # Passive drift can make slot zero unsafe. Use it as a fallback
            # only for an empty row from an external observation.
            mask = mask.clone()
            mask[:, 0] |= ~mask.any(dim=1)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic=False):
        latent_pi, latent_vf = self._latent(obs)
        values = self.value_net(latent_vf)
        distribution = self._masked_distribution(obs, latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        latent_pi, latent_vf = self._latent(obs)
        distribution = self._masked_distribution(obs, latent_pi)
        values = self.value_net(latent_vf)
        return values, distribution.log_prob(actions), distribution.entropy()

    def _predict(self, observation, deterministic=False):
        latent_pi, _ = self._latent(observation)
        return self._masked_distribution(observation, latent_pi).get_actions(deterministic=deterministic)

    def get_distribution(self, obs):
        latent_pi, _ = self._latent(obs)
        return self._masked_distribution(obs, latent_pi)


def critic_options(model):
    """Reconstruct the critic observation contract saved with the policy."""
    centralized = "central_state" in model.observation_space.spaces
    identity = isinstance(model.policy, (MAPPOMultiInputActorCriticPolicyV2,
                                         MaskedMAPPOMultiInputActorCriticPolicyV2))
    return dict(centralized_critic=centralized, critic_identity=identity)


def load_navigation_ppo(path, *, device="cpu", env=None, algorithm=PPO):
    """Use current implementations for legacy classes embedded by script-mode training."""
    data, _, _ = load_from_zip_file(path, device="cpu")
    overrides = {}
    policy = data["policy_class"]
    known = (MaskedMultiInputActorCriticPolicy, MAPPOMultiInputActorCriticPolicy,
             MaskedMAPPOMultiInputActorCriticPolicy, MAPPOMultiInputActorCriticPolicyV2,
             MaskedMAPPOMultiInputActorCriticPolicyV2)
    if policy.__module__ == "__main__":
        for cls in known:
            if cls.__name__ == policy.__name__:
                overrides["policy_class"] = cls
                break
    kwargs = data.get("policy_kwargs", {})
    extractor = kwargs.get("features_extractor_class")
    if extractor and extractor.__module__ == "__main__" and extractor.__name__ == "IPPOExtractor":
        overrides["policy_kwargs"] = dict(kwargs, features_extractor_class=IPPOExtractor)
    return algorithm.load(path, env=env, device=device, custom_objects=overrides)


class IPPOCallback(BaseCallback):
    def __init__(self, out, physical_steps, n_agents, save_every=8192):
        super().__init__()
        self.out = Path(out)
        self.target = int(physical_steps)
        self.n_agents = int(n_agents)
        self.save_every = int(save_every) * self.n_agents
        self.last_saved = 0
        self.started = time.monotonic()

    def _on_step(self):
        if self.n_calls % 256 == 0:
            env = self.training_env.physical
            status = dict(
                agent_steps=int(self.num_timesteps),
                environment_steps=int(self.num_timesteps // self.n_agents),
                target_environment_steps=self.target,
                minutes=(time.monotonic() - self.started) / 60.,
                frontier=int(env.frontier), difficulty=int(env.difficulty),
                scene=dict(env.scene),
            )
            tmp = self.out / "status.tmp"
            tmp.write_text(json.dumps(status, indent=2, ensure_ascii=False))
            tmp.replace(self.out / "status.json")
            # Lightweight live geometry snapshot for the NoMachine/browser viewer.
            # Keep it separate from status.json so the training status remains small.
            ep = env.ep.env
            _, targets, _, _ = env.current
            points = np.asarray(ep.transport.points, dtype=float)
            stride = max(1, int(np.ceil(len(points) / 1800)))
            flow = ep.transport.velocity_mm_s(ep.positions_mm[:env.clusters], ep.edges[:env.clusters], ep.solution)
            live = dict(updated=time.strftime("%Y-%m-%d %H:%M:%S"),
                        elapsed_s=float(ep.elapsed_s),
                        robot_positions_mm=np.asarray(ep.positions_mm[:env.clusters], float).round(5).tolist(),
                        robot_active=np.asarray(ep.active[:env.clusters], bool).tolist(),
                        target_indices=np.asarray(targets, int).tolist(),
                        target_positions_mm=np.asarray(ep.clot_positions_mm, float).round(5).tolist(),
                        flow_velocity_mm_s=np.asarray(flow, float).round(5).tolist(),
                        centerline_points_mm=points[::stride].round(5).tolist(),
                        scene=dict(env.scene))
            ltmp = self.out / "live_trace.tmp"
            ltmp.write_text(json.dumps(live, ensure_ascii=False))
            ltmp.replace(self.out / "live_trace.json")
        if not isinstance(self.model, PPO):
            self._save_if_due()
        return True

    def _on_rollout_start(self):
        if isinstance(self.model, PPO):
            self._save_if_due()

    def _on_training_end(self):
        if isinstance(self.model, PPO):
            self._save_if_due()

    def _save_if_due(self):
        if self.num_timesteps - self.last_saved >= self.save_every:
            checkpoint = self.out / f"policy_agent_steps{self.num_timesteps}.zip"
            temporary = self.out / "pending_checkpoint.zip"
            self.model.save(temporary)
            temporary.replace(checkpoint)
            self.last_saved = self.num_timesteps


def evaluate(model, out, clusters, count=12, action_set="semantic", anatomy=None,
             flow_feature=False, safety_parity=False, settle_radius=.3, exit_mask=False, centerline_match=False, fixed_scenes=None, horizon=300.):
    if count <= 0:
        summary = dict(episodes=0, navigation_success=None,
                       split="evaluation_skipped", deterministic=True)
        (Path(out) / "evaluation.json").write_text(json.dumps(summary, indent=2))
        return summary
    physical = CurriculumVascularEnv(
        seed=1041, clusters=clusters, curriculum=False, reward_mode="navigation", horizon=horizon,
        zero_latency=True, direct_action=True, centerline_match=centerline_match, fixed_scenes=fixed_scenes)
    if fixed_scenes:
        physical.train_anatomies = sorted({s["anatomy"] for s in fixed_scenes})
    elif anatomy:
        physical.train_anatomies = [anatomy]
    env = IPPOAgentVecEnv(physical, action_set=action_set, flow_feature=flow_feature,
                          safety_parity=safety_parity, settle_radius=settle_radius,
                          exit_mask=exit_mask,
                          **critic_options(model))
    rows = []
    obs = env.reset()
    try:
        while len(rows) < count:
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            if bool(dones[0]):
                info = infos[0]
                metrics = info.get("episode_metrics", {})
                rows.append(dict(navigation_success=bool(info["navigation_success"]),
                                 integration_failure=bool(info.get("integration_failure")),
                                 **{k: metrics.get(k) for k in ("seed", "anatomy", "difficulty",
                                    "navigation_reached", "wall_contact_s", "wall_after_reach_s", "lost",
                                    "particle_collision_events", "particle_contact_s")}))
        summary = dict(episodes=len(rows), navigation_success=float(np.mean([r["navigation_success"] for r in rows])),
                       integration_failures=sum(r["integration_failure"] for r in rows), rows=rows,
                       split="fixed_training_scenes" if fixed_scenes else "development_random_no_latency", deterministic=True)
        (Path(out) / "evaluation.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--timesteps", type=int, default=500000, help="target physical steps; PPO collects complete 256-step rollouts")
    p.add_argument("--clusters", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-count", type=int, default=12)
    p.add_argument("--ent-coef", type=float, default=0.01,
                   help="PPO entropy coefficient; higher values preserve exploration")
    p.add_argument("--action-set", choices=("semantic", "semantic_reduced", "semantic_minimal", "atomic", "continuous"), default="semantic")
    p.add_argument("--mask-invalid", action="store_true",
                   help="mask unavailable categorical candidates in the PPO logits")
    p.add_argument("--dataset-split", choices=("train",), default=None,
                   help="sample reset scenes sequentially from navigation_dataset_v1 train rows")
    p.add_argument("--dataset-limit", type=int, default=0,
                   help="limit manifest rows for a pilot; 0 uses all train rows")
    p.add_argument("--fixed-scenes", type=Path, default=None,
                   help="JSON list of recorded physical scenes with anatomy, seed, flow, difficulty, dynamic_obstacle_count and scenario_family.")
    p.add_argument("--horizon", type=float, default=300.,
                   help="episode horizon in seconds for training and self-evaluation")
    p.add_argument("--anatomy", default=None,
                   help="restrict training/evaluation scenes to one registered anatomy (e.g. hcc_portal_multifocal)")
    p.add_argument("--exit-mask", action="store_true",
                   help="mark candidates predicted to cross a boundary node as invalid. Loss is "
                        "irreversible and happens once per episode, so it carries almost no "
                        "gradient; masking removes the action instead of penalising it")
    p.add_argument("--safety-parity", action="store_true",
                   help="apply the same post-hoc guards the classical controllers get: forced settle "
                        "scaling inside the reach radius and WallGuard wall recovery. Without this the "
                        "RL path has neither, which is where its 10x loss rate comes from")
    p.add_argument("--settle-radius", type=float, default=.3,
                   help="reach radius inside which the command is scaled by d/radius")
    p.add_argument("--centerline-match", action="store_true",
                   help="match image positions to the closest mapped centerline with 0.06 mm hysteresis")
    p.add_argument("--flow-feature", action="store_true",
                   help="expose the online flow estimate (drift/gain/rmse/confidence) to the policy")
    p.add_argument("--flow-curriculum", action="store_true",
                   help="train through progressively stronger inlet flows")
    p.add_argument("--flow-boundaries", type=float, nargs="+",
                   default=(.2, .45, .7), metavar="FRACTION")
    p.add_argument("--flow-values", type=float, nargs="+",
                   default=(.025, .05, .1, .15), metavar="MM_S")
    p.add_argument("--time-cost", type=float, default=.01,
                   help="per-step time penalty; 0.02 makes a 300 s timeout cost -60 instead of -30")
    p.add_argument("--success-bonus", type=float, default=20.,
                   help="shared terminal credit for joint success")
    p.add_argument("--lost-penalty", type=float, default=8.,
                   help="penalty per lost cluster")
    p.add_argument("--dueling", action="store_true",
                   help="DQN only: V(s) + advantage centred over the valid candidates "
                        "(implies --mask-invalid, since the centring needs the mask)")
    p.add_argument("--algo", choices=("ppo", "mappo", "dqn"), default="ppo",
                   help="mappo uses the same local actor with a centralized training-only value function")
    p.add_argument("--extractor", choices=("mlp", "transformer"), default="mlp",
                   help="transformer keeps the six-branch topology and 128-d output, but encodes "
                        "the candidate/obstacle/peer sets with self-attention instead of "
                        "flatten/mean-pool")
    p.add_argument("--anatomy-repeat", default=None,
                   help="oversample one anatomy inside the mixed pool, e.g. hcc_portal_multifocal:9")
    p.add_argument("--init-from", type=Path, default=None,
                   help="warm-start policy weights from an existing SB3 .zip with the same action set")
    p.add_argument("--eval-anatomy", default=None,
                   help="anatomy for the post-training self-evaluation; defaults to the oversampled one")
    p.add_argument("--no-curriculum", action="store_true",
                   help="sample all four difficulty levels directly instead of waiting for navigation mastery")
    a = p.parse_args()
    if len(a.flow_values) != len(a.flow_boundaries) + 1:
        raise ValueError("--flow-values must have one more value than --flow-boundaries")
    if any(v <= 0. or v >= 1. for v in a.flow_boundaries):
        raise ValueError("--flow-boundaries must lie strictly between 0 and 1")
    if any(x >= y for x, y in zip(a.flow_boundaries, a.flow_boundaries[1:])):
        raise ValueError("--flow-boundaries must be strictly increasing")
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    dataset_rows = None
    dataset_sha256 = None
    if a.dataset_split and a.fixed_scenes:
        raise ValueError("--dataset-split and --fixed-scenes are mutually exclusive")
    if a.dataset_split:
        dataset_rows = list(iter_rows(split=a.dataset_split, clusters=a.clusters))
        if a.dataset_limit:
            dataset_rows = dataset_rows[:a.dataset_limit]
        if not dataset_rows:
            raise ValueError("dataset split produced no rows")
        dataset_sha256 = hashlib.sha256(Path(MANIFEST).read_bytes()).hexdigest()
    fixed_scenes = json.loads(a.fixed_scenes.read_text()) if a.fixed_scenes else None
    if a.fixed_scenes and not fixed_scenes:
        raise ValueError("empty fixed scene list")
    physical = CurriculumVascularEnv(
        seed=a.seed, clusters=a.clusters, budget=a.timesteps,
        curriculum=not a.no_curriculum, prediction=True, shaping=True, reward_mode="navigation", horizon=a.horizon,
        zero_latency=True, direct_action=True,
        episode_log=a.out / "episodes.jsonl", dataset_rows=dataset_rows, fixed_scenes=fixed_scenes, centerline_match=a.centerline_match,
        time_cost=a.time_cost, success_bonus=a.success_bonus, lost_penalty=a.lost_penalty,
        flow_curriculum=a.flow_curriculum, flow_boundaries=a.flow_boundaries,
        flow_values=a.flow_values)
    pool_default = set(physical.train_anatomies) | {"hcc_portal_multifocal"}
    if a.anatomy:
        # Keep the environment mechanics unchanged while replacing only the
        # anatomy pool.  A comma-separated list trains on a mixture: the
        # single-anatomy HCC run collapsed to 23.2% training success because the
        # pool carried no cross-anatomy variation, so a mixture is the
        # retraining protocol rather than a zero-shot transfer evaluation.
        names = [x.strip() for x in str(a.anatomy).split(",") if x.strip()]
        unknown = [n for n in names if n not in pool_default]
        if unknown:
            raise ValueError(f"unknown training anatomy: {', '.join(unknown)}")
        physical.train_anatomies = names
    if fixed_scenes:
        names = {row["anatomy"] for row in fixed_scenes}
        if not names <= pool_default:
            raise ValueError("unknown anatomy in fixed scenes")
        if not a.anatomy:
            physical.train_anatomies = sorted(names)
    if a.anatomy_repeat:
        # anatomy_bag reshuffles train_anatomies without replacement, so a
        # repeated entry oversamples that anatomy by exactly that factor.
        name, _, k = str(a.anatomy_repeat).partition(":")
        name = name.strip()
        factor = int(k) if k else 1
        if name not in pool_default:
            raise ValueError(f"unknown anatomy to oversample: {name}")
        if factor < 1:
            raise ValueError("--anatomy-repeat factor must be >= 1")
        physical.train_anatomies = [x for x in physical.train_anatomies if x != name] + [name] * factor
    env = IPPOAgentVecEnv(physical, action_set=a.action_set,
                          failure_log=str(a.out / "integration_failures.json"),
                          flow_feature=a.flow_feature,
                          safety_parity=a.safety_parity, settle_radius=a.settle_radius,
                          exit_mask=a.exit_mask, centralized_critic=(a.algo == "mappo"),
                          critic_identity=(a.algo == "mappo"))
    cfg = dict(algorithm=("IDQN_shared" if a.algo == "dqn" else "MAPPO_shared" if a.algo == "mappo" else "IPPO_shared"),
               critic_identity=(a.algo == "mappo"), optimiser=a.algo, feature_extractor=a.extractor, action_set=a.action_set,
               flow_feature=bool(a.flow_feature),
               safety_parity=bool(a.safety_parity), settle_radius=a.settle_radius,
               exit_mask=bool(a.exit_mask),
               reward_weights=dict(time_cost=a.time_cost, success_bonus=a.success_bonus,
                                   lost_penalty=a.lost_penalty, progress=1.5, newly_reached=8.,
                                   wall=2., particle_contact=2., particle_event=5., particle_near=.5),
               q_head=("dueling_masked" if (a.algo == "dqn" and a.dueling)
                       else "masked" if (a.algo == "dqn" and a.mask_invalid)
                       else "plain" if a.algo == "dqn" else None),
               framework="stable-baselines3 2.7.1",
               seed=a.seed, clusters=a.clusters, physical_steps=a.timesteps,
               agent_steps=a.timesteps * a.clusters, device=a.device,
               reward_mode="navigation", zero_latency=True, centerline_match=a.centerline_match,
               curriculum=not a.no_curriculum,
               flow_curriculum=bool(a.flow_curriculum),
               flow_boundaries=tuple(a.flow_boundaries),
               flow_values=tuple(a.flow_values),
               action_semantics=(f"per-cluster discrete atomic local velocity pulse (K={ATOMIC_CANDIDATE_COUNT})"
                                 if a.action_set == "atomic" else
                                 "per-cluster continuous direct local velocity, unit-ball bounded"
                                 if a.action_set == "continuous" else
                                 f"per-cluster discrete direct semantic candidate index (K={SEMANTIC_MINIMAL_COUNT if a.action_set == 'semantic_minimal' else SEMANTIC_REDUCED_COUNT if a.action_set == 'semantic_reduced' else SEMANTIC_CANDIDATE_COUNT})"),
               actor="parameter-shared local categorical actor; one candidate index per cluster",
               critic=("centralized MAPPO critic; joint deployable observation state" if a.algo == "mappo"
                       else "local IPPO critic; no privileged state"),
               architecture="GRU history + CNN image + map + masked obstacle/peer pooling",
               invalid_action_mask=bool(a.mask_invalid),
               dataset_split=a.dataset_split, dataset_rows=len(dataset_rows or []),
               dataset_manifest=(str(MANIFEST.relative_to(ROOT)) if a.dataset_split else
                                 str(a.fixed_scenes) if a.fixed_scenes else None),
               dataset_manifest_sha256=(hashlib.sha256(a.fixed_scenes.read_bytes()).hexdigest() if fixed_scenes else dataset_sha256),
               fixed_scene_count=len(fixed_scenes or []),
               training_anatomies=list(physical.train_anatomies),
               anatomy_pool=sorted(set(physical.train_anatomies)),
               anatomy_repeat=a.anatomy_repeat,
               warm_start_from=(str(a.init_from) if a.init_from else None),
               warm_start_sha256=(hashlib.sha256(Path(a.init_from).read_bytes()).hexdigest()
                                  if a.init_from else None),
               source_script=str(Path(__file__).resolve()))
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    if a.extractor == "transformer":
        from marl.transformer_extractor import TransformerIPPOExtractor
        extractor = TransformerIPPOExtractor
    else:
        extractor = IPPOExtractor
    if a.algo == "dqn":
        if a.action_set == "continuous":
            raise SystemExit("DQN requires a discrete action set")
        # Masking costs nothing on a Q-head: an invalid index would otherwise be
        # canonicalised at execution while still being regressed and maximised
        # over in the bootstrap target.
        if a.dueling:
            from marl.dqn_masked import DuelingMaskedDQNPolicy
            dqn_policy = DuelingMaskedDQNPolicy
        elif a.mask_invalid:
            from marl.dqn_masked import MaskedDQNPolicy
            dqn_policy = MaskedDQNPolicy
        else:
            dqn_policy = "MultiInputPolicy"
        # gamma and learning rate match the PPO runs so the comparison isolates
        # the optimiser, not the discounting.
        model = DQN(dqn_policy, env, seed=a.seed, device=a.device, verbose=0,
                    policy_kwargs=dict(features_extractor_class=extractor,
                                       normalize_images=False, net_arch=[128, 128]),
                    learning_rate=1e-4, gamma=.995, buffer_size=100_000,
                    learning_starts=5_000, batch_size=128, tau=1.0,
                    train_freq=4, gradient_steps=1, target_update_interval=2_000,
                    exploration_fraction=.2, exploration_initial_eps=1.0,
                    exploration_final_eps=.05)
    else:
        policy_kwargs = dict(features_extractor_class=extractor, normalize_images=False,
                             net_arch=dict(pi=[128, 128], vf=[128, 128]))
        if a.algo == "mappo":
            policy_name = (MaskedMAPPOMultiInputActorCriticPolicyV2 if a.mask_invalid and a.action_set != "continuous"
                           else MAPPOMultiInputActorCriticPolicyV2)
        else:
            policy_name = (MaskedMultiInputActorCriticPolicy if a.mask_invalid and a.action_set != "continuous"
                           else "MultiInputPolicy")
        model = PPO(policy_name, env, seed=a.seed, device=a.device, verbose=0,
                    policy_kwargs=policy_kwargs, learning_rate=1e-4, n_steps=256,
                    batch_size=128, n_epochs=4, gamma=.995, gae_lambda=.95,
                    clip_range=.2, ent_coef=a.ent_coef)
    if a.init_from:
        # Transfer policy weights while keeping the new run's optimiser fresh.
        _, parameters, _ = load_from_zip_file(a.init_from, device=model.device, load_data=False)
        model.policy.load_state_dict(parameters["policy"], strict=True)
    model.set_logger(configure(str(a.out), ["csv", "json", "tensorboard"]))
    model.save(a.out / "initial_model")
    started = time.monotonic()
    try:
        model.learn(total_timesteps=a.timesteps * a.clusters,
                    callback=IPPOCallback(a.out, a.timesteps, a.clusters))
        model.save(a.out / "final_model")
        summary = dict(environment_steps=physical.total_environment_steps, target_environment_steps=a.timesteps,
                       agent_steps=model.num_timesteps,
                       training_seconds=time.monotonic() - started)
        (a.out / "TRAINING_COMPLETE.json").write_text(json.dumps(summary, indent=2))
        eval_anatomy = a.eval_anatomy or (
            str(a.anatomy_repeat).partition(":")[0].strip() if a.anatomy_repeat
            else (physical.train_anatomies[0] if a.anatomy else None))
        result = evaluate(model, a.out, a.clusters, a.eval_count, a.action_set, eval_anatomy,
                          flow_feature=a.flow_feature, safety_parity=a.safety_parity,
                          settle_radius=a.settle_radius, exit_mask=a.exit_mask, centerline_match=a.centerline_match,
                          fixed_scenes=fixed_scenes, horizon=a.horizon)
        (a.out / "RUN_COMPLETE.json").write_text(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    from scripts.train_sb3_ippo import main as train
    train()
