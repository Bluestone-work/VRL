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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
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
    if action_set == "semantic_reduced":
        commands, valid = direct_candidates(physical, "semantic")
        idx = np.asarray(SEMANTIC_REDUCED_INDICES, dtype=np.int64)
        return commands[:, idx], valid[:, idx]
    if action_set == "atomic":
        return atomic_candidates(physical)
    if action_set != "semantic":
        raise ValueError(f"unknown action_set: {action_set}")
    est, targets, hold, frames = physical.current
    ep = physical.ep
    rule = np.asarray(getattr(physical, "current_rule", np.zeros((physical.clusters, 3))), np.float32)
    n = physical.clusters
    commands = np.zeros((n, SEMANTIC_CANDIDATE_COUNT, 3), np.float32)
    valid = np.ones((n, SEMANTIC_CANDIDATE_COUNT), np.float32)
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
        if np.linalg.norm(tangent) > 1e-8:
            # Deterministic local orthogonal basis; the route frame is already
            # supplied by the environment, so this is only a fallback for a
            # degenerate nominal command.
            normal = np.array([0., 1., 0.], np.float32)
            binormal = np.array([0., 0., 1.], np.float32)
        commands[i] = np.asarray([
            np.zeros(3), r, tangent, .75*tangent, .5*tangent,
            target, .5*target, .25*target, settle, -tangent,
            .35*normal, -.35*normal, .35*binormal, -.35*binormal,
            .7*normal, -.7*normal, .7*binormal, -.7*binormal,
        ], np.float32)
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

    def __init__(self, physical_env: CurriculumVascularEnv, action_set="semantic"):
        self.physical = physical_env
        if action_set not in ("semantic", "semantic_reduced", "atomic", "continuous"):
            raise ValueError("action_set must be semantic, semantic_reduced, atomic, or continuous")
        self.action_set = action_set
        # Continuous actors still receive the compact atomic basis as context,
        # but their output is a free 3-D local velocity rather than an index.
        self.candidate_count = (ATOMIC_CANDIDATE_COUNT
                                if action_set in ("atomic", "continuous")
                                else SEMANTIC_CANDIDATE_COUNT)
        if action_set == "semantic_reduced":
            self.candidate_count = SEMANTIC_REDUCED_COUNT
        self.n_agents = physical_env.clusters
        n = self.n_agents
        self._pending_actions = None
        self._obs = None
        self._candidate_commands = None
        self._candidate_valid = None
        self._last_reset_info = None
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
        })
        action_space = (spaces.Box(-1., 1., (3,), np.float32)
                        if action_set == "continuous" else
                        spaces.Discrete(self.candidate_count))
        super().__init__(n, observation_space, action_space)
        self.reset_infos = [{} for _ in range(n)]

    def _split(self, obs):
        return [{k: np.asarray(v[i]).copy() for k, v in obs.items()} for i in range(self.n_agents)]

    def _stack(self, rows):
        return {k: np.stack([r[k] for r in rows], axis=0) for k in rows[0]}

    def _with_candidates(self, obs):
        feature_set = "atomic" if self.action_set == "continuous" else self.action_set
        commands, valid = direct_candidates(self.physical, feature_set)
        self._candidate_commands = commands
        self._candidate_valid = valid
        enriched = dict(obs)
        enriched["candidates"] = commands
        enriched["candidate_valid"] = valid
        return enriched

    def reset(self, options=None):
        obs, info = self.physical.reset(options=options)
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
                    chosen[i] = int(valid_same[0]) if len(valid_same) else 0
            joint = self._candidate_commands[np.arange(self.n_agents), chosen].reshape(-1)
        obs, shared_reward, terminated, truncated, info = self.physical.step(joint)
        done = bool(terminated or truncated)
        rewards = np.asarray(info.get("agent_rewards", np.full(self.n_agents, shared_reward)), dtype=np.float32).reshape(-1)
        if rewards.size == 1:
            rewards = np.full(self.n_agents, float(rewards[0]), np.float32)
        if rewards.size != self.n_agents:
            rewards = np.full(self.n_agents, float(shared_reward), np.float32)
        infos = [dict(info) for _ in range(self.n_agents)]
        obs = self._with_candidates(obs)
        next_rows = self._split(obs)
        if done:
            # Preserve the terminal local observation for SB3 diagnostics and
            # immediately reset the one physical episode for the next rollout.
            for i in range(self.n_agents):
                infos[i]["terminal_observation"] = next_rows[i]
                infos[i]["TimeLimit.truncated"] = bool(truncated)
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
        return self.trunk(torch.cat([h, image, ctx, obstacle, peer, candidate], dim=-1))


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
        if self.num_timesteps - self.last_saved >= self.save_every:
            self.model.save(self.out / f"policy_agent_steps{self.num_timesteps}")
            self.last_saved = self.num_timesteps
        return True


def evaluate(model, out, clusters, count=12, action_set="semantic"):
    env = IPPOAgentVecEnv(CurriculumVascularEnv(
        seed=1041, clusters=clusters, curriculum=False, reward_mode="navigation",
        zero_latency=True, direct_action=True), action_set=action_set)
    rows = []
    obs = env.reset()
    try:
        while len(rows) < count:
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            if bool(dones[0]):
                info = infos[0]
                rows.append(dict(
                    navigation_success=bool(info.get("navigation_success", False)),
                    navigation_reached=int(info.get("navigation_reached", 0)),
                    wall_contact_s=float(np.asarray(info.get("agent_wall_contact_s", 0.)).sum()),
                    lost=int(np.asarray(info.get("agent_lost", 0.)).sum()),
                    particle_collision_events=int(np.asarray(info.get("agent_particle_collision_events", 0.)).sum()),
                    particle_contact_s=float(np.asarray(info.get("agent_particle_contact_s", 0.)).sum()),
                ))
        summary = dict(episodes=len(rows), navigation_success=float(np.mean([r.get("navigation_success", False) for r in rows])),
                       split="development_random_no_latency", deterministic=True)
        (Path(out) / "evaluation.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--timesteps", type=int, default=500000, help="physical environment steps")
    p.add_argument("--clusters", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-count", type=int, default=12)
    p.add_argument("--action-set", choices=("semantic", "semantic_reduced", "atomic", "continuous"), default="semantic")
    p.add_argument("--dataset-split", choices=("train",), default=None,
                   help="sample reset scenes sequentially from navigation_dataset_v1 train rows")
    p.add_argument("--dataset-limit", type=int, default=0,
                   help="limit manifest rows for a pilot; 0 uses all train rows")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    dataset_rows = None
    dataset_sha256 = None
    if a.dataset_split:
        dataset_rows = list(iter_rows(split=a.dataset_split, clusters=a.clusters))
        if a.dataset_limit:
            dataset_rows = dataset_rows[:a.dataset_limit]
        if not dataset_rows:
            raise ValueError("dataset split produced no rows")
        dataset_sha256 = hashlib.sha256(Path(MANIFEST).read_bytes()).hexdigest()
    physical = CurriculumVascularEnv(
        seed=a.seed, clusters=a.clusters, budget=a.timesteps,
        curriculum=True, prediction=True, shaping=True, reward_mode="navigation",
        zero_latency=True, direct_action=True,
        episode_log=a.out / "episodes.jsonl", dataset_rows=dataset_rows)
    env = IPPOAgentVecEnv(physical, action_set=a.action_set)
    cfg = dict(algorithm="IPPO_shared", framework="stable-baselines3 2.7.1",
               seed=a.seed, clusters=a.clusters, physical_steps=a.timesteps,
               agent_steps=a.timesteps * a.clusters, device=a.device,
               reward_mode="navigation", zero_latency=True,
               action_semantics=(f"per-cluster discrete atomic local velocity pulse (K={ATOMIC_CANDIDATE_COUNT})"
                                 if a.action_set == "atomic" else
                                 "per-cluster continuous direct local velocity, unit-ball bounded"
                                 if a.action_set == "continuous" else
                                 f"per-cluster discrete direct semantic candidate index (K={SEMANTIC_REDUCED_COUNT if a.action_set == 'semantic_reduced' else SEMANTIC_CANDIDATE_COUNT})"),
               actor="parameter-shared local categorical actor; one candidate index per cluster",
               critic="local IPPO critic; no privileged state",
               architecture="GRU history + CNN image + map + masked obstacle/peer pooling",
               dataset_split=a.dataset_split, dataset_rows=len(dataset_rows or []),
               dataset_manifest=str(MANIFEST.relative_to(ROOT)) if dataset_rows is not None else None,
               dataset_manifest_sha256=dataset_sha256,
               source_script=str(Path(__file__).resolve()))
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    policy_kwargs = dict(features_extractor_class=IPPOExtractor, normalize_images=False,
                         net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO("MultiInputPolicy", env, seed=a.seed, device=a.device, verbose=0,
                policy_kwargs=policy_kwargs, learning_rate=1e-4, n_steps=256,
                batch_size=128, n_epochs=4, gamma=.995, gae_lambda=.95,
                clip_range=.2, ent_coef=.01)
    model.set_logger(configure(str(a.out), ["csv", "json", "tensorboard"]))
    model.save(a.out / "initial_model")
    started = time.monotonic()
    try:
        model.learn(total_timesteps=a.timesteps * a.clusters,
                    callback=IPPOCallback(a.out, a.timesteps, a.clusters))
        model.save(a.out / "final_model")
        summary = dict(environment_steps=a.timesteps, agent_steps=model.num_timesteps,
                       training_seconds=time.monotonic() - started)
        (a.out / "TRAINING_COMPLETE.json").write_text(json.dumps(summary, indent=2))
        result = evaluate(model, a.out, a.clusters, a.eval_count, a.action_set)
        (a.out / "RUN_COMPLETE.json").write_text(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
