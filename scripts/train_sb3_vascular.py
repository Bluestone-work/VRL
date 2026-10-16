"""Train/evaluate stock SB3 algorithms on a shared finite vascular task.

A centralized joint policy controls N fixed robot slots. This is not MAPPO.
Only the feature extractor and Gymnasium adapter are task-specific; SB3 owns
rollout/replay, GAE, targets, optimizers and exploration. No pretrained actor,
motion predictor, reference action, or action prior is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from marl.deployable_sensing import DeployableConfig
from marl.dyn_predictor import DynFeatures
from marl.obstacle_features import forecast_features
from marl.vessel_sdf import VesselSDF, wall_safe_projection
from scripts.benchmark_lysis import LysisEpisode

ALGORITHMS = {"ppo": PPO, "sac": SAC, "td3": TD3}


class VascularSB3Env(gym.Env):
    """One joint action and one shared reward per physical control step.

    The 300 s deadline is an actual finite-task terminal, not a training cutoff.
    Remaining time is observable. Rollout boundaries never reset the episode.
    Gymnasium returns the terminal observation; SB3 alone performs auto-reset.
    """
    metadata = {"render_modes": []}

    def __init__(self, seed=0, clusters=3, dynamic_obstacle_count=16, horizon=300., episode_log=None,
                 preserve_action_magnitude=False, direct_action=False):
        super().__init__()
        if clusters not in (1, 2, 3) or horizon <= 0 or dynamic_obstacle_count < 0:
            raise ValueError("clusters must be 1..3, horizon positive, obstacle count nonnegative")
        self.clusters = clusters
        self.dynamic_obstacle_count = dynamic_obstacle_count
        self.horizon = horizon
        self.episode_log = Path(episode_log) if episode_log else None
        # Legacy joint SB3 runs normalize each 3-D action to a direction.
        # IPPO navigation runs can opt into a unit-ball command so magnitude
        # remains observable and learnable for slow/hold behaviour.
        self.preserve_action_magnitude = bool(preserve_action_magnitude)
        # Direct local commands already carry their intended speed. Legacy
        # continuous SB3 runs keep their historical normalization semantics.
        self.direct_action = bool(direct_action)
        self.rng = np.random.default_rng(seed)
        self.scene_base = 2813000000 + seed * 100000
        self.train_anatomies = json.loads((ROOT / "configs/evaluation_splits.json").read_text())["anatomy_holdout_v1"]["train"]
        n = clusters
        self.observation_space = spaces.Dict({
            "hist": spaces.Box(-np.inf, np.inf, (n, 24, 11), np.float32),
            "hmask": spaces.Box(0., 1., (n, 24), np.float32),
            "ctx": spaces.Box(-np.inf, np.inf, (n, 32), np.float32),
            "image": spaces.Box(-.3, 2., (n, 4, 32, 32), np.float16),
        })
        self.action_space = spaces.Box(-1., 1., (n * 3,), np.float32)
        self.ep = None
        self.episode_index = 0
        self.current = None
        self.ended = True
        self.total_environment_steps = 0
        self.total_agent_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.scene_base = 2813000000 + seed * 100000
            self.episode_index = 0
        options = options or {}
        if self.ep is not None:
            self.ep.close()
            self.ep = None
        anatomy = options.get("anatomy") or str(self.rng.choice(self.train_anatomies))
        if anatomy not in self.train_anatomies:
            raise ValueError("this development adapter accepts training anatomies only")
        flow = float(options.get("flow", self.rng.choice([.025, .05, .1])))
        latency = int(options.get("latency", self.rng.integers(1, 3)))
        scene_seed = int(options.get("scene_seed", self.scene_base + self.episode_index))
        self.episode_index += 1
        self.ep = LysisEpisode(
            self.clusters, anatomy, scene_seed, horizon=self.horizon,
            flow_inlet_mm_s=flow, sense_cfg=DeployableConfig(latency_steps=latency),
            dynamic_obstacle_count=self.dynamic_obstacle_count,
        )
        self.scene = dict(anatomy=anatomy, clusters=self.clusters, seed=scene_seed, flow=flow, latency=latency)
        self.features = DynFeatures(self.ep)
        self.sdf = VesselSDF(self.ep.sensor)
        self.previous_images = np.zeros((self.clusters, 2, 32, 32), np.float32)
        self.previous_frame = None
        self.spacing_previous = 0.
        self.return_sum = 0.
        self.episode_agent_steps = 0
        self.safety_changed = 0
        self.ended = False
        return self._observe(), {"scene": dict(self.scene)}

    def _observe(self):
        ep = self.ep
        est = ep.observe()
        targets = ep.plan_targets_now(est)
        rule = ep.ctl.act(targets, est)  # update route frames/carrot
        hold = ep.hold(est)
        frames = ep.ctl.frames(est)
        self.features.push(ep, est, hold)
        histories, masks, contexts = [], [], []
        for i in range(self.clusters):
            hist, mask = self.features.history(ep, est, i, frames[i])
            # Map geometry, delayed detections, issued-command history only.
            ctx = np.zeros(32, np.float32)
            ctx[:19] = self.features.context(ep, est, targets, i, frames[i])
            ctx[19:25] = forecast_features(ep, est, i)
            ctx[25:28] = [est.active[i], hold[i], targets[i] >= 0]
            ctx[28] = min(float(ep.env.elapsed_s) / self.horizon, 1.)
            ctx[29] = float(np.asarray(est.clot_alive).mean())
            ctx[30] = float(ep.env.config.control_dt_s)
            ctx[31] = self.clusters / 3.
            histories.append(hist)
            masks.append(mask)
            contexts.append(ctx)
        images = np.asarray(ep.sensor.crops, np.float32)
        image_pair = np.concatenate([images, self.previous_images], axis=1)
        frame = float(getattr(est, "frame_time_s", ep.env.elapsed_s))
        if frame != self.previous_frame:
            self.previous_images = images.copy()
            self.previous_frame = frame
        # Expose the deployable route command to the candidate-action adapter.
        # It is computed from the same map and estimated state as the actor;
        # no simulator flow, response gain, or true position is added.
        self.current_rule = np.asarray(rule, np.float32).copy()
        self.current = (est, targets, hold, frames)
        obs = dict(hist=np.asarray(histories, np.float32), hmask=np.asarray(masks, np.float32),
                   ctx=np.asarray(contexts, np.float32), image=np.clip(image_pair, -.3, 2.).astype(np.float16))
        if not all(np.isfinite(v).all() for v in obs.values()):
            raise FloatingPointError("non-finite SB3 observation")
        return obs

    def step(self, action):
        if self.ended:
            raise RuntimeError("reset() is required before stepping a new/finished episode")
        action = np.asarray(action, np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("action must be finite and match the joint action space")
        ep = self.ep
        est, targets, hold, frames = self.current
        local = np.clip(action, -1., 1.).reshape(self.clusters, 3).astype(float)
        if self.direct_action:
            # Candidate actions are deployable route-frame commands; do not
            # reinterpret them as residuals or renormalize their speed.
            pass
        elif self.preserve_action_magnitude:
            # Box[-1,1]^3 -> unit ball.  The vector direction and its norm are
            # retained; diagonal commands cannot exceed the physical unit
            # speed.  This gives the policy an actual low-speed/hold action.
            local /= np.sqrt(3.)
        else:
            local /= np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1.)
        # Fixed geometric safety projection; no learned selector or recovery action.
        for i in range(self.clusters):
            if not est.active[i] or hold[i] or targets[i] < 0:
                local[i] = 0.
            else:
                world = frames[i].T @ local[i]
                guarded = wall_safe_projection(world, est.pos[i], self.sdf, ep.ctl.body,
                                               margin=.2, edge=int(est.edge[i]))
                self.safety_changed += int(np.linalg.norm(guarded - world) > 1e-6)
                local[i] = frames[i] @ guarded
        done, out = ep.step(est, local, hold)
        active_steps = int(np.sum(out["active_before"]))
        self.total_agent_steps += active_steps
        self.episode_agent_steps += active_steps
        self.total_environment_steps += 1
        spacing = float(ep.spacing.summary()["spacing_violation_pair_s"])
        reward = (10. * float(out["removed"]) - 2. * float(np.sum(out["wall"]))
                  - 2. * (spacing - self.spacing_previous) - 10. * float(np.sum(out["lost"]))
                  - .01 * float(ep.info["step_duration_s"])
                  - 2. * float(np.sum(ep.info.get("particle_contact_s", 0.))))
        self.spacing_previous = spacing
        self.return_sum += reward
        info = dict(environment_steps=self.total_environment_steps, agent_steps=self.total_agent_steps,
                    action_saturation=float(np.mean(np.abs(action) > .95)))
        if done:
            row = ep.row("sb3_joint_direct", return_sum=self.return_sum, flow=self.scene["flow"],
                         latency=self.scene["latency"], environment_steps=ep.steps,
                         agent_steps=self.episode_agent_steps, wall_projection_changes=self.safety_changed)
            # Deadline is a finite-task terminal: do not bootstrap beyond it.
            info.update(episode_metrics=row, is_success=bool(row["strict_success"]))
            if self.episode_log:
                with self.episode_log.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.ended = True
        obs = self._observe()
        return obs, float(reward), bool(done), False, info

    def close(self):
        if self.ep is not None:
            self.ep.close()
            self.ep = None
        self.current = None
        self.ended = True


class VascularExtractor(BaseFeaturesExtractor):
    """Randomly initialized, shared CNN/history encoder, then joint features."""
    def __init__(self, observation_space):
        n = observation_space["ctx"].shape[0]
        super().__init__(observation_space, features_dim=128 * n)
        self.n = n
        self.cnn = nn.Sequential(nn.Conv2d(4, 16, 5, 2, 2), nn.ReLU(),
                                 nn.Conv2d(16, 24, 3, 2, 1), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(384, 32), nn.Tanh())
        self.gru = nn.GRU(11, 64, batch_first=True)
        self.ctx = nn.Sequential(nn.Linear(32, 32), nn.Tanh())

    def forward(self, obs):
        b = obs["ctx"].shape[0]
        hist = obs["hist"].reshape(b * self.n, 24, 11)
        mask = obs["hmask"].reshape(b * self.n, 24).bool()
        history = self.gru(hist.masked_fill(mask[..., None], 0.))[0][:, -1]
        image = self.cnn(obs["image"].reshape(b * self.n, 4, 32, 32))
        context = self.ctx(obs["ctx"].reshape(b * self.n, 32))
        return torch.cat([history, context, image], dim=-1).reshape(b, -1)


class RunCallback(BaseCallback):
    def __init__(self, out, save_every):
        super().__init__()
        self.out, self.save_every = out, save_every
        self.started = time.monotonic()
        self.last_saved = 0

    def _on_step(self):
        if self.n_calls % 256 == 0:
            row = dict(environment_steps=self.num_timesteps,
                       agent_steps=sum(i["agent_steps"] for i in self.locals["infos"]),
                       minutes=(time.monotonic() - self.started) / 60,
                       action_saturation=float(np.mean([i["action_saturation"] for i in self.locals["infos"]])))
            status_path = self.out / "status.json"
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(row, indent=2))
            temporary.replace(status_path)
        if self.num_timesteps - self.last_saved >= self.save_every:
            self.model.save(self.out / f"policy_steps{self.num_timesteps}")
            self.last_saved = self.num_timesteps
        return True


def build_model(algorithm, env, seed=0, device="cpu", smoke=False, extractor_class=VascularExtractor):
    policy_kwargs = dict(features_extractor_class=extractor_class, normalize_images=False,
                         net_arch=dict(pi=[128, 128], vf=[128, 128]) if algorithm == "ppo" else [128, 128])
    common = dict(policy="MultiInputPolicy", env=env, seed=seed, device=device,
                  verbose=0, policy_kwargs=policy_kwargs, gamma=.9995, learning_rate=3e-4)
    if algorithm == "ppo":
        return PPO(**common, n_steps=256, batch_size=64, n_epochs=4 if smoke else 10,
                   gae_lambda=.95, clip_range=.2, ent_coef=0.)
    offpolicy = dict(buffer_size=512 if smoke else 10000, learning_starts=64 if smoke else 2000,
                     train_freq=16, gradient_steps=1 if smoke else 16,
                     batch_size=32 if smoke else 128, tau=.005)
    if algorithm == "sac":
        return SAC(**common, **offpolicy, ent_coef="auto")
    dims = env.action_space.shape[0]
    noise = NormalActionNoise(np.zeros(dims), .1 * np.ones(dims))
    return TD3(**common, **offpolicy, action_noise=noise, policy_delay=2)


def evaluate(model, out, clusters, obstacles, count=3):
    """Fixed development scenes on train anatomies; never access sealed tests."""
    env = VascularSB3Env(clusters=clusters, dynamic_obstacle_count=obstacles)
    rows = []
    try:
        splits = json.loads((ROOT / "configs/evaluation_splits.json").read_text())
        all_anatomies = splits["anatomy_order"]
        development = splits["multicluster_benchmark_v1"]["development"]
        if count > len(env.train_anatomies) * development["count"]:
            raise ValueError("requested evaluation exceeds the registered development pool")
        with (out / "evaluation.jsonl").open("w", encoding="utf-8") as f:
            for i in range(count):
                anatomy = env.train_anatomies[i % len(env.train_anatomies)]
                options = dict(anatomy=anatomy, scene_seed=development["seed_base"] + all_anatomies.index(anatomy) * development["stride_per_anatomy"] + i // len(env.train_anatomies),
                               flow=[.025, .05, .1][i % 3], latency=1 + i % 2)
                obs, _ = env.reset(options=options)
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    if done or truncated:
                        row = dict(info["episode_metrics"], split="multicluster_development")
                        rows.append(row)
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        break
        summary = dict(episodes=len(rows), strict_success=float(np.mean([r["strict_success"] for r in rows])),
                       removal=float(np.mean([r["removal"] for r in rows])),
                       wall_contact_s=float(np.mean([r["wall_contact_s"] for r in rows])),
                       split="multicluster_development", deterministic=True)
        (out / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--timesteps", type=int, default=100000)
    p.add_argument("--clusters", type=int, default=3)
    p.add_argument("--dynamic-obstacles", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--eval-count", type=int, default=3)
    p.add_argument("--model", type=Path, help="evaluate saved SB3 zip only; skip training")
    a = p.parse_args()
    if a.timesteps <= 0 or not 1 <= a.eval_count <= 270 or not 0 <= a.seed <= 10000:
        p.error("timesteps must be positive, eval-count 1..270, seed 0..10000")
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=False)
    if a.model:
        source_cfg = json.loads((a.model.parent / "config.json").read_text())
        if (source_cfg["algorithm"], source_cfg["clusters"], source_cfg["dynamic_obstacles"]) != (a.algorithm, a.clusters, a.dynamic_obstacles):
            raise ValueError("evaluation configuration differs from saved model")
        (a.out / "config.json").write_text(json.dumps(dict(source_cfg, model=str(a.model.resolve()), model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest()), indent=2))
        model = ALGORITHMS[a.algorithm].load(a.model, device=a.device)
        evaluate(model, a.out, a.clusters, a.dynamic_obstacles, a.eval_count)
        return
    timesteps = 512 if a.smoke else a.timesteps
    cfg = dict(algorithm=a.algorithm, seed=a.seed, environment_steps_requested=timesteps, clusters=a.clusters,
               dynamic_obstacles=a.dynamic_obstacles, action_semantics="joint_direct_route_frame_unit_ball",
               framework="stable-baselines3 2.7.1", gamma=.9995, horizon_s=300.,
               initialization="random; no pretrained weights, predictor or action prior",
               retained_components="preoperative allocation/routes, TPG, geometric wall projection, spacing shield",
               policy="centralized joint control; shared per-robot CNN/GRU, joint SB3 actor/critic",
               reward="10*removal -2*wall_s -2*spacing_pair_s -10*lost -.01*dt -2*particle_contact_s",
               reward_shaping=False, budget_unit="one actual joint physical environment step",
               device=a.device, smoke=a.smoke, eval_count=a.eval_count,
               source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                              for folder in ("scripts", "marl", "environments")
                              for f in sorted((ROOT / folder).glob("*.py"))})
    for name in cfg["source_sha256"]:
        snapshot = a.out / "source_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, snapshot)
    (a.out / "source_snapshot/configs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "configs/evaluation_splits.json", a.out / "source_snapshot/configs/evaluation_splits.json")
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2))
    env = DummyVecEnv([lambda: Monitor(VascularSB3Env(seed=a.seed, clusters=a.clusters,
                           dynamic_obstacle_count=a.dynamic_obstacles, episode_log=a.out / "episodes.jsonl"))])
    try:
        model = build_model(a.algorithm, env, a.seed, a.device, a.smoke)
        cfg["hyperparameters"] = {k: getattr(model, k) for k in
            ("learning_rate", "gamma", "batch_size", "n_steps", "n_epochs", "gae_lambda", "buffer_size", "learning_starts", "gradient_steps", "tau", "policy_delay") if hasattr(model, k)}
        cfg["train_freq"] = [model.train_freq.frequency, model.train_freq.unit.value] if hasattr(model, "train_freq") else None
        cfg["hyperparameters"] = {k: v if isinstance(v, (int, float, str, bool)) else str(v) for k, v in cfg["hyperparameters"].items()}
        (a.out / "config.json").write_text(json.dumps(cfg, indent=2))
        model.set_logger(configure(str(a.out), ["csv", "json", "tensorboard"]))
        model.save(a.out / "initial_model")
        initial = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
        started = time.monotonic()
        model.learn(total_timesteps=timesteps, callback=RunCallback(a.out, 10000))
        model.save(a.out / "final_model")
        if a.algorithm != "ppo":
            model.save_replay_buffer(a.out / "replay_buffer.pkl")
        change = sum(float((v.detach().cpu() - initial[k]).square().sum()) for k, v in model.policy.state_dict().items()) ** .5
        completion = dict(environment_steps=model.num_timesteps, updates=model._n_updates,
                          parameter_change_l2=change, training_seconds=time.monotonic() - started,
                          checkpoint_resume="SB3 model/optimizer and replay saved; in-flight episode/RNG not restored exactly")
        (a.out / "TRAINING_COMPLETE.json").write_text(json.dumps(completion, indent=2))
        print(json.dumps(completion), flush=True)
    finally:
        env.close()
    summary = evaluate(model, a.out, a.clusters, a.dynamic_obstacles, a.eval_count)
    print(json.dumps(summary), flush=True)
    (a.out / "RUN_COMPLETE.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
