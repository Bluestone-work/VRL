"""Single-agent planar transfer using the upstream vessel simulator and released weights."""
import argparse
import hashlib
import json
import math
import random
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from alg.actor_critic import ActorCritic
from alg.normalizer import EmpiricalNormalization
from cfg.vessel_env_cfg import VesselEnvCfg
from env.vessel_env import VesselEnv
from scripts.train_sb3_ippo import IPPOExtractor


class ManualResetVessel(VesselEnv):
    """Expose the native terminal state before the caller begins the next episode."""
    stepping = False

    def reset(self, reset_ids):
        if not self.stepping:
            super().reset(reset_ids)

    def advance(self, action):
        self.stepping = True
        try:
            return super().step(action)
        finally:
            self.stepping = False


def make_map(root, number):
    directory = root / str(number)
    directory.mkdir()
    rng = random.Random(number)
    image = Image.new("RGB", (128, 128))
    pen = ImageDraw.Draw(image)
    fork = (65, rng.randrange(50, 79))
    for points in ([(7, 66), (34, rng.randrange(41, 88)), fork],
                   [fork, (91, rng.randrange(28, 54)), (119, 22)],
                   [fork, (91, rng.randrange(78, 101)), (119, 108)]):
        pen.line(points, fill="white", width=rng.randrange(9, 17), joint="curve")
    image.save(directory / "map.png")
    cfg = VesselEnvCfg()
    cfg.num_envs = 1
    cfg.sim = cfg.sim()
    cfg.sim.sim_vessel2d_dataset_path = str(directory)
    return ManualResetVessel(cfg), hashlib.sha256(image.tobytes()).hexdigest()


class PlanarTask(gym.Env):
    """VRL local encoder sees only deterministic transforms of the native 77 values."""
    def __init__(self, maps):
        self.maps = maps
        self.action_space = gym.spaces.Box(-np.pi, np.pi, (1,), np.float32)
        shapes = dict(hist=(1,11), hmask=(1,), image=(4,32,32), ctx=(32,),
                      obstacles=(36,25), obstacle_valid=(36,), peers=(1,8), peer_valid=(1,),
                      candidates=(1,3), candidate_valid=(1,))
        self.observation_space = gym.spaces.Dict({
            k: gym.spaces.Box(-np.inf, np.inf, shape, np.float32) for k, shape in shapes.items()
        })

    def encode(self):
        native = self.env.obs[0].detach().cpu().numpy()
        points, target, pos = native[:72].reshape(36, 2), native[72:74], native[74:76]
        delta = (points - pos) / self.env.lidar_max_depth
        obs = {k: np.zeros(space.shape, np.float32) for k, space in self.observation_space.spaces.items()}
        obs["ctx"][:7] = [*list(pos/128), *list(target/128), *list((target-pos)/128), native[-1]/np.pi]
        obs["hist"][0, :7] = obs["ctx"][:7]
        obs["obstacles"][:, :2] = delta
        obs["obstacle_valid"][:] = 1.
        x, y = np.clip(np.rint(15.5 + 15.5 * delta), 0, 31).astype(int).T
        obs["image"][0, y, x] = 1.
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = options["map"] if options else int(self.np_random.integers(len(self.maps)))
        self.env, self.map_hash = self.maps[index]
        if seed is not None:
            torch.manual_seed(seed)
        self.env.reset([0])
        self.env.actions.zero_()
        self.env.compute_observation()
        return self.encode(), {}

    def step(self, action):
        e = self.env
        cmd = torch.as_tensor(action, dtype=torch.float32, device=e.device).reshape(1, 1)
        _, reward, done, _ = e.advance(cmd)
        collision = bool((~e.in_vessel | e.out_of_bound_buffer).item())
        reached = bool(e.targets_reached.item())
        terminal = reached or collision
        info = dict(reached=reached, collision=collision, strict_success=reached and not collision)
        return self.encode(), float(reward.item()), terminal, bool(done.item()) and not terminal, info


def load_author(checkpoint, device):
    bundle = torch.load(checkpoint, map_location=device, weights_only=True)
    net = ActorCritic(77, 77, 1, actor_hidden_dims=[256,128,16], critic_hidden_dims=[256,128,16],
                      activation="elu", distribution_type="normal").to(device)
    norm = EmpiricalNormalization([77], until=1e8).to(device)
    net.load_state_dict(bundle["model_state_dict"])
    norm.load_state_dict(bundle["obs_norm_state_dict"])
    net.eval()
    norm.eval()
    return lambda x: net.act_inference(norm(x)).flatten().cpu().numpy()


@torch.inference_mode()
def evaluate_scene(task, model, author, i):
    rows = []
    seed = 1000000 + i
    identity = None
    for method in ("mr_nav_pretrained", "vrl_encoder_planar_ppo", "target_heading"):
        obs, _ = task.reset(seed=seed, options={"map": i})
        e = task.env
        start, goal = e.pos[0].tolist(), e.targets[0].tolist()
        current = (task.map_hash, start, goal)
        if identity is not None and current != identity:
            raise RuntimeError(f"initial state differs at scene {i}")
        identity = current
        # Independent disturbance stream, replayed for every deterministic controller.
        torch.manual_seed(2000000 + i)
        for step in range(1, e.max_episode_length + 2):
            if method == "mr_nav_pretrained":
                action = author(e.obs)
            elif method == "target_heading":
                dx, dy = (e.targets - e.pos)[0].tolist()
                action = [math.atan2(dx, dy)]
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, terminal, timeout, info = task.step(action)
            if terminal or timeout:
                rows.append(dict(scene=i, method=method, seed=seed, noise_seed=2000000+i,
                                 map_sha256=task.map_hash, start=start, goal=goal, steps=step,
                                 timeout=timeout, **info))
                break
        else:
            raise RuntimeError("native simulator exceeded its timeout")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--steps", type=int, default=50176)
    p.add_argument("--scenes", type=int, default=100)
    a = p.parse_args()
    if a.steps < 256 or a.steps % 256 or not 1 <= a.scenes <= 10000:
        p.error("use a positive multiple of 256 steps and 1..10000 scenes")
    torch.set_num_threads(1)
    ckpt = a.upstream / "saved_model/VesselEnv/model_499.pt"
    a.out.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # The map families use distinct seeds. Geometry is fixed across training seeds.
        train = PlanarTask([make_map(root, j) for j in range(16)])
        author = load_author(ckpt, train.maps[0][0].device)
        model = PPO("MultiInputPolicy", train, seed=a.seed, device="cpu", n_steps=256,
                    batch_size=128, n_epochs=4, learning_rate=1e-4, gamma=.995,
                    policy_kwargs=dict(features_extractor_class=IPPOExtractor, normalize_images=False,
                                       net_arch=dict(pi=[128,128], vf=[128,128])))
        model.learn(total_timesteps=a.steps)
        model.save(a.out / "planar_model")
        # Construct one evaluation map at a time to bound GPU/CPU memory.
        results = []
        for i in range(a.scenes):
            item = make_map(root, 10000+i)
            task = PlanarTask({i: item})
            results.extend(evaluate_scene(task, model, author, i))
        data = dict(upstream_revision="c129da1447d9854e31b1d7b970b7f48b8a553ea7",
                    author_model_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                    seed=a.seed, planar_ppo_steps=model.num_timesteps, device=str(item[0].device),
                    geometry="synthetic planar bifurcations", episodes=results)
        (a.out / "paired_planar.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
