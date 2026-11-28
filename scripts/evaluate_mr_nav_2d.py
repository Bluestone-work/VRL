"""Paired single-robot evaluation on the published mr-nav planar vessel maps."""
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(args):
    sys.path.insert(0, str(args.upstream.resolve()))
    from alg.actor_critic import ActorCritic
    from alg.normalizer import EmpiricalNormalization
    from cfg.vessel_env_cfg import VesselEnvCfg, VesselEnvActorCriticCfg
    from env.vessel_env import VesselEnv

    cfg = VesselEnvCfg()
    cfg.num_envs = args.count
    weights = args.upstream / 'saved_model/VesselEnv/model_499.pt'
    net = VesselEnvActorCriticCfg()
    model = ActorCritic(cfg.num_obs, cfg.num_obs, cfg.num_actions, net.actor_hidden_dims,
                        net.critic_hidden_dims, net.activation, net.distribution_type)
    normalizer = EmpiricalNormalization([cfg.num_obs])
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    normalizer.load_state_dict(checkpoint['obs_norm_state_dict'])
    model.eval()
    normalizer.eval()
    maps = random.Random(args.seed).sample(sorted(p for p in args.dataset.iterdir() if p.is_file()), args.count)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for p in maps:
            (Path(directory) / p.name).symlink_to(p.resolve())
        cfg.sim.sim_vessel2d_dataset_path = directory
        identity = None
        for method in ('mr_nav', 'direct_pursuit'):
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            env = VesselEnv(cfg)
            model.to(env.device)
            normalizer.to(env.device)
            env.compute_observation()
            # Identify the actual map assigned by upstream's randomized loader.
            signatures = {}
            for p in maps:
                pixels = torch.load(p, map_location='cpu', weights_only=True).to(torch.uint8)
                signatures[hashlib.sha256(pixels.numpy().tobytes()).hexdigest()] = p.name
            starts = env.pos.tolist()
            targets = env.targets.tolist()
            names = [signatures[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()]
                     for x in env.rgb_imgs_tensor]
            scene_keys = list(zip(names, starts, targets))
            if identity is not None and scene_keys != identity:
                raise ValueError('Paired map, start or target mismatch')
            identity = scene_keys
            finished = torch.zeros(args.count, dtype=torch.bool, device=env.device)
            guarded_steps = torch.zeros_like(finished, dtype=torch.long)
            for step in range(1, cfg.max_episode_length + 2):
                obs = env.get_observations()
                if method == 'mr_nav':
                    raw = model.act_inference(normalizer(obs)).flatten()
                else:
                    delta = obs[:, 72:74] - obs[:, 74:76]
                    raw = torch.atan2(delta[:, 0], delta[:, 1])
                # Both methods use upstream's angular clip and a common local
                # obstacle guard. It uses only the 36 observed scan endpoints.
                requested = raw.clamp(-math.pi, math.pi)
                rays = obs[:, :72].reshape(-1, 36, 2) - obs[:, None, 74:76]
                distances = torch.linalg.vector_norm(rays, dim=-1)
                headings = torch.arange(36, device=env.device) * (2 * math.pi / 36)
                similarity = torch.cos((math.pi / 2 - headings)[None, :] - requested[:, None])
                clear = distances >= 2.
                obstructed = ~clear.gather(1, similarity.argmax(1, keepdim=True)).flatten()
                use_guard = obstructed & clear.any(1)
                nearest_clear = similarity.masked_fill(~clear, -2).argmax(1)
                alternative = math.pi / 2 - headings[nearest_clear]
                alternative = torch.atan2(torch.sin(alternative), torch.cos(alternative))
                command = torch.where(use_guard, alternative, requested)
                guarded_steps += use_guard & ~finished
                # Upstream samples fresh reset states after a termination. Reset
                # the disturbance stream per tick to isolate it from those draws.
                torch.manual_seed(args.seed + step)
                _, _, done, _ = env.step(command[:, None])
                for i in torch.where(done & ~finished)[0].tolist():
                    reached = bool(env.targets_reached[i])
                    collision = not bool(env.in_vessel[i])
                    lost = bool(env.out_of_bound_buffer[i])
                    rows.append(dict(method=method, map=names[i], lane=i, start=starts[i],
                                     target=targets[i], steps=step, elapsed_s=step * cfg.dt,
                                     reached=reached, success=reached and not collision and not lost,
                                     collision=collision, out_of_bounds=lost,
                                     timeout=step > cfg.max_episode_length,
                                     guard_steps=int(guarded_steps[i])))
                finished |= done
                if finished.all():
                    break
            if not finished.all():
                raise ValueError('Missing terminal record')
    return dict(upstream_commit=subprocess.check_output(
        ['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip(),
        license_sha256=digest(args.upstream / 'LICENSE'), checkpoint_sha256=digest(weights),
        dataset_doi='10.5281/zenodo.21719761', seed=args.seed, maps=len(maps),
        map_sha256={p.name: digest(p) for p in maps}, dt_s=cfg.dt,
        deadline_steps=cfg.max_episode_length + 1, noise_level=cfg.noise_level,
        scope='Shared 2-D vessel task. Frozen author actor and normalizer vs direct target heading.',
        episodes=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--count', type=int, default=48)
    p.add_argument('--seed', type=int, default=20261127)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        result = evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
