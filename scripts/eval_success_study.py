"""Resumable, paired anatomical evaluation with episode-level failure metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from environments.vessel_geometry import resolve_pool
from marl.geometric_control import policy_action
from marl.policy_loader import load_policy


def summarize(records):
    keys = ("success", "removal_rate", "return", "wall_hits_total",
            "wall_hits_per_step", "robot_collisions_total", "steps", "contact_miss")
    per_territory = {}
    for scenario in sorted({row["scenario"] for row in records}):
        selected = [row for row in records if row["scenario"] == scenario]
        per_territory[scenario] = {
            key: float(np.mean([row[key] for row in selected])) for key in keys
        }
    macro = {
        key: float(np.mean([row[key] for row in per_territory.values()])) for key in keys
    }
    successes = sum(row["success"] for row in records)
    count = len(records)
    proportion = successes / count
    denominator = 1 + 1.96 ** 2 / count
    center = (proportion + 1.96 ** 2 / (2 * count)) / denominator
    radius = 1.96 * np.sqrt(
        proportion * (1 - proportion) / count + 1.96 ** 2 / (4 * count ** 2)
    ) / denominator
    return {
        "episodes": count, "successes": successes, "macro": macro,
        "per_territory": per_territory,
        "success_wilson95": [float(center - radius), float(center + radius)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="")
    parser.add_argument("--controller", choices=("guided", "flow_guided", "flow_spread", "zero"), default="guided")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=700000)
    parser.add_argument("--scenario-pool", default="anatomical")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--clots", type=int, default=3)
    parser.add_argument("--robot-radius", type=float, default=0.0011)
    parser.add_argument("--guidance-speed", type=float, default=0.65)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    if args.policy:
        config["policy_sha256"] = hashlib.sha256(Path(args.policy).read_bytes()).hexdigest()
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("refusing to mix evaluation configurations")
    config_path.write_text(json.dumps(config, indent=2))
    records_path = output / "episodes.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    completed = {(row["scenario"], row["seed"]) for row in records}
    for scenario_index, scenario in enumerate(resolve_pool(args.scenario_pool)):
        env = Vascular3DMARLEnv(
            scenario=scenario, randomize_scenario=False,
            num_robots=args.robots, num_clots=args.clots,
            horizon=args.horizon, robot_radius=args.robot_radius,
        )
        policy = load_policy(args.policy, env, device=args.device) if args.policy else None
        for episode in range(args.episodes):
            episode_seed = args.seed + scenario_index * 10000 + episode
            if (scenario, episode_seed) in completed:
                continue
            obs, _ = env.reset(seed=episode_seed)
            total_return = 0.0
            wall_hits = 0
            robot_collisions = 0
            contacts = 0
            for step in range(args.horizon):
                if policy is None:
                    raw = np.zeros((args.robots, 3), np.float32)
                    action = policy_action(
                        raw, obs, env, mode="world" if args.controller == "zero" else args.controller,
                        residual_scale=0.0, guidance_speed=args.guidance_speed,
                    )
                else:
                    action = policy(obs, env)
                obs, reward, terminated, truncated, info = env.step(action)
                total_return += reward
                wall_hits += info["wall_collisions"]
                robot_collisions += info["robot_collisions"]
                contacts += info["active_contacts"]
                if terminated or truncated:
                    break
            record = {
                "scenario": scenario, "seed": episode_seed,
                "success": int(info["success"]), "removal_rate": info["removal_rate"],
                "return": total_return, "steps": step + 1,
                "wall_hits_total": wall_hits, "wall_hits_per_step": wall_hits / (step + 1),
                "robot_collisions_total": robot_collisions,
                "contact_miss": int(contacts == 0), "first_contact_step": info["first_contact_step"],
                "active_clots": info["active_clots"],
            }
            with records_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            records.append(record)
            print(json.dumps(record), flush=True)
        env.close()
    summary = summarize(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["macro"], indent=2))


if __name__ == "__main__":
    main()
