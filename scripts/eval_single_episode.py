"""Run exactly one evaluation episode so native simulator crashes are isolated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from marl.geometric_control import policy_action
from marl.policy_loader import load_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--controller", choices=("guided", "flow_guided", "flow_spread", "zero"), default="zero")
    parser.add_argument("--policy", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--clots", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--robot-radius", type=float, default=0.0011)
    args = parser.parse_args()
    torch.set_num_threads(1)
    env = Vascular3DMARLEnv(
        scenario=args.scenario, randomize_scenario=False,
        num_robots=args.robots, num_clots=args.clots, horizon=args.horizon,
        robot_radius=args.robot_radius,
    )
    policy = load_policy(args.policy, env, device="cpu") if args.policy else None
    obs, _ = env.reset(seed=args.seed)
    total_return = 0.0
    wall_hits = 0
    robot_collisions = 0
    contacts = 0
    info = {}
    for step in range(args.horizon):
        if policy:
            action = policy(obs, env)
        else:
            raw = np.zeros((args.robots, 3), np.float32)
            mode = "world" if args.controller == "zero" else args.controller
            action = policy_action(raw, obs, env, mode=mode, residual_scale=0.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += reward
        wall_hits += int(info.get("wall_collisions", 0))
        robot_collisions += int(info.get("robot_collisions", 0))
        contacts += int(info.get("active_contacts", 0))
        if terminated or truncated:
            break
    record = {
        "scenario": args.scenario, "seed": args.seed,
        "success": int(info["success"]), "removal_rate": float(info["removal_rate"]),
        "return": float(total_return), "steps": step + 1,
        "wall_hits_total": wall_hits, "wall_hits_per_step": wall_hits / (step + 1),
        "robot_collisions_total": robot_collisions, "contact_miss": int(contacts == 0),
        "first_contact_step": int(info["first_contact_step"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
