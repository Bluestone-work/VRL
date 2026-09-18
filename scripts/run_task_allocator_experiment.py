"""Evaluate explicit task allocation on top of the existing flow controller."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from environments.vessel_geometry import ALL_SCENARIOS
from marl.geometric_control import policy_action
from marl.hierarchical_allocator import (
    HierarchicalAllocator,
    allocator_features,
    sample_allocation,
)
from marl.task_allocator import balanced_assignment
from marl.policy_loader import load_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("nearest_flow", "hungarian_flow", "learned_allocator"),
        required=True,
    )
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--clots", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--allocation-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=910000)
    parser.add_argument("--output-root", default="experiments/hierarchical_marl_research_20260906/evaluation")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initialization-mode", default="legacy",
                        choices=("legacy", "stratified", "random"))
    parser.add_argument("--low-level-policy", default="")
    return parser.parse_args()


def scenario_list(requested: str) -> tuple[str, ...]:
    if requested == "all":
        return tuple(ALL_SCENARIOS)
    if requested not in ALL_SCENARIOS:
        raise ValueError(f"unknown scenario {requested!r}")
    return (requested,)


def load_allocator(args, scenario: str):
    if args.method != "learned_allocator":
        return None, None
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for learned_allocator")
    probe = Vascular3DMARLEnv(
        scenario=scenario,
        num_robots=args.robots,
        num_clots=args.clots,
        horizon=args.horizon,
        seed=args.seed,
        randomize_scenario=False,
        randomize_clots=True,
        robot_radius=0.0011,
        reward_mode="milestone",
        obs_mode="geometric",
        initialization_mode=args.initialization_mode,
    )
    probe.reset(seed=args.seed)
    feature_probe, _ = allocator_features(
        probe, np.full((args.robots,), -1, dtype=np.int32)
    )
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = checkpoint["meta"]
    model = HierarchicalAllocator(
        feature_probe.shape[-1], meta["slots"], meta["hidden_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    probe.close()
    return model, device


def evaluate_episode(args, scenario: str, episode_seed: int, model=None, device=None,
                     low_level_policy=None) -> dict:
    env = Vascular3DMARLEnv(
        scenario=scenario,
        num_robots=args.robots,
        num_clots=args.clots,
        horizon=args.horizon,
        seed=episode_seed,
        randomize_scenario=False,
        randomize_clots=True,
        robot_radius=0.0011,
        reward_mode="milestone",
        obs_mode="geometric",
        initialization_mode=args.initialization_mode,
    )
    obs, reset_info = env.reset(seed=episode_seed)
    assignments = np.full((args.robots,), -1, dtype=np.int32)
    target_load = np.zeros((0,), dtype=np.int32)
    task_switches = 0
    previous_assignments = assignments.copy()
    rows = []
    total_reward = 0.0
    total_wall_hits = 0
    total_robot_collisions = 0
    steps = 0

    while steps < args.horizon:
        if args.method == "hungarian_flow" and (
            steps == 0 or steps % args.allocation_interval == 0
        ):
            allocation = balanced_assignment(env)
            assignments = allocation.assignments.copy()
            env.set_task_assignments(assignments)
            target_load = allocation.target_load.copy()
        elif args.method == "nearest_flow":
            env.clear_task_assignments()
            assignments = env._assigned_clot().copy()
            target_load = np.bincount(
                assignments[assignments >= 0], minlength=max(env.active_clots, 0)
            ).astype(np.int32)
        elif args.method == "learned_allocator" and (
            steps == 0 or steps % args.allocation_interval == 0
        ):
            features, valid = allocator_features(env, assignments)
            action = sample_allocation(
                model, features, valid, device, deterministic=True
            )
            assignments = action.assignments.copy()
            env.set_task_assignments(assignments)
            target_load = np.bincount(
                assignments[assignments >= 0], minlength=max(env.active_clots, 0)
            ).astype(np.int32)

        task_switches += int(np.count_nonzero(assignments != previous_assignments))
        previous_assignments = assignments.copy()
        obs = env._build_observation()
        raw = np.zeros((args.robots, 3), dtype=np.float32)
        if low_level_policy is None:
            action = policy_action(raw, obs, env, mode="flow_guided", residual_scale=0.0)
        else:
            action = low_level_policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_wall_hits += int(info["wall_collisions"])
        total_robot_collisions += int(info["robot_collisions"])
        steps += 1
        rows.append({
            "step": steps,
            "reward": float(reward),
            "team_reward": float(info["team_reward"]),
            "agent_reward_mean": float(np.mean(info["agent_rewards"])),
            "removed_mass": float(info["removed_mass"]),
            "remaining_mass": float(info["remaining_mass"]),
            "removal_rate": float(info["removal_rate"]),
            "active_clots": int(info["active_clots"]),
            "clots_engaged": int(info["clots_engaged"]),
            "wall_collisions": int(info["wall_collisions"]),
            "robot_collisions": int(info["robot_collisions"]),
            "assignments": assignments.tolist(),
            "target_load": target_load.tolist(),
            "distinct_targets": int(np.count_nonzero(target_load)),
        })
        if terminated or truncated:
            break

    env.close()
    return {
        "scenario": scenario,
        "seed": episode_seed,
        "method": args.method,
        "success": bool(rows[-1]["remaining_mass"] <= 1e-8),
        "steps": steps,
        "return": total_reward,
        "removal_rate": rows[-1]["removal_rate"] if rows else 0.0,
        "wall_hits": total_wall_hits,
        "robot_collisions": total_robot_collisions,
        "task_switches": task_switches,
        "initial_difficulty": float(reset_info.get("difficulty", 1.0)),
        "trace": rows,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.output_root) / args.method
    root.mkdir(parents=True, exist_ok=True)
    model, device = load_allocator(args, scenario_list(args.scenario)[0])
    low_level_policy = None
    if args.low_level_policy:
        probe = Vascular3DMARLEnv(
            scenario=scenario_list(args.scenario)[0], num_robots=args.robots,
            num_clots=args.clots, horizon=args.horizon, seed=args.seed,
            randomize_scenario=False, randomize_clots=True, robot_radius=0.0011,
            reward_mode="milestone", obs_mode="geometric",
            initialization_mode=args.initialization_mode,
        )
        probe.reset(seed=args.seed)
        low_level_policy = load_policy(
            args.low_level_policy, probe, device=args.device,
        )
        probe.close()
    episodes = []
    for scenario_index, scenario in enumerate(scenario_list(args.scenario)):
        for episode in range(args.episodes):
            seed = args.seed + scenario_index * 1000 + episode
            result = evaluate_episode(
                args, scenario, seed, model, device, low_level_policy
            )
            episodes.append(result)
            path = root / f"{scenario}_{seed}.json"
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps({key: result[key] for key in (
                "scenario", "seed", "success", "steps", "removal_rate",
                "task_switches", "wall_hits", "robot_collisions",
            )}, ensure_ascii=False))

    summary = {
        "method": args.method,
        "scenarios": len(scenario_list(args.scenario)),
        "episodes": len(episodes),
        "success_rate": float(np.mean([row["success"] for row in episodes])) if episodes else 0.0,
        "mean_steps": float(np.mean([row["steps"] for row in episodes])) if episodes else 0.0,
        "mean_removal_rate": float(np.mean([row["removal_rate"] for row in episodes])) if episodes else 0.0,
        "mean_task_switches": float(np.mean([row["task_switches"] for row in episodes])) if episodes else 0.0,
        "mean_wall_hits": float(np.mean([row["wall_hits"] for row in episodes])) if episodes else 0.0,
        "mean_robot_collisions": float(np.mean([row["robot_collisions"] for row in episodes])) if episodes else 0.0,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
