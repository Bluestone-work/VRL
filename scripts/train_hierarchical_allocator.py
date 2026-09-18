"""Train a high-level PPO clot allocator over the existing low-level controller."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import optim

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from environments.vessel_geometry import resolve_pool
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
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--clots", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--allocation-interval", type=int, default=20)
    parser.add_argument("--initialization-mode", default="legacy",
                        choices=("legacy", "stratified", "random"))
    parser.add_argument("--low-level-policy", default="")
    parser.add_argument("--scenario-pool", default="anatomical")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--imitation-coef", type=float, default=0.5)
    parser.add_argument("--allocation-bonus", type=float, default=0.5)
    return parser.parse_args()


def make_env(scenario: str, args, seed: int) -> Vascular3DMARLEnv:
    return Vascular3DMARLEnv(
        scenario=scenario,
        num_robots=args.robots,
        num_clots=args.clots,
        horizon=args.horizon,
        seed=seed,
        randomize_scenario=False,
        randomize_clots=True,
        robot_radius=0.0011,
        reward_mode="milestone",
        obs_mode="geometric",
        initialization_mode=args.initialization_mode,
    )


def rollout_episode(model, args, scenario: str, seed: int, device: torch.device,
                    low_level_policy=None):
    env = make_env(scenario, args, seed)
    env.reset(seed=seed)
    assignments = np.full((args.robots,), -1, dtype=np.int32)
    transitions = []
    total_reward = 0.0
    steps = 0
    while steps < args.horizon:
        features, valid = allocator_features(env, assignments)
        expert = balanced_assignment(env).assignments
        action = sample_allocation(model, features, valid, device)
        env.set_task_assignments(action.assignments)
        segment_reward = 0.0
        selected_load = np.bincount(
            action.assignments[action.assignments >= 0],
            minlength=max(env.active_clots, 1),
        )
        distinct_targets = int(np.count_nonzero(selected_load))
        duplicate_targets = int(np.maximum(selected_load - 1, 0).sum())
        segment_reward += args.allocation_bonus * (
            distinct_targets - 0.4 * duplicate_targets
        )
        done = False
        for _ in range(args.allocation_interval):
            obs = env._build_observation()
            raw = np.zeros((args.robots, 3), dtype=np.float32)
            if low_level_policy is None:
                low_action = policy_action(raw, obs, env, mode="flow_guided", residual_scale=0.0)
            else:
                low_action = low_level_policy(obs, env)
            _, reward, terminated, truncated, info = env.step(low_action)
            segment_reward += float(reward)
            total_reward += float(reward)
            steps += 1
            if terminated or truncated or steps >= args.horizon:
                done = True
                break
        transitions.append({
            "features": features,
            "valid": valid,
            "actions": action.assignments.copy(),
            "sampled_actions": np.where(action.assignments >= 0, action.assignments, model.slots),
            "expert_actions": np.where(expert >= 0, expert, model.slots),
            "log_probability": action.log_probability,
            "value": action.value,
            "reward": segment_reward,
            "done": done,
        })
        if done:
            break
        assignments = action.assignments.copy()
    success = bool(env.active_clots == 0 or np.all(env.clot_masses <= 0))
    env.close()
    return transitions, total_reward, success, steps


def compute_gae(episodes, gamma: float, gae_lambda: float):
    features, valid, actions, experts, old_logp, returns, advantages = [], [], [], [], [], [], []
    for transitions in episodes:
        gae = 0.0
        next_value = 0.0
        episode_advantages = [0.0] * len(transitions)
        episode_returns = [0.0] * len(transitions)
        for index in reversed(range(len(transitions))):
            item = transitions[index]
            mask = 0.0 if item["done"] else 1.0
            delta = item["reward"] + gamma * next_value * mask - item["value"]
            gae = delta + gamma * gae_lambda * mask * gae
            episode_advantages[index] = gae
            episode_returns[index] = gae + item["value"]
            next_value = item["value"]
        for item, advantage, return_value in zip(transitions, episode_advantages, episode_returns):
            features.append(item["features"])
            valid.append(item["valid"])
            actions.append(item["sampled_actions"])
            experts.append(item["expert_actions"])
            old_logp.append(item["log_probability"])
            returns.append(return_value)
            advantages.append(advantage)
    advantages = np.asarray(advantages, dtype=np.float32)
    advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-6)
    return (
        features,
        valid,
        np.asarray(actions, dtype=np.int64),
        np.asarray(experts, dtype=np.int64),
        np.asarray(old_logp, dtype=np.float32),
        np.asarray(returns, dtype=np.float32),
        advantages,
    )


def update(model, optimizer, batch, args, device):
    features, valid, actions, experts, old_logp, returns, advantages = batch
    feature_tensor = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    valid_tensor = torch.as_tensor(np.asarray(valid), dtype=torch.bool, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
    expert_tensor = torch.as_tensor(experts, dtype=torch.long, device=device)
    old_logp_tensor = torch.as_tensor(old_logp, dtype=torch.float32, device=device)
    return_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
    advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    metrics = {}
    for _ in range(args.epochs):
        distribution, value = model.distribution(feature_tensor, valid_tensor)
        logp = distribution.log_prob(action_tensor).sum(dim=-1)
        expert_logp = distribution.log_prob(expert_tensor).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1).mean()
        ratio = torch.exp(logp - old_logp_tensor)
        clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
        actor_loss = -torch.minimum(ratio * advantage_tensor, clipped * advantage_tensor).mean()
        critic_loss = torch.nn.functional.mse_loss(value, return_tensor)
        imitation_loss = -expert_logp.mean()
        loss = (
            actor_loss
            + args.value_coef * critic_loss
            - args.entropy_coef * entropy
            + args.imitation_coef * imitation_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        metrics = {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
            "imitation_loss": float(imitation_loss.item()),
            "approx_kl": float((old_logp_tensor - logp).mean().item()),
        }
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    scenarios = tuple(resolve_pool(args.scenario_pool))
    probe = make_env(scenarios[0], args, args.seed)
    probe.reset(seed=args.seed)
    features, _ = allocator_features(probe, np.full((args.robots,), -1, dtype=np.int32))
    model = HierarchicalAllocator(features.shape[-1], args.clots + 1, args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    low_level_policy = None
    if args.low_level_policy:
        low_level_policy = load_policy(args.low_level_policy, probe, device=args.device)
    probe.close()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    log_path = run_dir / "training_log.jsonl"
    best_score = -float("inf")
    for update_index in range(args.updates):
        episodes = []
        returns = []
        successes = []
        steps = []
        for episode in range(args.episodes_per_update):
            scenario = scenarios[(update_index * args.episodes_per_update + episode) % len(scenarios)]
            seed = args.seed + update_index * args.episodes_per_update + episode
            transition, episode_return, success, episode_steps = rollout_episode(
                model, args, scenario, seed, device, low_level_policy
            )
            episodes.append(transition)
            returns.append(episode_return)
            successes.append(float(success))
            steps.append(episode_steps)
        batch = compute_gae(episodes, args.gamma, args.gae_lambda)
        metrics = update(model, optimizer, batch, args, device)
        record = {
            "update": update_index + 1,
            "episodes": len(episodes),
            "mean_return": float(np.mean(returns)),
            "success_rate": float(np.mean(successes)),
            "mean_steps": float(np.mean(steps)),
            **metrics,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        score = record["success_rate"] + 0.001 * record["mean_return"]
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "meta": {
                "feature_dim": features.shape[-1],
                "slots": args.clots + 1,
                "hidden_dim": args.hidden_dim,
                "allocation_interval": args.allocation_interval,
            },
        }
        torch.save(state, run_dir / "last_allocator.pt")
        if score > best_score:
            best_score = score
            torch.save(state, run_dir / "best_allocator.pt")
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
