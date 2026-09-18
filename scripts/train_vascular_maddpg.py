"""Train MADDPG on vascular thrombolysis task.

MADDPG (Lowe et al. 2017) uses:
- Decentralized actors (local obs only)
- Centralized critics (observe all agents)
- Continuous actions (no discretization)
- Replay buffer for off-policy learning
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from collections import deque

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from marl.maddpg_policy import MADDPG


def train(args):
    print(f"Training MADDPG on vascular thrombolysis")
    print(f"  robots={args.robots}, clots={args.clots}, timesteps={args.timesteps}")
    print(f"  device={args.device}, seed={args.seed}")

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Environment
    env = Vascular3DMARLEnv(
        scenario=args.scenario,
        num_robots=args.robots,
        num_clots=args.clots,
        horizon=args.horizon,
        seed=args.seed,
        render_mode=None,
        randomize_scenario=args.randomize_scenario,
        randomize_clots=args.randomize_clots,
        use_pybullet=False,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        curriculum=args.curriculum,
    )

    obs_dict, info = env.reset(seed=args.seed)
    obs_dim = obs_dict["nodes"].shape[1]
    action_dim = 3
    # Global state for the centralized critic: the flattened clot state. The
    # environment already computes it and the old trainer discarded it, so the
    # critic could not see overall task progress. Legitimate under CTDE.
    state_dim = 0 if args.no_critic_state else int(obs_dict["clot_state"].size)

    # MADDPG agent
    maddpg = MADDPG(
        n_agents=args.robots,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=device,
        state_dim=state_dim,
        share_parameters=not args.no_share_parameters,
        seed=args.seed,
    )

    # Logging
    suffix = f"_{args.tag}" if args.tag else ""
    logdir = Path(args.logdir) / f"maddpg_{args.scenario}_r{args.robots}_c{args.clots}{suffix}"
    run_number = 1
    while (logdir / f"run_{run_number}").exists():
        run_number += 1
    run_dir = logdir / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  logdir: {run_dir}")

    config = vars(args).copy()
    config.update({
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "state_dim": state_dim,
    })
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    metrics_path = run_dir / "metrics.jsonl"
    metrics_file = metrics_path.open("a")

    # Training loop
    obs_dict, info = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    episode_count = 0
    best_return = -np.inf
    start_time = time.time()

    # Rolling windows for the curriculum controller and for reporting the metric
    # that actually diagnosed the original failure: how often the swarm never
    # touches a clot at all.
    recent_success: deque[float] = deque(maxlen=args.curriculum_window)
    recent_contact_miss: deque[float] = deque(maxlen=args.curriculum_window)
    difficulty = 0.0 if args.curriculum else 1.0
    if args.curriculum:
        env.set_difficulty(difficulty)
        obs_dict, info = env.reset(seed=args.seed)

    # Exploration schedule: epsilon decay from 1.0 to 0.05 over 100k steps
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = args.epsilon_decay_steps

    def critic_state(d: dict) -> np.ndarray | None:
        return None if state_dim == 0 else d["clot_state"].reshape(-1)

    loss_info: dict = {}
    for timestep in range(1, args.timesteps + 1):
        # Epsilon for exploration
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * timestep / epsilon_decay_steps)

        # Select actions.
        # `scale_noise` fixes a real bug: epsilon used to be an on/off gate only,
        # so OU noise ran at full sigma for 100k steps and then fell off a cliff
        # to exactly zero. With scaling on, epsilon actually anneals exploration.
        # Noise never reaches zero: a deterministic policy driving an off-policy
        # replay buffer stops generating new experience entirely.
        obs = obs_dict["nodes"]  # [n_agents, obs_dim]
        actions = maddpg.select_actions(
            obs,
            add_noise=True,
            noise_scale=max(epsilon, epsilon_end) if args.scale_noise else 1.0,
        )

        # Step environment
        obs_dict_next, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # Store the transition with PER-AGENT rewards. The environment computes
        # the decomposition in info["agent_rewards"]; storing only the team scalar
        # made every critic regress an identical target.
        agent_rewards = np.asarray(info["agent_rewards"], dtype=np.float32)
        # Team terms are shared credit and belong in every agent's target.
        shared = float(info["team_reward"]) / max(args.robots, 1)
        per_agent = agent_rewards + shared

        maddpg.replay_buffer.store(
            obs=obs,
            actions=actions,
            rewards=per_agent,
            next_obs=obs_dict_next["nodes"],
            # Bootstrapping must continue through a time-limit truncation: the
            # episode ended for bookkeeping reasons, not because the state is
            # terminal. Treating truncation as terminal teaches the value function
            # that the world stops at the horizon.
            done=terminated,
            state=critic_state(obs_dict),
            next_state=critic_state(obs_dict_next),
        )

        episode_return += reward
        episode_length += 1

        # Update networks. A gradient step costs far more than an environment
        # step, so `--update-freq` sets how many env steps pass between updates:
        # the default of 1 makes the optimiser, not the simulator, the throughput
        # ceiling. Raising it collects more experience per gradient step.
        if (
            len(maddpg.replay_buffer) >= max(args.batch_size, args.learning_starts)
            and timestep % args.update_freq == 0
        ):
            for _ in range(args.updates_per_step):
                loss_info = maddpg.update()

            if timestep % (args.log_interval * 100) == 0 and loss_info:
                print(f"    [update @ {timestep}] actor_loss={loss_info['actor_loss']:.3f} "
                      f"critic_loss={loss_info['critic_loss']:.3f} "
                      f"q={loss_info.get('mean_q', 0.0):.3f} eps={epsilon:.3f}")

        if done:
            episode_count += 1
            if episode_return > best_return:
                best_return = episode_return
                maddpg.save(run_dir / "best_policy.pt")

            contact_miss = info.get("first_contact_step", -1) < 0
            recent_success.append(1.0 if info.get("success", False) else 0.0)
            recent_contact_miss.append(1.0 if contact_miss else 0.0)

            log_entry = {
                "timestep": timestep,
                "episode": episode_count,
                "return": round(episode_return, 3),
                "length": episode_length,
                "success": bool(info.get("success", False)),
                "removal_rate": round(info.get("removal_rate", 0.0), 3),
                "robot_collisions": info.get("robot_collisions", 0),
                "wall_collisions": info.get("wall_collisions", 0),
                "clots_engaged": info.get("clots_engaged", 0),
                "first_contact": info.get("first_contact_step", -1),
                "contact_miss": bool(contact_miss),
                "scenario": info.get("scenario", ""),
                "difficulty": round(difficulty, 3),
            }
            metrics_file.write(json.dumps(log_entry) + "\n")
            metrics_file.flush()

            # Curriculum: widen once the recent success rate clears the gate. The
            # diagnosed bottleneck was exploration -- a swarm that never reaches a
            # clot never observes the lysis reward, so there is nothing to learn
            # from. Starting near and widening gives the critic a foothold.
            if args.curriculum and len(recent_success) == recent_success.maxlen:
                rate = float(np.mean(recent_success))
                if rate >= args.curriculum_promote and difficulty < 1.0:
                    difficulty = min(1.0, difficulty + args.curriculum_step)
                    env.set_difficulty(difficulty)
                    recent_success.clear()
                    print(f"  [curriculum] success {rate:.2f} -> difficulty {difficulty:.2f}")
                elif rate <= args.curriculum_demote and difficulty > 0.0:
                    difficulty = max(0.0, difficulty - args.curriculum_step)
                    env.set_difficulty(difficulty)
                    recent_success.clear()
                    print(f"  [curriculum] success {rate:.2f} -> difficulty {difficulty:.2f}")

            if episode_count % args.log_interval == 0:
                elapsed = time.time() - start_time
                fps = timestep / elapsed
                miss = float(np.mean(recent_contact_miss)) if recent_contact_miss else 0.0
                print(f"ep {episode_count:4d} | t {timestep:7d} | ret {episode_return:+7.2f} | "
                      f"len {episode_length:3d} | succ {int(info.get('success', False))} | "
                      f"rem {info.get('removal_rate', 0.0):.2f} | miss {miss:.2f} | "
                      f"diff {difficulty:.2f} | eps {epsilon:.3f} | fps {fps:.0f}")

            # Reset
            obs_dict, info = env.reset()
            maddpg.reset_noise()
            episode_return = 0.0
            episode_length = 0
        else:
            obs_dict = obs_dict_next

    # Save final model
    maddpg.save(run_dir / "final_policy.pt")
    metrics_file.close()
    env.close()
    print(f"\nTraining complete. Best return: {best_return:.2f}")
    print(f"Logs saved to {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="bifurcation", choices=[
        "straight", "bifurcation", "anastomosis", "stenotic"
    ])
    parser.add_argument("--robots", type=int, default=12)
    parser.add_argument("--clots", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--update-freq", type=int, default=1,
                        help="env steps between gradient updates; >1 trades sample "
                             "efficiency for wall-clock throughput")
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=0,
                        help="collect this many transitions before the first update")
    parser.add_argument("--randomize-scenario", action="store_true", default=True)
    parser.add_argument("--randomize-clots", action="store_true", default=True)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--reward-mode", default="milestone",
                        choices=["baseline", "milestone"])
    parser.add_argument("--obs-mode", default="geometric", choices=["geometric", "legacy"],
                        help="geometric adds vessel geometry to the observation; "
                             "legacy reproduces the original 20-D layout")
    parser.add_argument("--scale-noise", action="store_true", default=False,
                        help="anneal OU noise magnitude by epsilon instead of on/off gating")
    parser.add_argument("--epsilon-decay-steps", type=int, default=100000)
    parser.add_argument("--no-share-parameters", action="store_true", default=False,
                        help="give every agent its own actor/critic instead of sharing "
                             "one set across the homogeneous swarm")
    parser.add_argument("--no-critic-state", action="store_true", default=False,
                        help="withhold the global clot state from the centralized critic")
    parser.add_argument("--curriculum", action="store_true", default=False,
                        help="start with one proximal clot and widen as success rises")
    parser.add_argument("--curriculum-window", type=int, default=50)
    parser.add_argument("--curriculum-promote", type=float, default=0.5)
    parser.add_argument("--curriculum-demote", type=float, default=0.1)
    parser.add_argument("--curriculum-step", type=float, default=0.2)
    parser.add_argument("--tag", default="", help="suffix for the logdir, so ablations stay distinguishable")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", default="logdir")

    args = parser.parse_args()
    train(args)
