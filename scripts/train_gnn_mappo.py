"""
Training script for GNN-MAPPO on vascular navigation task.

This script implements the multi-agent reinforcement learning approach inspired by
the DGR_VDS project, using Graph Attention Networks (GAT) combined with MAPPO
for coordinated blood clot removal by microrobot swarms.

Key features:
- GAT-based policy for modeling agent interactions
- MAPPO for stable on-policy learning
- Curriculum learning for progressive difficulty
- Detailed logging and evaluation
"""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from marl.mappo_policy import MAPPO


def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN-MAPPO on vascular navigation")

    # Environment
    parser.add_argument("--scenario", type=str, default="bifurcation", help="Vessel scenario")
    parser.add_argument("--robots", type=int, default=3, help="Number of robots")
    parser.add_argument("--clots", type=int, default=3, help="Number of clots")
    parser.add_argument("--horizon", type=int, default=300, help="Episode horizon")
    parser.add_argument("--obs-mode", type=str, default="geometric", choices=["geometric", "legacy"])
    parser.add_argument("--reward-mode", type=str, default="milestone", help="Reward shaping mode")

    # Training
    parser.add_argument("--timesteps", type=int, default=500000, help="Total training timesteps")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO update epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning")

    # Algorithm
    parser.add_argument("--use-gat", action="store_true", default=True, help="Use GAT encoder")
    parser.add_argument("--no-gat", dest="use_gat", action="store_false", help="Disable GAT, use MLP")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--num-gat-layers", type=int, default=2, help="Number of GAT layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")

    # Hyperparameters
    parser.add_argument("--lr-actor", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")

    # Logging
    parser.add_argument("--log-dir", type=str, default="logdir", help="Log directory")
    parser.add_argument("--exp-name", type=str, default="gnn_mappo", help="Experiment name")
    parser.add_argument("--eval-interval", type=int, default=10000, help="Evaluation interval")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes for evaluation")
    parser.add_argument("--save-interval", type=int, default=50000, help="Model save interval")

    # System
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    return parser.parse_args()


class CurriculumScheduler:
    """Progressive curriculum for multi-clot navigation."""

    def __init__(self, start_clots: int = 1, max_clots: int = 3, success_threshold: float = 0.6):
        self.start_clots = start_clots
        self.max_clots = max_clots
        self.current_clots = start_clots
        self.success_threshold = success_threshold
        self.recent_successes = []
        self.window_size = 20

    def update(self, success: bool) -> int:
        """Update curriculum based on recent performance."""
        self.recent_successes.append(float(success))

        # Keep only recent results
        if len(self.recent_successes) > self.window_size:
            self.recent_successes.pop(0)

        # Check if we should increase difficulty
        if len(self.recent_successes) >= self.window_size:
            success_rate = np.mean(self.recent_successes)

            if success_rate >= self.success_threshold and self.current_clots < self.max_clots:
                self.current_clots += 1
                self.recent_successes = []  # Reset after curriculum change
                print(f"\n🎯 Curriculum advanced: {self.current_clots} clots (success rate: {success_rate:.2%})")

        return self.current_clots


def evaluate_policy(
    agent: MAPPO,
    env: Vascular3DMARLEnv,
    n_episodes: int = 10,
) -> Dict[str, float]:
    """Evaluate policy performance."""
    metrics = {
        'success': [],
        'episode_return': [],
        'removal_rate': [],
        'contact_miss': [],
        'episode_length': [],
    }

    for _ in range(n_episodes):
        obs_dict, info = env.reset()
        episode_return = 0
        done = False

        while not done:
            # Extract observations
            obs = obs_dict['nodes']

            # Get agent positions for adjacency matrix
            positions = env.robot_positions.copy() if hasattr(env, 'robot_positions') else None

            # Select action (deterministic during evaluation)
            actions, _, _, _ = agent.select_action(obs, positions, deterministic=True)

            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            episode_return += reward

        # Record metrics
        metrics['success'].append(float(info.get('success', False)))
        metrics['episode_return'].append(episode_return)
        metrics['removal_rate'].append(info.get('removal_rate', 0.0))
        metrics['contact_miss'].append(float(info.get('contact_miss', False)))
        metrics['episode_length'].append(info.get('episode_length', 0))

    # Average metrics
    return {k: np.mean(v) for k, v in metrics.items()}


def train():
    args = parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create log directory
    log_dir = Path(args.log_dir) / args.exp_name / f"seed_{args.seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(log_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print(f"GNN-MAPPO Training on Vascular Navigation")
    print("=" * 80)
    print(f"Environment: {args.robots} robots, {args.clots} clots, {args.scenario} scenario")
    print(f"Algorithm: MAPPO with {'GAT' if args.use_gat else 'MLP'}")
    print(f"Device: {args.device}")
    print(f"Log directory: {log_dir}")
    print("=" * 80)

    # Create environment
    env = Vascular3DMARLEnv(
        scenario=args.scenario,
        num_robots=args.robots,
        num_clots=args.clots if not args.curriculum else 1,
        horizon=args.horizon,
        seed=args.seed,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        curriculum=False,  # We handle curriculum manually
    )

    # Get dimensions
    # Observation space is a Dict with 'nodes', 'adjacency', 'clot_state'
    sample_obs = env.observation_space.sample()
    obs_dim = sample_obs['nodes'].shape[1]  # Feature dimension per robot
    action_dim = env.action_space.shape[1]  # Action dimension per robot
    n_agents = args.robots

    # State dimension: use clot_state from observation
    state_dim = sample_obs['clot_state'].size  # Flatten entire clot state

    print(f"Observation dim: {obs_dim} (per robot)")
    print(f"Action dim: {action_dim} (per robot)")
    print(f"State dim: {state_dim} (global clot state)")
    print("=" * 80)

    # Create agent
    agent = MAPPO(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_gat_layers,
        num_heads=args.num_heads,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm,
        use_gat=args.use_gat,
        device=args.device,
    )

    # Curriculum scheduler
    curriculum = None
    if args.curriculum:
        curriculum = CurriculumScheduler(start_clots=1, max_clots=args.clots)
        print(f"Curriculum learning enabled: starting with {curriculum.current_clots} clots")

    # Training loop
    total_steps = 0
    episode_count = 0
    best_success_rate = 0.0

    # Metrics tracking
    episode_metrics = []

    start_time = time.time()

    obs_dict, info = env.reset()
    episode_return = 0
    episode_start_step = 0

    print("\nStarting training...")
    print("=" * 80)

    while total_steps < args.timesteps:
        # Collect rollout
        for step in range(args.n_steps):
            # Extract components from observation dict
            obs = obs_dict['nodes']  # [num_robots, obs_dim]
            adj_matrix_env = obs_dict['adjacency']  # [num_robots, num_robots]
            clot_state = obs_dict['clot_state']  # [max_clots, 6]

            # Get agent positions for adjacency matrix (use environment's adjacency)
            positions = env.robot_positions.copy() if hasattr(env, 'robot_positions') else None

            # Get global state for critic - flatten clot state
            state = clot_state.flatten()

            # Select action - pass state to critic
            actions, log_probs, values, adj_matrix = agent.select_action(
                obs, positions, state, deterministic=False
            )

            # Step environment
            next_obs_dict, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # Get per-agent rewards from info (MADDPG needs individual rewards)
            agent_rewards = info.get('agent_rewards', np.full(n_agents, reward / n_agents))

            # Store transition
            agent.buffer.store(
                obs=obs,
                actions=actions,
                rewards=agent_rewards,  # Use per-agent rewards
                dones=np.full(n_agents, done, dtype=np.float32),  # Broadcast scalar done to all agents
                log_probs=log_probs,
                values=values,
                adj_matrix=adj_matrix,
                state=state,
            )

            # Update state
            obs_dict = next_obs_dict
            episode_return += reward  # Total team reward for logging
            total_steps += 1

            # Episode end
            if done:
                episode_count += 1
                episode_length = total_steps - episode_start_step

                # Record episode metrics
                episode_data = {
                    'episode': episode_count,
                    'steps': total_steps,
                    'return': episode_return,
                    'length': episode_length,
                    'success': float(info.get('success', False)),
                    'removal_rate': info.get('removal_rate', 0.0),
                    'contact_miss': float(info.get('contact_miss', False)),
                    'n_clots': env.num_clots,
                }
                episode_metrics.append(episode_data)

                # Update curriculum
                if curriculum is not None:
                    new_clots = curriculum.update(episode_data['success'])
                    if new_clots != env.num_clots:
                        env.num_clots = new_clots

                # Print progress
                if episode_count % 10 == 0:
                    recent = episode_metrics[-10:]
                    avg_return = np.mean([e['return'] for e in recent])
                    avg_success = np.mean([e['success'] for e in recent])
                    elapsed = time.time() - start_time
                    fps = total_steps / elapsed

                    print(f"Episode {episode_count:5d} | Steps {total_steps:7d} | "
                          f"Return {avg_return:7.1f} | Success {avg_success:.2%} | "
                          f"FPS {fps:.0f}")

                # Reset environment
                obs_dict, info = env.reset()
                episode_return = 0
                episode_start_step = total_steps

            # Evaluation
            if total_steps % args.eval_interval == 0:
                print("\n" + "=" * 80)
                print(f"Evaluation at step {total_steps}")
                eval_metrics = evaluate_policy(agent, env, args.eval_episodes)

                print(f"Success rate: {eval_metrics['success']:.2%}")
                print(f"Average return: {eval_metrics['episode_return']:.1f}")
                print(f"Removal rate: {eval_metrics['removal_rate']:.2%}")
                print(f"Contact miss: {eval_metrics['contact_miss']:.2%}")
                print("=" * 80 + "\n")

                # Save best model
                if eval_metrics['success'] > best_success_rate:
                    best_success_rate = eval_metrics['success']
                    agent.save(log_dir / "best_policy.pt")
                    print(f"✓ New best model saved (success rate: {best_success_rate:.2%})")

                # Save evaluation metrics
                with open(log_dir / "eval_metrics.json", "a") as f:
                    eval_data = {
                        'step': total_steps,
                        **eval_metrics
                    }
                    f.write(json.dumps(eval_data) + "\n")

            # Save checkpoint
            if total_steps % args.save_interval == 0:
                agent.save(log_dir / f"checkpoint_{total_steps}.pt")
                print(f"✓ Checkpoint saved at step {total_steps}")

        # Update policy
        if len(agent.buffer) > 0:
            update_metrics = agent.update(
                n_epochs=args.n_epochs,
                batch_size=args.batch_size,
            )

            # Log update metrics
            if total_steps % (args.n_steps * 5) == 0:
                print(f"Update | Actor loss: {update_metrics['actor_loss']:.4f} | "
                      f"Critic loss: {update_metrics['critic_loss']:.4f} | "
                      f"Entropy: {update_metrics['entropy']:.4f}")

    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Evaluation")
    final_metrics = evaluate_policy(agent, env, args.eval_episodes * 2)

    print(f"Success rate: {final_metrics['success']:.2%}")
    print(f"Average return: {final_metrics['episode_return']:.1f}")
    print(f"Removal rate: {final_metrics['removal_rate']:.2%}")
    print(f"Contact miss: {final_metrics['contact_miss']:.2%}")
    print("=" * 80)

    # Save final model
    agent.save(log_dir / "final_policy.pt")

    # Save all episode metrics
    with open(log_dir / "episode_metrics.json", "w") as f:
        json.dump(episode_metrics, f, indent=2)

    # Save summary
    summary = {
        'total_episodes': episode_count,
        'total_steps': total_steps,
        'best_success_rate': best_success_rate,
        'final_metrics': final_metrics,
        'training_time': time.time() - start_time,
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete! Results saved to {log_dir}")
    print(f"Training time: {summary['training_time']:.1f} seconds")


if __name__ == "__main__":
    train()
