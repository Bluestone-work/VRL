"""
Advanced training script with pluggable architectures and progressive training.

Usage examples:
  # Edge-feature GAT
  python scripts/train_advanced.py --architecture edge_gat --robots 5 --clots 3

  # Full Transformer
  python scripts/train_advanced.py --architecture transformer --robots 8 --clots 5

  # Sparse Transformer (k=8 neighbors)
  python scripts/train_advanced.py --architecture sparse_transformer --k-neighbors 8

  # Hierarchical GNN
  python scripts/train_advanced.py --architecture hierarchical_gnn

  # With adaptive curriculum
  python scripts/train_advanced.py --adaptive-curriculum

  # With pre-training
  python scripts/train_advanced.py --pretrain

  # With exploration bonus
  python scripts/train_advanced.py --exploration-bonus 0.01
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from marl.gat_policy import build_adjacency_matrix
from marl.gnn_advanced import compute_edge_features
from marl.mappo_advanced import MAPPOAdvanced, ContextRolloutBuffer, ARCHITECTURES
from marl.progressive_training import (
    AdaptiveCurriculum,
    PretrainingTasks,
    ExplorationBonus,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # Architecture
    parser.add_argument("--architecture", type=str, default="gat",
                        choices=ARCHITECTURES, help="Network architecture")
    parser.add_argument("--k-neighbors", type=int, default=8,
                        help="k for sparse transformer")

    # Environment
    parser.add_argument("--scenario", default="bifurcation",
                        help="topology used when --fixed-scenario is set, and the "
                             "first-reset fallback otherwise")
    parser.add_argument("--scenario-pool", default="legacy",
                        help="pool the per-reset randomisation draws from: "
                             "legacy, generated, anatomical, arterial, venous, all")
    parser.add_argument("--fixed-scenario", action="store_true",
                        help="train on --scenario alone instead of randomising")
    parser.add_argument("--robot-radius", type=float, default=0.0045,
                        help="device radius; 0.0011 is needed for the anatomical "
                             "territories' stenoses to survive the lumen floor")
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--clots", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--obs-mode", type=str, default="geometric")
    parser.add_argument("--reward-mode", choices=("baseline", "milestone"),
                        default="milestone")

    # Training
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)

    # Progressive training
    parser.add_argument("--adaptive-curriculum", action="store_true")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--exploration-bonus", type=float, default=0.0)

    # Network
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)

    # Hyperparameters
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-dir", type=str, default="logdir")
    parser.add_argument("--eval-interval", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--save-interval", type=int, default=50000)
    parser.add_argument("--resume", type=str, default="",
                        help="resume networks, optimizers, RNG and simulator state")

    return parser.parse_args()


def build_context(env, obs_dict) -> dict:
    """Extract graph context from environment and observations."""
    ctx = {}

    # Positions (always available)
    if hasattr(env, 'robot_positions'):
        ctx['positions'] = env.robot_positions.copy()

    # Velocities
    if hasattr(env, 'robot_velocities'):
        ctx['velocities'] = env.robot_velocities.copy()

    # Adjacency from observation dict (environment already computed it)
    if 'adjacency' in obs_dict:
        ctx['adjacency'] = obs_dict['adjacency']

    return ctx


def evaluate(agent, env, n_episodes=10, seed_base=None):
    """Evaluate policy performance."""
    metrics = {
        'success': [],
        'episode_return': [],
        'removal_rate': [],
        'contact_miss': [],
    }

    for episode in range(n_episodes):
        seed = None if seed_base is None else seed_base + episode
        obs_dict, info = env.reset(seed=seed)
        obs = obs_dict['nodes']
        episode_return = 0
        done = False

        while not done:
            ctx = build_context(env, obs_dict)
            state = obs_dict['clot_state'].flatten()

            actions, _, _ = agent.act(obs, ctx, state, deterministic=True)

            obs_dict, reward, terminated, truncated, info = env.step(actions)
            obs = obs_dict['nodes']
            done = terminated or truncated
            episode_return += reward

        metrics['success'].append(float(info.get('success', False)))
        metrics['episode_return'].append(episode_return)
        metrics['removal_rate'].append(info.get('removal_rate', 0.0))
        metrics['contact_miss'].append(float(info.get('contact_miss', False)))

    return {k: np.mean(v) for k, v in metrics.items()}


def capture_training_state(
    env, total_steps, episode_count, best_success_rate, episode_return, curriculum
):
    state = {
        "total_steps": total_steps,
        "episode_count": episode_count,
        "best_success_rate": best_success_rate,
        "episode_return": episode_return,
        "env": env.state_dict(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    if curriculum is not None:
        state["curriculum_state"] = curriculum.__dict__
    return state


def pretrain_stage(agent, env, task_config, buffer):
    """Run one pre-training stage."""
    print(f"\nPre-training: {task_config['task']}")
    print(f"  Episodes: {task_config['num_episodes']}")

    # Modify environment
    env.num_clots = task_config.get('num_clots', env.num_clots)

    for ep in range(task_config['num_episodes']):
        obs_dict, info = env.reset()
        obs = obs_dict['nodes']
        done = False
        episode_return = 0

        while not done:
            ctx = build_context(env, obs_dict)
            state = obs_dict['clot_state'].flatten()

            actions, log_probs, values = agent.act(obs, ctx, state)

            next_obs_dict, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            next_ctx = build_context(env, next_obs_dict)
            next_state = next_obs_dict['clot_state'].flatten()

            agent_rewards = info.get('agent_rewards', np.full(env.num_robots, reward / env.num_robots))

            buffer.store(
                obs, actions, agent_rewards,
                np.full(env.num_robots, done, dtype=np.float32),
                log_probs, values, ctx, state,
                next_obs=next_obs_dict['nodes'], next_ctx=next_ctx,
                next_state=next_state,
                terminals=np.full(env.num_robots, terminated, dtype=np.float32),
            )

            obs_dict = next_obs_dict
            obs = obs_dict['nodes']
            episode_return += reward

            # Update policy
            if len(buffer) >= agent.n_steps:
                agent.buffer = buffer  # Temporarily swap
                agent.update(n_epochs=5, batch_size=128)  # Fewer epochs for pre-training
                buffer = ContextRolloutBuffer()
                agent.buffer = buffer

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{task_config['num_episodes']}, Return: {episode_return:.1f}")

    return buffer


def train():
    args = parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create log directory
    exp_name = f"{args.architecture}_r{args.robots}_c{args.clots}"
    if args.adaptive_curriculum:
        exp_name += "_adaptive"
    if args.pretrain:
        exp_name += "_pretrain"
    if args.exploration_bonus > 0:
        exp_name += f"_explore{args.exploration_bonus}"

    log_dir = Path(args.log_dir) / exp_name / f"seed_{args.seed}"
    if args.resume:
        log_dir = Path(args.resume).resolve().parent
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print(f"Advanced MAPPO Training")
    print("=" * 80)
    print(f"Architecture: {args.architecture}")
    print(f"Environment: {args.robots} robots, {args.clots} clots")
    print(f"Adaptive curriculum: {args.adaptive_curriculum}")
    print(f"Pre-training: {args.pretrain}")
    print(f"Exploration bonus: {args.exploration_bonus}")
    print(f"Device: {args.device}")
    print(f"Log: {log_dir}")
    print("=" * 80)

    # Create environment
    env = Vascular3DMARLEnv(
        num_robots=args.robots,
        num_clots=1 if (args.adaptive_curriculum or args.pretrain) else args.clots,
        horizon=args.horizon,
        seed=args.seed,
        obs_mode=args.obs_mode,
        scenario=args.scenario,
        randomize_scenario=not args.fixed_scenario,
        scenario_pool=args.scenario_pool,
        robot_radius=args.robot_radius,
        reward_mode=args.reward_mode,
    )
    eval_env = Vascular3DMARLEnv(
        num_robots=args.robots,
        num_clots=args.clots,
        horizon=args.horizon,
        seed=args.eval_seed,
        obs_mode=args.obs_mode,
        scenario=args.scenario,
        randomize_scenario=not args.fixed_scenario,
        scenario_pool=args.scenario_pool,
        robot_radius=args.robot_radius,
        reward_mode=args.reward_mode,
    )

    sample_obs = env.observation_space.sample()
    obs_dim = sample_obs['nodes'].shape[1]
    action_dim = env.action_space.shape[1]
    state_dim = sample_obs['clot_state'].size

    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}, State dim: {state_dim}")
    print("=" * 80)

    # Create agent
    agent = MAPPOAdvanced(
        n_agents=args.robots,
        obs_dim=obs_dim,
        action_dim=action_dim,
        architecture=args.architecture,
        state_dim=state_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        sparse_k=args.k_neighbors,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=args.device,
    )

    # Custom buffer that stores context
    buffer = ContextRolloutBuffer()
    agent.buffer = buffer
    resume_state = agent.load(args.resume) if args.resume else {}

    # Optional modules
    curriculum = AdaptiveCurriculum() if args.adaptive_curriculum else None
    exploration = ExplorationBonus(state_dim) if args.exploration_bonus > 0 else None

    # Pre-training
    if args.pretrain and not args.resume:
        print("\n" + "=" * 80)
        print("PRE-TRAINING PHASE")
        print("=" * 80)

        for stage_name, stage_func in PretrainingTasks.get_stages():
            task_config = stage_func(env)
            buffer = pretrain_stage(agent, env, task_config, buffer)

        # Reset to full task
        env.num_clots = args.clots
        print("\nPre-training complete. Starting main training...")

    # Main training
    print("\n" + "=" * 80)
    print("MAIN TRAINING")
    print("=" * 80)

    total_steps = int(resume_state.get("total_steps", 0))
    episode_count = int(resume_state.get("episode_count", 0))
    best_success_rate = float(resume_state.get("best_success_rate", 0.0))

    if resume_state.get("env") is not None:
        env.load_state_dict(resume_state["env"])
        obs_dict = env._build_observation()
        info = {"scenario": env.active_scenario}
        np.random.set_state(resume_state["numpy_rng_state"])
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state" in resume_state:
            torch.cuda.set_rng_state_all([
                state.cpu() for state in resume_state["cuda_rng_state"]
            ])
        if curriculum is not None and "curriculum_state" in resume_state:
            curriculum.__dict__.update(resume_state["curriculum_state"])
        print(f"Resumed from {args.resume} at step {total_steps}")
    else:
        obs_dict, info = env.reset()
    obs = obs_dict['nodes']
    episode_return = float(resume_state.get("episode_return", 0.0))
    last_checkpoint_step = total_steps

    start_time = time.time()

    while total_steps < args.timesteps:
        # Collect rollout
        for _ in range(min(args.n_steps, args.timesteps - total_steps)):
            ctx = build_context(env, obs_dict)
            state = obs_dict['clot_state'].flatten()

            actions, log_probs, values = agent.act(obs, ctx, state)

            next_obs_dict, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            next_ctx = build_context(env, next_obs_dict)
            next_state = next_obs_dict['clot_state'].flatten()

            agent_rewards = info.get('agent_rewards', np.full(args.robots, reward / args.robots))

            # Exploration bonus
            if exploration:
                bonus = exploration.compute_bonus(state)
                agent_rewards = agent_rewards + bonus

            buffer.store(
                obs, actions, agent_rewards,
                np.full(args.robots, done, dtype=np.float32),
                log_probs, values, ctx, state,
                next_obs=next_obs_dict['nodes'], next_ctx=next_ctx,
                next_state=next_state,
                terminals=np.full(args.robots, terminated, dtype=np.float32),
            )

            obs_dict = next_obs_dict
            obs = obs_dict['nodes']
            episode_return += reward
            total_steps += 1

            # Episode end
            if done:
                episode_count += 1

                # Update curriculum
                if curriculum:
                    curriculum_info = curriculum.update(
                        info.get('success', False),
                        {'removal_rate': info.get('removal_rate', 0.0)},
                    )
                    if curriculum_info['difficulty_changed']:
                        params = curriculum.get_env_params()
                        env.num_clots = params['num_clots']
                        print(f"\n🎯 Curriculum: {params['num_clots']} clots "
                              f"(reason: {curriculum_info['reason']})")

                # Logging
                if episode_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = total_steps / elapsed
                    print(f"Episode {episode_count:5d} | Steps {total_steps:7d} | "
                          f"Return {episode_return:7.1f} | FPS {fps:.0f}")

                with open(log_dir / "episode_metrics.jsonl", "a") as f:
                    f.write(json.dumps({
                        "step": total_steps,
                        "episode": episode_count,
                        "scenario": info.get("scenario"),
                        "return": episode_return,
                        "success": bool(info.get("success", False)),
                        "removal_rate": float(info.get("removal_rate", 0.0)),
                        "wall_collisions": int(info.get("wall_collisions", 0)),
                    }) + "\n")

                obs_dict, info = env.reset()
                obs = obs_dict['nodes']
                episode_return = 0

            # Evaluation
            if total_steps % args.eval_interval == 0:
                print("\n" + "-" * 80)
                print(f"Evaluation at step {total_steps}")
                eval_metrics = evaluate(
                    agent, eval_env, n_episodes=args.eval_episodes,
                    seed_base=args.eval_seed,
                )

                print(f"  Success: {eval_metrics['success']:.2%}")
                print(f"  Return: {eval_metrics['episode_return']:.1f}")
                print(f"  Removal: {eval_metrics['removal_rate']:.2%}")
                print(f"  Contact miss: {eval_metrics['contact_miss']:.2%}")
                print("-" * 80 + "\n")

                if eval_metrics['success'] > best_success_rate:
                    best_success_rate = eval_metrics['success']
                    agent.save(log_dir / "best_policy.pt")
                    print(f"✓ Best model saved (success: {best_success_rate:.2%})")

                with open(log_dir / "eval_metrics.jsonl", "a") as f:
                    f.write(json.dumps({'step': total_steps, **eval_metrics}) + "\n")

        # Update policy
        if len(buffer) > 0:
            metrics = agent.update(n_epochs=args.n_epochs, batch_size=args.batch_size)

            with open(log_dir / "update_metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": total_steps, **metrics}) + "\n")

            if total_steps % (args.n_steps * 5) == 0:
                print(f"Update | Actor: {metrics['actor_loss']:.4f} | "
                      f"Critic: {metrics['critic_loss']:.4f} | "
                      f"Entropy: {metrics['entropy']:.4f}")

            if total_steps - last_checkpoint_step >= args.save_interval:
                training_state = capture_training_state(
                    env, total_steps, episode_count, best_success_rate,
                    episode_return, curriculum,
                )
                agent.save(log_dir / f"checkpoint_{total_steps}.pt", training_state)
                last_checkpoint_step = total_steps

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    final_metrics = evaluate(
        agent, eval_env, n_episodes=max(20, args.eval_episodes),
        seed_base=args.eval_seed,
    )
    print(f"Success rate: {final_metrics['success']:.2%}")
    print(f"Average return: {final_metrics['episode_return']:.1f}")
    print(f"Removal rate: {final_metrics['removal_rate']:.2%}")
    print(f"Contact miss: {final_metrics['contact_miss']:.2%}")
    print("=" * 80)

    final_state = capture_training_state(
        env, total_steps, episode_count, best_success_rate, episode_return, curriculum
    )
    agent.save(log_dir / "final_policy.pt", final_state)

    summary = {
        'architecture': args.architecture,
        'total_steps': total_steps,
        'best_success_rate': best_success_rate,
        'final_metrics': final_metrics,
        'training_time': time.time() - start_time,
    }

    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    env.close()
    eval_env.close()

    print(f"\nTraining complete! Results saved to {log_dir}")


if __name__ == "__main__":
    train()
