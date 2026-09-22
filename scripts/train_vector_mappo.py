"""Territory-balanced vectorized MAPPO with auditable transition recording."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.balanced_vector_env import BalancedVectorVascularEnv
from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from environments.vessel_geometry import resolve_pool
from marl.mappo_advanced import MAPPOAdvanced, ContextRolloutBuffer
from marl.transition_dataset import EpisodeTransitionWriter


TRAIN_ARCHITECTURES = ("gat", "edge_bias_gat", "mlp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=TRAIN_ARCHITECTURES, default="gat")
    parser.add_argument("--control-mode", choices=("world", "local", "guided", "flow_guided", "flow_spread"), default="world")
    parser.add_argument("--residual-scale", type=float, default=0.2)
    parser.add_argument("--guidance-speed", type=float, default=0.65)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--robots", type=int, default=5,
                        help="tensor agent dimension (max agents for the policy)")
    parser.add_argument("--active-robots", type=int, default=None,
                        help="real agents simulated; <= --robots. Default: all "
                             "slots real (legacy fixed-N behaviour)")
    parser.add_argument("--max-agents", type=int, default=0,
                        help="declared padding capacity recorded in checkpoint "
                             "meta; 0 = fixed-N (legacy). Must be >= --robots "
                             "when nonzero")
    parser.add_argument("--initialization-mode",
                        choices=("legacy", "stratified", "random", "separated"),
                        default="legacy",
                        help="robot spawn rule; 'separated' uses geodesic "
                             "farthest-point sampling with min Euclidean and "
                             "geodesic separation constraints")
    parser.add_argument("--separated-min-euclidean-radii", type=float, default=8.0,
                        help="minimum Euclidean spawn separation in robot radii")
    parser.add_argument("--separated-min-geodesic-fraction", type=float,
                        default=0.12,
                        help="minimum geodesic spawn separation as a fraction "
                             "of total vessel arclength")
    parser.add_argument("--clots", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="real environment transitions across all vector slots")
    parser.add_argument("--n-steps", type=int, default=128,
                        help="batched environment steps per PPO update")
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--scenario-pool", default="anatomical")
    parser.add_argument("--tree-resample-interval", type=int, default=900)
    parser.add_argument("--robot-radius", type=float, default=0.0011)
    parser.add_argument("--obs-mode", choices=("geometric", "geometric_v2", "legacy"),
                        default="geometric")
    parser.add_argument("--reward-mode", choices=("milestone", "baseline"), default="milestone")
    parser.add_argument("--contact-mode", choices=("geodesic", "euclidean"), default="geodesic")
    parser.add_argument("--critic-value-mode", choices=("v", "q"), default="v")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--coverage-bonus", type=float, default=0.2)
    parser.add_argument("--step-cost", type=float, default=0.0)
    parser.add_argument("--approach-scale", type=float, default=0.1)
    parser.add_argument("--reward-double-count", choices=("on", "off"), default="on")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-clip", type=float, default=10.0)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--no-control-margin", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-dir", default="experiments/anatomical_vector")
    parser.add_argument("--run-dir", default="",
                        help="explicit output directory, including for branched resume runs")
    parser.add_argument("--eval-interval", type=int, default=100000)
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="episodes per territory during training")
    parser.add_argument("--final-eval-episodes", type=int, default=20)
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="skip simulator-heavy evaluations; use an external evaluator")
    parser.add_argument("--eval-seed", type=int, default=100000)
    parser.add_argument("--save-interval", type=int, default=100000)
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--dataset-shard-size", type=int, default=32768)
    parser.add_argument("--no-dataset", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--world-model", default="")
    parser.add_argument("--imagination-horizon", type=int, default=3)
    parser.add_argument("--imagination-blend", type=float, default=0.25)
    parser.add_argument("--imagination-uncertainty", type=float, default=0.05)
    parser.add_argument("--force-world-model", action="store_true")
    parser.add_argument("--curriculum", action="store_true",
                        help="stage clot count difficulty by real-transition budget")
    parser.add_argument("--curriculum-boundaries", type=float, nargs="+",
                        default=(0.2, 0.5), metavar="FRACTION",
                        help="cumulative transition fractions where difficulty changes")
    parser.add_argument("--curriculum-difficulties", type=float, nargs="+",
                        default=(0.2, 0.6, 1.0), metavar="DIFFICULTY")
    return parser.parse_args()


def validate_curriculum(args):
    boundaries = tuple(float(value) for value in args.curriculum_boundaries)
    difficulties = tuple(float(value) for value in args.curriculum_difficulties)
    if len(difficulties) != len(boundaries) + 1:
        raise ValueError("curriculum difficulties must have one more value than boundaries")
    if any(value <= 0.0 or value >= 1.0 for value in boundaries):
        raise ValueError("curriculum boundaries must be strictly between 0 and 1")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("curriculum boundaries must be strictly increasing")
    if any(value < 0.0 or value > 1.0 for value in difficulties):
        raise ValueError("curriculum difficulties must be in [0, 1]")
    return boundaries, difficulties


def curriculum_difficulty(transitions, total_transitions, boundaries, difficulties):
    fraction = transitions / max(total_transitions, 1)
    for boundary, difficulty in zip(boundaries, difficulties):
        if fraction < boundary:
            return difficulty
    return difficulties[-1]


def build_context(env, obs):
    """Context dict for one policy forward pass.

    `agent_mask` is included whenever the observation carries one (padded
    rollouts); legacy obs without it omit the key and nothing downstream
    changes.
    """
    ctx = {
        "positions": env.robot_positions.copy(),
        "velocities": env.robot_velocities.copy(),
        "adjacency": obs["adjacency"].copy(),
    }
    if "agent_mask" in obs:
        ctx["agent_mask"] = obs["agent_mask"].copy()
    return ctx


def evaluate_territories(agent, args, episodes_per_territory):
    actor_mode, critic_mode = agent.actor.training, agent.critic.training
    agent.actor.eval()
    agent.critic.eval()
    env = None
    try:
        per_territory = {}
        episode_records = []
        for scenario_index, scenario in enumerate(resolve_pool(args.scenario_pool)):
            env = Vascular3DMARLEnv(
                scenario=scenario,
                scenario_pool=[scenario],
                randomize_scenario=False,
                num_robots=args.robots,
                num_clots=args.clots,
                horizon=args.horizon,
                robot_radius=args.robot_radius,
                obs_mode=args.obs_mode,
                reward_mode=args.reward_mode,
                contact_mode=args.contact_mode,
                coverage_bonus=args.coverage_bonus,
                step_cost=args.step_cost,
                approach_scale=args.approach_scale,
                reward_double_count=args.reward_double_count,
                control_margin=not args.no_control_margin,
                seed=args.eval_seed + scenario_index * 10000,
            )
            records = {"success": [], "removal_rate": [], "return": [], "wall_hits": [], "wall_hits_total": [], "wall_hits_per_step": []}
            for episode in range(episodes_per_territory):
                obs, _ = env.reset(seed=args.eval_seed + scenario_index * 10000 + episode)
                total_return = 0.0
                wall_total = 0
                episode_steps = 0
                while True:
                    ctx = build_context(env, obs)
                    state = obs["clot_state"].reshape(-1)
                    action, _, _ = agent.act(
                        obs["nodes"], ctx, state, deterministic=True
                    )
                    obs, reward, terminated, truncated, info = env.step(
                        agent.env_action(action, obs, env)
                    )
                    total_return += reward
                    wall_total += int(info["wall_collisions"])
                    episode_steps += 1
                    if terminated or truncated:
                        break
                episode_records.append({"scenario": scenario,
                                        "episode_seed": args.eval_seed + scenario_index * 10000 + episode,
                                        "success": float(info["success"]),
                                        "removal_rate": float(info["removal_rate"]),
                                        "wall_hits_total": wall_total, "steps": episode_steps})
                records["success"].append(float(info["success"]))
                records["removal_rate"].append(float(info["removal_rate"]))
                records["return"].append(total_return)
                records["wall_hits"].append(float(wall_total))
                records["wall_hits_total"].append(float(wall_total))
                records["wall_hits_per_step"].append(wall_total / max(episode_steps, 1))
            env.close()
            per_territory[scenario] = {
                key: float(np.mean(values)) for key, values in records.items()
            }

        macro = {
            key: float(np.mean([metrics[key] for metrics in per_territory.values()]))
            for key in ("success", "removal_rate", "return", "wall_hits", "wall_hits_total", "wall_hits_per_step")
        }
        return {"macro": macro, "per_territory": per_territory, "episodes": episode_records, "split": "validation"}
    finally:
        if env is not None:
            env.close()
        agent.actor.train(actor_mode)
        agent.critic.train(critic_mode)

def capture_training_state(
    env, writer, transitions, episodes, best_score, episode_returns,
    origin_transitions,
):
    if writer is not None:
        writer.flush()
    state = {
        "transitions": transitions,
        "origin_transitions": origin_transitions,
        "episodes": episodes,
        "best_score": best_score,
        "episode_returns": episode_returns.copy(),
        "env": env.state_dict(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if writer is not None:
        state["dataset_writer"] = writer.state_dict()
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return state


def main():
    args = parse_args()
    curriculum_boundaries, curriculum_difficulties = validate_curriculum(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.resume:
        run_dir = Path(args.resume).resolve().parent
    else:
        run_dir = (
            Path(args.log_dir) / f"{args.architecture}_r{args.robots}_c{args.clots}"
            / f"seed_{args.seed}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_name = f"resume_config_{int(time.time())}.json" if args.resume else "config.json"
    (run_dir / config_name).write_text(json.dumps(vars(args), indent=2))
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")

    env = BalancedVectorVascularEnv(
        n_envs=args.n_envs,
        scenario_pool=args.scenario_pool,
        tree_resample_interval=args.tree_resample_interval,
        seed=args.seed,
        num_robots=args.robots,
        num_clots=args.clots,
        horizon=args.horizon,
        robot_radius=args.robot_radius,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        contact_mode=args.contact_mode,
        coverage_bonus=args.coverage_bonus,
        step_cost=args.step_cost,
        approach_scale=args.approach_scale,
        reward_double_count=args.reward_double_count,
        control_margin=not args.no_control_margin,
        active_robots=args.active_robots,
        initialization_mode=args.initialization_mode,
        separated_min_euclidean_radii=args.separated_min_euclidean_radii,
        separated_min_geodesic_fraction=args.separated_min_geodesic_fraction,
    )
    if args.curriculum:
        env.set_difficulty(curriculum_difficulty(
            0, args.timesteps, curriculum_boundaries, curriculum_difficulties
        ))
    obs = env.reset_all()
    obs_dim = obs["nodes"].shape[-1]
    state_dim = obs["clot_state"].shape[-2] * obs["clot_state"].shape[-1]
    agent = MAPPOAdvanced(
        n_agents=args.robots,
        obs_dim=obs_dim,
        action_dim=3,
        architecture=args.architecture,
        control_mode=args.control_mode,
        residual_scale=args.residual_scale,
        guidance_speed=args.guidance_speed,
        state_dim=state_dim,
        max_agents=args.max_agents,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        device=args.device,
        critic_value_mode=args.critic_value_mode,
        dropout=args.dropout,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_clip=args.value_clip,
        value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm,
        log_std_init=args.log_std_init,
    )
    agent.meta.update({key: getattr(args, key) for key in
                       ("obs_mode", "contact_mode", "reward_double_count", "coverage_bonus",
                        "step_cost", "approach_scale", "no_control_margin")})
    agent.meta.update({
        "initialization_mode": args.initialization_mode,
        "active_robots": args.active_robots,
    })
    agent.buffer = ContextRolloutBuffer()

    if args.world_model:
        from marl.world_model import load_world_model
        world_model, world_metrics = load_world_model(args.world_model, args.device)
        gate_passed = bool(world_metrics.get("gate", {}).get("passed", False))
        if not gate_passed and not args.force_world_model:
            raise RuntimeError(
                "world-model validation gate did not pass; use "
                "--force-world-model only for an explicit ablation"
            )
        agent.configure_world_model(
            world_model,
            horizon=args.imagination_horizon,
            blend=args.imagination_blend,
            uncertainty_threshold=args.imagination_uncertainty,
        )

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else run_dir / "dataset"
    writer = None if args.no_dataset else EpisodeTransitionWriter(
        dataset_dir, args.n_envs, run_id=args.seed,
        shard_size=args.dataset_shard_size,
    )

    if args.resume:
        original_config = Path(args.resume).resolve().parent / "config.json"
        if original_config.exists():
            previous = json.loads(original_config.read_text())
            legacy_defaults = {"contact_mode": "euclidean", "reward_double_count": "on",
                               "coverage_bonus": 0.2, "step_cost": 0.0, "approach_scale": 0.1,
                               "no_control_margin": False}
            for key in ("obs_mode", "reward_mode", "robots", "clots", "horizon", "n_envs",
                        "scenario_pool", "robot_radius", *legacy_defaults):
                if previous.get(key, legacy_defaults.get(key)) != getattr(args, key):
                    raise ValueError(f"resume configuration mismatch for {key}")
    resume_state = agent.load(args.resume) if args.resume else {}
    transitions = int(resume_state.get("transitions", 0))
    origin_path = run_dir / "run_origin.json"
    branched_resume = bool(
        args.run_dir and args.resume
        and Path(args.resume).resolve().parent != run_dir.resolve()
    )
    if origin_path.exists():
        initial_transitions = int(json.loads(
            origin_path.read_text()
        )["initial_real_transitions"])
    elif branched_resume:
        initial_transitions = transitions
    elif "origin_transitions" in resume_state:
        initial_transitions = int(resume_state["origin_transitions"])
    elif (run_dir / "config.json").exists():
        initial_transitions = 0
    else:
        initial_transitions = transitions
    origin_path.write_text(json.dumps({
        "initial_real_transitions": initial_transitions,
    }, indent=2))
    episodes = int(resume_state.get("episodes", 0))
    best_score = tuple(resume_state.get("best_score", (-1.0, -1.0)))
    episode_returns = np.asarray(
        resume_state.get("episode_returns", np.zeros(args.n_envs)), dtype=np.float64
    )
    if resume_state:
        env.load_state_dict(resume_state["env"])
        obs = env.observe()
        np.random.set_state(resume_state["numpy_rng_state"])
        torch.set_rng_state(resume_state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state" in resume_state:
            torch.cuda.set_rng_state_all([
                state.cpu() for state in resume_state["cuda_rng_state"]
            ])
        if writer is not None and "dataset_writer" in resume_state:
            writer.load_state_dict(resume_state["dataset_writer"])

    difficulty = env.difficulty
    if args.curriculum:
        scheduled_difficulty = curriculum_difficulty(
            transitions, args.timesteps,
            curriculum_boundaries, curriculum_difficulties,
        )
        if not np.isclose(difficulty, scheduled_difficulty):
            raise ValueError(
                f"checkpoint difficulty {difficulty} does not match the requested "
                f"curriculum stage {scheduled_difficulty} at {transitions} transitions"
            )

    recent_success = deque(maxlen=500)
    recent_removal = deque(maxlen=500)
    start = time.time()
    invocation_start_transitions = transitions
    next_eval = ((transitions // args.eval_interval) + 1) * args.eval_interval
    next_save = ((transitions // args.save_interval) + 1) * args.save_interval

    print(f"run_dir={run_dir}")
    print(f"architecture={args.architecture} real_transitions={args.timesteps}")
    print(f"territories={len(env.scenarios)} groups={env.group_sizes}")
    print(f"curriculum={args.curriculum} difficulty={difficulty:.3f}")
    if resume_state:
        print(f"resumed={args.resume} transitions={transitions}")

    episode_log = (run_dir / "episode_metrics.jsonl").open("a")
    update_log = (run_dir / "update_metrics.jsonl").open("a")
    eval_log = (run_dir / "eval_metrics.jsonl").open("a")
    curriculum_log = (run_dir / "curriculum_metrics.jsonl").open("a")

    try:
        if not args.skip_evaluation and not args.resume and args.control_mode in ("guided", "flow_guided"):
            initial_evaluation = evaluate_territories(agent, args, args.eval_episodes)
            initial_macro = initial_evaluation["macro"]
            best_score = (initial_macro["success"], initial_macro["removal_rate"])
            eval_log.write(json.dumps({"transitions": 0, **initial_evaluation}) + "\n")
            eval_log.flush()
            agent.save(run_dir / "best_policy.pt")
            agent.save(run_dir / "initial_policy.pt")
            print(f"initial eval macro success={best_score[0]:.3f} removal={best_score[1]:.3f}")
        while transitions < args.timesteps:
            remaining_steps = int(np.ceil((args.timesteps - transitions) / args.n_envs))
            rollout_steps = min(args.n_steps, remaining_steps)
            for _ in range(rollout_steps):
                nodes = obs["nodes"]
                states = obs["clot_state"].reshape(args.n_envs, -1)
                ctx = build_context(env, obs)
                geometry_features = env.geometry_features.copy()

                actions, log_probs, values = agent.act_batch(nodes, ctx, states)
                executed_actions = agent.env_action(actions, obs, env)
                returned_obs, reward, terminated, truncated, info = env.step(executed_actions)
                done = terminated | truncated
                returned_ctx = build_context(env, returned_obs)

                transition_next_obs = {
                    key: value.copy() for key, value in returned_obs.items()
                }
                transition_next_ctx = {
                    key: value.copy() for key, value in returned_ctx.items()
                }
                if "final_observation" in info:
                    final_obs = info["final_observation"]
                    for key in transition_next_obs:
                        mask_shape = (args.n_envs,) + (1,) * (transition_next_obs[key].ndim - 1)
                        transition_next_obs[key] = np.where(
                            done.reshape(mask_shape), final_obs[key], transition_next_obs[key]
                        )
                    final_ctx = info["final_context"]
                    for key in ("positions", "velocities"):
                        transition_next_ctx[key] = np.where(
                            done[:, None, None], final_ctx[key], transition_next_ctx[key]
                        )
                    transition_next_ctx["adjacency"] = transition_next_obs["adjacency"]

                next_states = transition_next_obs["clot_state"].reshape(args.n_envs, -1)
                next_geometry = np.where(
                    done[:, None], geometry_features, env.geometry_features
                )
                # Per-agent reward for the rollout buffer. The team share is
                # divided by the number of ACTIVE robots (env.robot_team_size
                # when the env exposes one, else args.robots), and padding
                # rows are zeroed so PPO never sees a phantom agent-step.
                team_size = getattr(env, "robot_team_size", args.robots)
                if isinstance(team_size, np.ndarray):
                    team_size = team_size.astype(np.float32)[:, None]
                agent_rewards = (
                    info["agent_rewards"]
                    + info["team_reward"][:, None] / np.maximum(team_size, 1)
                ).astype(np.float32)
                rollout_mask = info.get("agent_mask") if isinstance(info, dict) else None
                if rollout_mask is None:
                    rollout_mask = (
                        obs["agent_mask"] if "agent_mask" in obs else None
                    )
                if rollout_mask is not None:
                    agent_rewards = agent_rewards * rollout_mask
                    stored_done = np.repeat(
                        done[:, None], args.robots, axis=1
                    ).astype(np.float32) * rollout_mask
                    stored_terminals = np.repeat(
                        terminated[:, None], args.robots, axis=1
                    ).astype(np.float32) * rollout_mask
                else:
                    stored_done = np.repeat(
                        done[:, None], args.robots, axis=1
                    ).astype(np.float32)
                    stored_terminals = np.repeat(
                        terminated[:, None], args.robots, axis=1
                    ).astype(np.float32)
                agent.buffer.store(
                    nodes, actions, agent_rewards,
                    stored_done,
                    log_probs, values, ctx, states,
                    next_obs=transition_next_obs["nodes"],
                    next_ctx=transition_next_ctx, next_state=next_states,
                    geometry_features=geometry_features,
                    next_geometry_features=next_geometry,
                    terminals=stored_terminals,
                    scenario_id=info["scenario_id"],
                )

                if writer is not None:
                    writer.append(
                        done=done,
                        obs=nodes,
                        next_obs=transition_next_obs["nodes"],
                        action=actions,
                        policy_action=actions,
                        executed_action=executed_actions,
                        rewards=agent_rewards,
                        team_reward=info["team_reward"].astype(np.float32),
                        terminated=terminated,
                        truncated=truncated,
                        state=states,
                        next_state=next_states,
                        adjacency=ctx["adjacency"],
                        next_adjacency=transition_next_ctx["adjacency"],
                        positions=ctx["positions"],
                        next_positions=transition_next_ctx["positions"],
                        velocities=ctx["velocities"],
                        next_velocities=transition_next_ctx["velocities"],
                        scenario_id=info["scenario_id"],
                        geometry_id=info["geometry_id"],
                        geometry_features=geometry_features,
                    )

                transitions += args.n_envs
                episode_returns += reward
                for index in np.flatnonzero(done):
                    episodes += 1
                    success = bool(info["success"][index])
                    removal = float(info["removal_rate"][index])
                    recent_success.append(float(success))
                    recent_removal.append(removal)
                    episode_log.write(json.dumps({
                        "transitions": transitions,
                        "episode": episodes,
                        "env_index": int(index),
                        "scenario": str(info["scenario"][index]),
                        "geometry_id": int(info["geometry_id"][index]),
                        "return": float(episode_returns[index]),
                        "success": success,
                        "removal_rate": removal,
                        "terminated": bool(terminated[index]),
                        "truncated": bool(truncated[index]),
                        "wall_collisions": int(info["wall_collisions"][index]),
                        "wall_hits_total": int(info["wall_hits_total"][index]),
                        "difficulty": difficulty,
                    }) + "\n")
                    episode_returns[index] = 0.0
                if np.any(done):
                    episode_log.flush()
                obs = returned_obs

            update_metrics = agent.update(
                n_epochs=args.n_epochs, batch_size=args.batch_size
            )
            elapsed = time.time() - start
            record = {
                "transitions": transitions,
                "episodes": episodes,
                "fps": (transitions - invocation_start_transitions) / max(elapsed, 1e-6),
                "recent_success": float(np.mean(recent_success)) if recent_success else 0.0,
                "recent_removal": float(np.mean(recent_removal)) if recent_removal else 0.0,
                "difficulty": difficulty,
                **update_metrics,
            }
            update_log.write(json.dumps(record) + "\n")
            update_log.flush()
            print(
                f"transitions={transitions} episodes={episodes} "
                f"success={record['recent_success']:.3f} "
                f"removal={record['recent_removal']:.3f} fps={record['fps']:.0f}"
            )

            if args.curriculum:
                next_difficulty = curriculum_difficulty(
                    transitions, args.timesteps,
                    curriculum_boundaries, curriculum_difficulties,
                )
                if not np.isclose(next_difficulty, difficulty):
                    curriculum_record = {
                        "transitions": transitions,
                        "fraction": transitions / args.timesteps,
                        "old_difficulty": difficulty,
                        "new_difficulty": next_difficulty,
                    }
                    curriculum_log.write(json.dumps(curriculum_record) + "\n")
                    curriculum_log.flush()
                    difficulty = next_difficulty
                    env.set_difficulty(difficulty)
                    print(
                        f"curriculum transitions={transitions} "
                        f"difficulty={difficulty:.3f}"
                    )

            if not args.skip_evaluation and transitions >= next_eval:
                evaluation = evaluate_territories(agent, args, args.eval_episodes)
                eval_log.write(json.dumps({
                    "transitions": transitions, **evaluation
                }) + "\n")
                eval_log.flush()
                macro = evaluation["macro"]
                score = (macro["success"], macro["removal_rate"])
                print(f"eval macro success={score[0]:.3f} removal={score[1]:.3f}")
                if score > best_score:
                    best_score = score
                    agent.save(run_dir / "best_policy.pt")
                next_eval += args.eval_interval

            if transitions >= next_save:
                state = capture_training_state(
                    env, writer, transitions, episodes, best_score, episode_returns,
                    initial_transitions,
                )
                agent.save(run_dir / f"checkpoint_{transitions}.pt", state)
                next_save += args.save_interval

        if args.skip_evaluation:
            final_evaluation = {
                "macro": {"success": None, "removal_rate": None,
                           "return": None, "wall_hits": None},
                "per_territory": {},
                "skipped": True,
            }
        else:
            final_evaluation = evaluate_territories(
                agent, args, args.final_eval_episodes
            )
        final_state = capture_training_state(
            env, writer, transitions, episodes, best_score, episode_returns,
            initial_transitions,
        )
        agent.save(run_dir / "final_policy.pt", final_state)
        if args.skip_evaluation:
            agent.save(run_dir / "best_policy.pt")
        summary = {
            "architecture": args.architecture,
            "control_mode": args.control_mode,
            "residual_scale": args.residual_scale,
            "guidance_speed": args.guidance_speed,
            "seed": args.seed,
            "real_transitions": transitions,
            "initial_real_transitions": initial_transitions,
            "added_real_transitions": transitions - initial_transitions,
            "episodes": episodes,
            "training_seconds": time.time() - start,
            "best_score": best_score,
            "final_evaluation": final_evaluation,
            "dataset_transitions": 0 if writer is None else writer.total_written,
            "curriculum": {
                "enabled": args.curriculum,
                "boundaries": curriculum_boundaries,
                "difficulties": curriculum_difficulties,
                "final_difficulty": difficulty,
            },
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary["final_evaluation"]["macro"], indent=2))
    finally:
        episode_log.close()
        update_log.close()
        eval_log.close()
        curriculum_log.close()
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    main()
