"""Train MADDPG against the batched environment.

Same algorithm as `train_vascular_maddpg.py`, but rollouts come from
`VectorVascularEnv`, so each env step contributes `n_envs` transitions to the
replay buffer. That changes the economics: with the single env the optimiser was
the bottleneck (a gradient step costs several times an env step, so one update
per env step left the simulator idle), while here one batched step yields enough
new experience to justify several updates.

    python scripts/train_vector.py --n-envs 64 --timesteps 400000 --device cuda

`--timesteps` counts TOTAL transitions (n_envs * batched steps), so runs stay
comparable with the single-env trainer.
"""

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

from environments.vector_env import VectorVascularEnv
from marl.maddpg_policy import MADDPG


def train(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print(f"Training MADDPG (vectorized) on vascular thrombolysis")
    print(f"  n_envs={args.n_envs} robots={args.robots} clots={args.clots}")
    print(f"  device={device} seed={args.seed}")

    env = VectorVascularEnv(
        n_envs=args.n_envs,
        scenario=args.scenario,
        num_robots=args.robots,
        num_clots=args.clots,
        horizon=args.horizon,
        seed=args.seed,
        randomize_scenario=args.randomize_scenario,
        obs_mode=args.obs_mode,
    )
    obs = env.reset_all()
    obs_dim = obs["nodes"].shape[2]
    state_dim = 0 if args.no_critic_state else int(obs["clot_state"][0].size)

    agent = MADDPG(
        n_agents=args.robots,
        obs_dim=obs_dim,
        action_dim=3,
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

    suffix = f"_{args.tag}" if args.tag else ""
    logdir = Path(args.logdir) / f"vec_{args.scenario}_r{args.robots}_c{args.clots}{suffix}"
    run = 1
    while (logdir / f"run_{run}").exists():
        run += 1
    run_dir = logdir / f"run_{run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  logdir: {run_dir}")

    config = vars(args).copy()
    config.update({"obs_dim": obs_dim, "state_dim": state_dim, "action_dim": 3})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    metrics = (run_dir / "metrics.jsonl").open("a")

    ep_returns = np.zeros((args.n_envs,), np.float32)
    recent_return: deque[float] = deque(maxlen=200)
    recent_success: deque[float] = deque(maxlen=args.curriculum_window)
    recent_miss: deque[float] = deque(maxlen=args.curriculum_window)
    difficulty = 0.0 if args.curriculum else 1.0
    env.set_difficulty(difficulty)

    episode_count = 0
    best = -np.inf
    transitions = 0
    start = time.time()
    batched_steps = args.timesteps // args.n_envs

    for it in range(1, batched_steps + 1):
        frac = transitions / max(args.epsilon_decay_steps, 1)
        eps = max(args.epsilon_end, 1.0 - (1.0 - args.epsilon_end) * frac)

        nodes = obs["nodes"]
        # One actor forward for the whole batch: [n_envs * n_robots, obs_dim].
        flat = nodes.reshape(-1, obs_dim)
        with torch.no_grad():
            t = torch.as_tensor(flat, device=device)
            act = agent.actor(0)(t).cpu().numpy()
        act = act.reshape(args.n_envs, args.robots, 3)
        if eps > 0:
            act = np.clip(
                act + eps * np.random.normal(0, 0.2, act.shape).astype(np.float32),
                -1.0, 1.0,
            )

        nxt, reward, term, trunc, info = env.step(act)
        done = term | trunc
        # On auto-reset the observation for a finished env belongs to the NEXT
        # episode; the stored transition must use the terminal one.
        next_nodes = nxt["nodes"]
        next_state_src = nxt["clot_state"]
        if "final_observation" in info:
            next_nodes = np.where(
                done[:, None, None], info["final_observation"]["nodes"], next_nodes
            )
            next_state_src = np.where(
                done[:, None, None],
                info["final_observation"]["clot_state"],
                next_state_src,
            )

        shared = info["team_reward"] / max(args.robots, 1)
        per_agent = info["agent_rewards"] + shared[:, None]
        states = None if state_dim == 0 else obs["clot_state"].reshape(args.n_envs, -1)
        next_states = (
            None if state_dim == 0 else next_state_src.reshape(args.n_envs, -1)
        )

        for e in range(args.n_envs):
            agent.replay_buffer.store(
                obs=nodes[e], actions=act[e], rewards=per_agent[e],
                next_obs=next_nodes[e],
                # Truncation is not terminal: bootstrapping must continue.
                done=bool(term[e]),
                state=None if states is None else states[e],
                next_state=None if next_states is None else next_states[e],
            )
        transitions += args.n_envs
        ep_returns += reward

        if len(agent.replay_buffer) >= max(args.batch_size, args.learning_starts):
            for _ in range(args.updates_per_step):
                loss = agent.update()

        finished = np.flatnonzero(done)
        for e in finished:
            episode_count += 1
            miss = bool(info["contact_miss"][e])
            recent_return.append(float(ep_returns[e]))
            recent_success.append(1.0 if info["success"][e] else 0.0)
            recent_miss.append(1.0 if miss else 0.0)
            metrics.write(json.dumps({
                "timestep": transitions,
                "episode": episode_count,
                "return": round(float(ep_returns[e]), 3),
                "success": bool(info["success"][e]),
                "removal_rate": round(float(info["removal_rate"][e]), 3),
                "wall_collisions": int(info["wall_collisions"][e]),
                "clots_engaged": int(info["clots_engaged"][e]),
                "contact_miss": miss,
                "difficulty": round(difficulty, 3),
            }) + "\n")
            ep_returns[e] = 0.0
        if finished.size:
            metrics.flush()

        if recent_return and float(np.mean(recent_return)) > best:
            best = float(np.mean(recent_return))
            agent.save(run_dir / "best_policy.pt")

        if (
            args.curriculum
            and len(recent_success) == recent_success.maxlen
        ):
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

        if it % args.log_interval == 0:
            elapsed = time.time() - start
            print(
                f"it {it:6d} | trans {transitions:8d} | eps_done {episode_count:5d} | "
                f"ret {np.mean(recent_return) if recent_return else 0.0:+8.2f} | "
                f"succ {np.mean(recent_success) if recent_success else 0.0:.2f} | "
                f"miss {np.mean(recent_miss) if recent_miss else 0.0:.2f} | "
                f"diff {difficulty:.2f} | eps {eps:.3f} | "
                f"{transitions / elapsed:.0f} trans/s"
            )

    agent.save(run_dir / "final_policy.pt")
    metrics.close()
    env.close()
    print(f"\nDone. Best mean return: {best:.2f}\nLogs: {run_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=64)
    p.add_argument("--scenario", default="bifurcation")
    p.add_argument("--robots", type=int, default=3)
    p.add_argument("--clots", type=int, default=3)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--timesteps", type=int, default=400000,
                   help="total transitions (n_envs * batched steps)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=200000)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--learning-starts", type=int, default=5000)
    p.add_argument("--randomize-scenario", action="store_true", default=True)
    p.add_argument("--obs-mode", default="geometric", choices=["geometric", "legacy"])
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--epsilon-decay-steps", type=int, default=200000)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--no-share-parameters", action="store_true", default=False)
    p.add_argument("--no-critic-state", action="store_true", default=False)
    p.add_argument("--curriculum", action="store_true", default=False)
    p.add_argument("--curriculum-window", type=int, default=100)
    p.add_argument("--curriculum-promote", type=float, default=0.5)
    p.add_argument("--curriculum-demote", type=float, default=0.1)
    p.add_argument("--curriculum-step", type=float, default=0.2)
    p.add_argument("--tag", default="")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logdir", default="logdir")
    train(p.parse_args())
