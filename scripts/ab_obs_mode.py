"""A/B the observation change: does vessel geometry reduce `contact_miss`?

The diagnosis behind the rewrite was that ~45% of episodes ended with the swarm
never touching a clot, and that this was an *observability* failure rather than a
reward-shaping one. That claim is testable: hold the physics, the reward and the
seeds fixed, and vary only `obs_mode`.

  python scripts/ab_obs_mode.py --timesteps 40000 --seeds 3

Reports contact-miss rate, success rate and mean removal for each arm. This is a
short run on a hard task, so the headline number is contact_miss (which moves
early), not success (which needs far more steps).
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

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from marl.maddpg_policy import MADDPG


def run_arm(obs_mode: str, seed: int, args) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = Vascular3DMARLEnv(
        scenario=args.scenario,
        num_robots=args.robots,
        num_clots=args.clots,
        horizon=args.horizon,
        seed=seed,
        randomize_scenario=True,
        randomize_clots=True,
        use_pybullet=False,
        reward_mode="milestone",
        obs_mode=obs_mode,
        curriculum=False,
    )
    obs_dict, _ = env.reset(seed=seed)
    state_dim = int(obs_dict["clot_state"].size)

    agent = MADDPG(
        n_agents=args.robots,
        obs_dim=obs_dict["nodes"].shape[1],
        action_dim=3,
        hidden_dim=args.hidden_dim,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.01,
        buffer_size=100000,
        batch_size=256,
        device=args.device,
        state_dim=state_dim,
        share_parameters=True,
        seed=seed,
    )

    episodes: list[dict] = []
    ep_return = 0.0
    t0 = time.time()
    for timestep in range(1, args.timesteps + 1):
        eps = max(0.05, 1.0 - 0.95 * timestep / args.epsilon_decay_steps)
        obs = obs_dict["nodes"]
        actions = agent.select_actions(obs, add_noise=True, noise_scale=eps)
        nxt, reward, terminated, truncated, info = env.step(actions)

        shared = float(info["team_reward"]) / max(args.robots, 1)
        agent.replay_buffer.store(
            obs=obs, actions=actions,
            rewards=np.asarray(info["agent_rewards"], np.float32) + shared,
            next_obs=nxt["nodes"], done=terminated,
            state=obs_dict["clot_state"].reshape(-1),
            next_state=nxt["clot_state"].reshape(-1),
        )
        ep_return += reward

        if len(agent.replay_buffer) >= 1000 and timestep % args.update_freq == 0:
            agent.update()

        if terminated or truncated:
            episodes.append({
                "return": ep_return,
                "success": bool(info["success"]),
                "removal": float(info["removal_rate"]),
                "contact_miss": info["first_contact_step"] < 0,
                "wall": int(info["wall_collisions"]),
            })
            obs_dict, _ = env.reset()
            agent.reset_noise()
            ep_return = 0.0
        else:
            obs_dict = nxt

    env.close()
    # Report over the last half, after some learning has happened.
    tail = episodes[len(episodes) // 2:] or episodes
    return {
        "obs_mode": obs_mode,
        "seed": seed,
        "episodes": len(episodes),
        "contact_miss": float(np.mean([e["contact_miss"] for e in tail])),
        "success": float(np.mean([e["success"] for e in tail])),
        "removal": float(np.mean([e["removal"] for e in tail])),
        "return": float(np.mean([e["return"] for e in tail])),
        "wall": float(np.mean([e["wall"] for e in tail])),
        "seconds": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=40000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--clots", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--scenario", default="bifurcation")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--update-freq", type=int, default=4)
    ap.add_argument("--epsilon-decay-steps", type=int, default=30000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="ab_obs_mode.json")
    args = ap.parse_args()

    results = []
    for obs_mode in ("legacy", "geometric"):
        for seed in range(args.seeds):
            r = run_arm(obs_mode, 100 + seed, args)
            results.append(r)
            print(
                f"{obs_mode:10s} seed={r['seed']} eps={r['episodes']:4d} "
                f"contact_miss={r['contact_miss']:.3f} success={r['success']:.3f} "
                f"removal={r['removal']:.3f} ret={r['return']:+8.2f} "
                f"({r['seconds']:.0f}s)"
            )

    print("\n=== summary (mean over seeds, last half of episodes) ===")
    for obs_mode in ("legacy", "geometric"):
        arm = [r for r in results if r["obs_mode"] == obs_mode]
        print(
            f"{obs_mode:10s} contact_miss={np.mean([r['contact_miss'] for r in arm]):.3f} "
            f"success={np.mean([r['success'] for r in arm]):.3f} "
            f"removal={np.mean([r['removal'] for r in arm]):.3f} "
            f"return={np.mean([r['return'] for r in arm]):+8.2f}"
        )
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
