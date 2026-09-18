"""A/B the observation change on the batched environment.

Same question as `ab_obs_mode.py` -- does putting vessel geometry in the
observation reduce `contact_miss`? -- but run through `VectorVascularEnv`, which
is ~5x faster per transition, so each arm sees far more experience for the same
wall-clock.

Physics, reward, network and seeds are identical across arms; only `obs_mode`
differs. The vectorized env is verified against the single env in
`tests/test_vector_env.py`, so a difference here is attributable to the
observation and not to a divergent simulator.

    python scripts/ab_obs_vector.py --transitions 300000 --seeds 3
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


def run_arm(obs_mode: str, seed: int, args) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    env = VectorVascularEnv(
        n_envs=args.n_envs,
        scenario=args.scenario,
        num_robots=args.robots,
        num_clots=args.clots,
        horizon=args.horizon,
        seed=seed,
        randomize_scenario=True,
        obs_mode=obs_mode,
    )
    obs = env.reset_all()
    obs_dim = obs["nodes"].shape[2]
    state_dim = int(obs["clot_state"][0].size)

    agent = MADDPG(
        n_agents=args.robots, obs_dim=obs_dim, action_dim=3,
        hidden_dim=args.hidden_dim, actor_lr=1e-4, critic_lr=1e-3,
        gamma=0.99, tau=0.01, buffer_size=200000, batch_size=256,
        device=args.device, state_dim=state_dim, share_parameters=True, seed=seed,
    )
    device = agent.device

    episodes: list[dict] = []
    ep_returns = np.zeros((args.n_envs,), np.float32)
    transitions = 0
    t0 = time.time()
    iters = args.transitions // args.n_envs

    for _ in range(iters):
        eps = max(0.05, 1.0 - 0.95 * transitions / max(args.epsilon_decay, 1))
        nodes = obs["nodes"]
        with torch.no_grad():
            t = torch.as_tensor(nodes.reshape(-1, obs_dim), device=device)
            act = agent.actor(0)(t).cpu().numpy().reshape(args.n_envs, args.robots, 3)
        act = np.clip(
            act + eps * rng.normal(0, 0.2, act.shape).astype(np.float32), -1.0, 1.0
        )

        nxt, reward, term, trunc, info = env.step(act)
        done = term | trunc
        next_nodes, next_clot = nxt["nodes"], nxt["clot_state"]
        if "final_observation" in info:
            next_nodes = np.where(
                done[:, None, None], info["final_observation"]["nodes"], next_nodes
            )
            next_clot = np.where(
                done[:, None, None], info["final_observation"]["clot_state"], next_clot
            )

        per_agent = info["agent_rewards"] + (
            info["team_reward"] / max(args.robots, 1)
        )[:, None]
        states = obs["clot_state"].reshape(args.n_envs, -1)
        next_states = next_clot.reshape(args.n_envs, -1)
        for e in range(args.n_envs):
            agent.replay_buffer.store(
                obs=nodes[e], actions=act[e], rewards=per_agent[e],
                next_obs=next_nodes[e], done=bool(term[e]),
                state=states[e], next_state=next_states[e],
            )
        transitions += args.n_envs
        ep_returns += reward

        if len(agent.replay_buffer) >= max(256, args.learning_starts):
            agent.update()

        for e in np.flatnonzero(done):
            episodes.append({
                "return": float(ep_returns[e]),
                "success": bool(info["success"][e]),
                "removal": float(info["removal_rate"][e]),
                "contact_miss": bool(info["contact_miss"][e]),
            })
            ep_returns[e] = 0.0
        obs = nxt

    env.close()
    tail = episodes[len(episodes) // 2:] or episodes
    return {
        "obs_mode": obs_mode,
        "seed": seed,
        "episodes": len(episodes),
        "contact_miss": float(np.mean([e["contact_miss"] for e in tail])),
        "success": float(np.mean([e["success"] for e in tail])),
        "removal": float(np.mean([e["removal"] for e in tail])),
        "return": float(np.mean([e["return"] for e in tail])),
        "seconds": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", type=int, default=300000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--robots", type=int, default=3)
    ap.add_argument("--clots", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--scenario", default="bifurcation")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--epsilon-decay", type=int, default=200000)
    ap.add_argument("--learning-starts", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="ab_obs_vector.json")
    args = ap.parse_args()

    results = []
    for obs_mode in ("legacy", "geometric"):
        for s in range(args.seeds):
            r = run_arm(obs_mode, 100 + s, args)
            results.append(r)
            print(
                f"{obs_mode:10s} seed={r['seed']} eps={r['episodes']:5d} "
                f"contact_miss={r['contact_miss']:.3f} success={r['success']:.3f} "
                f"removal={r['removal']:.3f} ret={r['return']:+8.2f} ({r['seconds']:.0f}s)",
                flush=True,
            )

    print("\n=== summary (mean over seeds, last half of episodes) ===", flush=True)
    summary = {}
    for obs_mode in ("legacy", "geometric"):
        arm = [r for r in results if r["obs_mode"] == obs_mode]
        summary[obs_mode] = {
            k: float(np.mean([r[k] for r in arm]))
            for k in ("contact_miss", "success", "removal", "return")
        }
        s = summary[obs_mode]
        print(
            f"{obs_mode:10s} contact_miss={s['contact_miss']:.3f} "
            f"success={s['success']:.3f} removal={s['removal']:.3f} "
            f"return={s['return']:+8.2f}",
            flush=True,
        )
    Path(args.out).write_text(json.dumps(
        {"results": results, "summary": summary}, indent=2
    ))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
