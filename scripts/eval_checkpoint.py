"""Evaluate a saved checkpoint over N episodes.

The number printed at the end of training comes from that run's own env
instance. This script re-evaluates from the checkpoint on a freshly seeded
env, which is what you want before quoting a success rate: it separates
"the policy is good" from "the last few episodes happened to go well".

    python scripts/eval_checkpoint.py --policy <path> --architecture edge_gat \
        --robots 5 --clots 3 --episodes 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from marl.policy_loader import load_policy
from marl.connectivity_allocator import allocate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--architecture", default=None)
    ap.add_argument("--scenario", default="bifurcation",
                    help="topology to evaluate on when --fixed-scenario is set")
    ap.add_argument("--scenario-pool", default="legacy",
                    help="pool to sample from: legacy, anatomical, arterial, "
                         "venous, all")
    ap.add_argument("--fixed-scenario", action="store_true",
                    help="evaluate on --scenario alone; use this to report "
                         "per-territory numbers rather than a pooled average")
    ap.add_argument("--robot-radius", type=float, default=0.0045,
                    help="must match what the checkpoint was trained with")
    ap.add_argument("--robots", type=int, default=5)
    ap.add_argument("--clots", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--obs-mode", default="geometric")
    ap.add_argument("--seed", type=int, default=1000, help="eval seeds start here")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--task-allocator", default="none",
                      choices=("none", "nearest", "flow_spread", "connectivity_aware", "risk_aware_connectivity", "predictive_risk_connectivity"))
    ap.add_argument("--dynamic-particles", action="store_true")
    ap.add_argument("--particle-count", type=int, default=24)
    ap.add_argument("--particle-radius-ratio", type=float, default=1.6)
    ap.add_argument("--particle-lateral-drift", type=float, default=0.15)
    args = ap.parse_args()

    env = Vascular3DMARLEnv(
        num_robots=args.robots,
        num_clots=args.clots,
        horizon=args.horizon,
        seed=args.seed,
        obs_mode=args.obs_mode,
        scenario=args.scenario,
        randomize_scenario=not args.fixed_scenario,
        scenario_pool=args.scenario_pool,
        robot_radius=args.robot_radius,
        dynamic_intravascular_particles=args.dynamic_particles,
        particle_count=args.particle_count,
        particle_radius_ratio=args.particle_radius_ratio,
        particle_lateral_drift=args.particle_lateral_drift,
    )
    policy = load_policy(args.policy, env, device=args.device,
                         architecture=args.architecture)

    rec = {"success": [], "removal": [], "return": [],
           "wall_hits": [], "steps": [], "contact_miss": [],
           "path_length": [], "path_length_per_removed_mass": []}
    previous = None

    for ep in range(args.episodes):
        # Seeds disjoint from training so this is not a replay of seen scenes.
        obs, _ = env.reset(seed=args.seed + ep)
        total, steps = 0.0, 0
        removed_mass_total = 0.0
        wall_hits = 0
        made_contact = False
        while True:
            if args.task_allocator != "none":
                result = allocate(env, args.task_allocator, previous_assignments=previous)
                env.set_task_assignments(result.assignments)
                previous = result.assignments.copy()
            action = policy(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            removed_mass_total += float(info.get("removed_mass", 0.0))
            steps += 1
            wall_hits += int(info.get("wall_collisions", 0))
            made_contact |= bool(info.get("active_contacts", 0))
            if terminated or truncated:
                break

        rec["success"].append(float(info.get("success", False)))
        rec["removal"].append(info.get("removal_rate", 0.0))
        rec["return"].append(total)
        rec["wall_hits"].append(wall_hits)
        rec["steps"].append(steps)
        rec["contact_miss"].append(float(not made_contact))
        rec["path_length"].append(float(info.get("path_length", 0.0)))
        rec["path_length_per_removed_mass"].append(
            np.nan if removed_mass_total <= 1e-8 else
            float(info.get("path_length", 0.0)) / removed_mass_total
        )
        previous = None

    n = args.episodes
    # Binomial standard error on the success rate -- with 50 episodes the
    # uncertainty is a few percent, which matters when comparing arms.
    p = float(np.mean(rec["success"]))
    se = (p * (1 - p) / n) ** 0.5

    print(f"\n{'='*62}")
    print(f"{args.policy}")
    print(f"{n} episodes | {args.robots} robots | {args.clots} clots")
    print(f"{'='*62}")
    print(f"  success rate   {p:6.1%}  ± {se:.1%} (binomial SE)")
    print(f"  removal rate   {np.mean(rec['removal']):6.1%}")
    print(f"  contact miss   {np.mean(rec['contact_miss']):6.1%}")
    print(f"  return         {np.mean(rec['return']):+7.1f} ± {np.std(rec['return']):.1f}")
    print(f"  wall hits      {np.mean(rec['wall_hits']):6.1f} per episode")
    print(f"  episode length {np.mean(rec['steps']):6.0f} steps")
    print(f"  path length   {np.mean(rec['path_length']):8.4f}")
    print(f"  path / mass   {np.nanmean(rec['path_length_per_removed_mass']):8.4f}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
