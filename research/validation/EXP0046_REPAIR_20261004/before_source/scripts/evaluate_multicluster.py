"""Evaluate method 1 and method 2 under one shared physical protocol.

This is a diagnostic/feasibility runner for the new multi-cluster research
line.  It deliberately uses the fair local observer and never touches the
sealed-test ledger.  One controlled entity is interpreted as one magnetic
cluster, so ``clusters=1`` is the sequential baseline and ``clusters=2/3`` are
parallel runs with a minimum-spacing shield.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from environments.mca_physical_env import MCAPhysicalEnv, DynamicsConfig
from marl.multicluster import MultiClusterConfig, MultiClusterController
from scripts.safe_metrics import WallTracker, aggregate, episode_metrics
from scripts.train_mca_compiled import reset_with_valid_particles

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/EXP_0029_MCA_ALL_RANDOM_DYNAMICS.json"
SEQUENTIAL_CONFIG = ROOT / "configs/experiments/EXP_0044_WAIT_PRIOR_DYNAMICS.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("single_sequential", "multi_parallel"), required=True)
    p.add_argument("--clusters", type=int, default=2)
    p.add_argument("--d-min-mm", type=float, default=2.0)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=1300000000)
    p.add_argument("--anatomy", default="mca_m1_lvo")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--noise", type=float, default=.025)
    p.add_argument(
        "--duration-s", type=float, default=5.0,
        help="episode horizon for development feasibility runs; use the registered horizon only after the integrator gate passes",
    )
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="optional hard guard for development runs",
    )
    return p.parse_args()


def run_episode(cfg: DynamicsConfig, protocol: MultiClusterConfig, seed: int, max_steps=None) -> dict:
    # The reference integrator is used here intentionally: the compiled
    # kernel's fixed historical assumptions are not yet certified for the
    # variable-N cluster protocol.
    env = MCAPhysicalEnv(cfg)
    _obs, reset_info = reset_with_valid_particles(env, seed)
    controller = MultiClusterController(env, protocol)
    controller.observer.cfg = replace(controller.observer.cfg, noise=protocol.observation_noise)
    controller.reset(seed)
    tracker = WallTracker(env.num_robots)
    trace = []
    reward = 0.0
    guard_hit = False
    while True:
        action, control_info = controller.act()
        active_before = env.active[:env.num_robots].copy()
        _obs, r, terminated, truncated, info = env.step(action)
        dt = float(info["step_duration_s"])
        tracker.update(info, active_before, dt)
        reward += float(r)
        trace.append({"step": int(env.steps), **control_info})
        if terminated or truncated:
            break
        if max_steps is not None and len(trace) >= max_steps:
            guard_hit = True
            break
    metrics = episode_metrics(info, tracker, float(env.initial_mass.sum()))
    result = dict(seed=seed, reset_info=reset_info, reward=reward,
                  clusters=protocol.clusters, method=protocol.method,
                  min_spacing_threshold_mm=protocol.min_spacing_mm,
                  development_guard_hit=guard_hit,
                  controller=controller.summary(), trace=trace, **metrics,
                  particle_contact_s=float(info["episode_particle_contact_s"]))
    env.close()
    return result


def main():
    args = parse_args()
    if args.method == "single_sequential" and args.clusters != 1:
        raise SystemExit("single_sequential requires --clusters 1")
    if args.method == "multi_parallel" and args.clusters < 2:
        raise SystemExit("multi_parallel requires at least two clusters")
    cfg = DynamicsConfig.from_json(SEQUENTIAL_CONFIG if args.method == "single_sequential" else args.config)
    # A single field enters from the inlet; parallel clusters start at
    # independently sampled branches so interaction is actually exercised.
    cfg = replace(cfg, num_robots=args.clusters, anatomy=args.anatomy,
                  episode_duration_s=args.duration_s,
                  robot_initialization=("upstream_fixed" if args.method == "single_sequential"
                                         else "distributed_branches"))
    protocol = MultiClusterConfig(method=args.method, clusters=args.clusters,
                                  min_spacing_mm=args.d_min_mm,
                                  observation_noise=args.noise)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("w", encoding="utf-8") as stream:
        for index in range(args.episodes):
            row = run_episode(cfg, protocol, args.seed_base + index, args.max_steps)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps({
                "seed": row["seed"], "task_success": row["task_success"],
                "safe_success": row["safe_success"], "removal": row["removal"],
                "elapsed_s": row["elapsed_s"], "wall_contact_s": row["wall_contact_s"],
                "yield_events": row["controller"]["yield_events"],
                "deadlock_events": row["controller"]["deadlock_events"],
                "spacing_violation_steps": row["controller"]["spacing_violation_steps"],
            }, ensure_ascii=False), flush=True)
    summary = dict(method=args.method, clusters=args.clusters,
                   min_spacing_mm=args.d_min_mm, episodes=len(rows),
                   aggregate=aggregate(rows),
                   mean_yield_events=float(np.mean([r["controller"]["yield_events"] for r in rows])),
                   mean_deadlock_events=float(np.mean([r["controller"]["deadlock_events"] for r in rows])),
                   mean_spacing_violation_steps=float(np.mean([r["controller"]["spacing_violation_steps"] for r in rows])),
                   min_observed_spacing_mm=float(np.nanmin([
                       r["controller"]["minimum_spacing_mm"] for r in rows
                       if r["controller"]["minimum_spacing_mm"] is not None
                   ])) if any(r["controller"]["minimum_spacing_mm"] is not None for r in rows) else None,
                   config=str(args.config), anatomy=args.anatomy,
                   duration_s=args.duration_s, max_steps=args.max_steps,
                   development_feasibility=True,
                   local_observation=True, sealed_test_used=False)
    args.out.with_name(args.out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
