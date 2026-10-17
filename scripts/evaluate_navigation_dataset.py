"""Evaluate an IPPO/DQN local policy on frozen navigation_dataset_v1 rows.

This runner never invents scene seeds.  It records the manifest hash, raw row
identity, reset information, and condition-stratified metrics so seen/unseen
comparisons remain paired and auditable.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from stable_baselines3 import DQN, PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.navigation_dataset import MANIFEST, condition_key, iter_rows, reset_options
from scripts.train_sb3_curriculum import CurriculumVascularEnv
from scripts.train_sb3_ippo import IPPOAgentVecEnv


def aggregate(rows):
    metrics = ("navigation_success", "strict_success", "wall_contact_s", "lost",
               "particle_collision_events", "particle_contact_s")
    out = {"episodes": len(rows)}
    for key in metrics:
        values = [float(row[key]) for row in rows if key in row]
        if values:
            out[key] = float(np.mean(values))
    if rows:
        out["navigation_success_rate"] = out.get("navigation_success", 0.)
        out["strict_success_rate"] = out.get("strict_success", 0.)
        out["lost_rate"] = float(np.mean([bool(row.get("lost", 0)) for row in rows]))
        out["collision_episode_rate"] = float(np.mean([
            float(row.get("particle_collision_events", 0.)) > 0. for row in rows]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kind", choices=("ppo", "dqn"), required=True)
    parser.add_argument("--action-set", choices=("semantic", "semantic_reduced", "atomic"), default="semantic")
    parser.add_argument("--split", choices=("train", "validation", "holdout_test"), required=True)
    parser.add_argument("--clusters", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--max-episodes", type=int, default=36,
                        help="0 evaluates every row in the selected split/cluster")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_episodes < 0:
        parser.error("max-episodes must be nonnegative")
    torch.set_num_threads(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(iter_rows(split=args.split, clusters=args.clusters))
    if args.max_episodes:
        rows = rows[:args.max_episodes]
    if not rows:
        raise SystemExit("no manifest rows match the requested split and cluster count")
    manifest_path = Path(MANIFEST)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    anatomies = sorted({row["anatomy"] for row in rows})
    env = IPPOAgentVecEnv(CurriculumVascularEnv(
        seed=4101, clusters=args.clusters, curriculum=False, reward_mode="navigation",
        zero_latency=True, direct_action=True), action_set=args.action_set)
    # Holdout rows are permitted only because this is an explicit evaluation
    # runner. They are never used by the training environment or selector.
    env.physical.train_anatomies = anatomies
    model_cls = PPO if args.kind == "ppo" else DQN
    model = model_cls.load(args.model, env=env, device=args.device)
    completed = []
    try:
        for index, row in enumerate(rows):
            obs = env.reset(options=reset_options(row))
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, infos = env.step(action)
                if bool(done[0]):
                    info = infos[0]
                    metrics = dict(info.get("episode_metrics", {}))
                    completed.append({
                        "manifest_index": index,
                        "split": row["split"],
                        "anatomy": row["anatomy"],
                        "clusters": int(row["clusters"]),
                        "scene_seed": int(row["scene_seed"]),
                        "condition_id": int(row["condition_id"]),
                        "flow_mode": row["flow_mode"],
                        "clot_pattern": row["clot_pattern"],
                        "obstacle_density": row["obstacle_density"],
                        "obstacle_layout": row["obstacle_layout"],
                        "scenario_hash": row["scenario_hash"],
                        "accepted_seed": info.get("scene", {}).get("accepted_seed"),
                        "navigation_success": bool(info.get("navigation_success", metrics.get("navigation_success", False))),
                        "strict_success": bool(metrics.get("strict_success", False)),
                        "wall_contact_s": float(metrics.get("wall_contact_s", np.asarray(info.get("agent_wall_contact_s", 0.)).sum())),
                        "lost": int(metrics.get("lost", np.asarray(info.get("agent_lost", 0.)).sum())),
                        "particle_collision_events": float(metrics.get("particle_collision_events", np.asarray(info.get("agent_particle_collision_events", 0.)).sum())),
                        "particle_contact_s": float(metrics.get("particle_contact_s", np.asarray(info.get("agent_particle_contact_s", 0.)).sum())),
                    })
                    break
    finally:
        env.close()
    by_condition = defaultdict(list)
    by_anatomy = defaultdict(list)
    by_flow = defaultdict(list)
    for row in completed:
        by_condition[(row["anatomy"], row["clusters"], row["flow_mode"], row["clot_pattern"],
                     row["obstacle_density"], row["obstacle_layout"])].append(row)
        by_anatomy[row["anatomy"]].append(row)
        by_flow[row["flow_mode"]].append(row)
    result = {
        "dataset_version": "navigation_dataset_v1",
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": manifest_sha,
        "model": str(args.model),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "algorithm": args.kind,
        "action_set": args.action_set,
        "split": args.split,
        "clusters": args.clusters,
        "protocol": "frozen_manifest_evaluation; holdout rows are never used for selection",
        "overall": aggregate(completed),
        "by_anatomy": {key: aggregate(value) for key, value in sorted(by_anatomy.items())},
        "by_flow_mode": {key: aggregate(value) for key, value in sorted(by_flow.items())},
        "by_condition": {"|".join(map(str, key)): aggregate(value) for key, value in sorted(by_condition.items())},
        "episodes": completed,
    }
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
