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
from stable_baselines3 import DQN, PPO, SAC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.navigation_dataset import MANIFEST, condition_key, iter_rows, reset_options
from scripts.train_sb3_curriculum import CurriculumVascularEnv
from scripts.train_sb3_ippo import IPPOAgentVecEnv, load_navigation_ppo, critic_options


def aggregate(rows):
    metrics = ("navigation_success", "strict_success", "wall_contact_s", "wall_after_reach_s", "lost",
               "particle_collision_events", "particle_contact_s")
    errors = sum(bool(row.get("error") or row.get("integration_failure")) for row in rows)
    out = {"episodes": len(rows) - errors, "requested": len(rows), "errors": errors, "complete": not errors}
    if errors:
        return dict(out, **dict.fromkeys((*metrics, "navigation_success_rate", "strict_success_rate",
                                         "lost_rate", "collision_episode_rate")))
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


def grouped_aggregates(rows):
    by_anatomy = defaultdict(list)
    by_flow = defaultdict(list)
    by_condition = defaultdict(list)
    for row in rows:
        by_anatomy[row["anatomy"]].append(row)
        by_flow[row["flow_mode"]].append(row)
        by_condition[condition_key(row)].append(row)
    return {
        "by_anatomy": {key: aggregate(value) for key, value in sorted(by_anatomy.items())},
        "by_flow_mode": {key: aggregate(value) for key, value in sorted(by_flow.items())},
        "by_condition": {"|".join(map(str, key)): aggregate(value)
                         for key, value in sorted(by_condition.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kind", choices=("ppo", "dqn", "sac"), required=True)
    parser.add_argument("--action-set", choices=("semantic", "semantic_reduced", "semantic_minimal", "atomic", "continuous"), default="semantic")
    parser.add_argument("--split", choices=("train", "validation", "holdout_test"), required=True)
    parser.add_argument("--clusters", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--max-episodes", type=int, default=36,
                        help="0 evaluates every row in the selected split/cluster")
    parser.add_argument("--index-list", default=None,
                        help="comma-separated zero-based indices in the filtered manifest. This preserves exact rows for chunked evaluation.")
    parser.add_argument("--scene-list", type=Path, default=None,
                        help="explicit paired manifest JSON created by build_paired_manifest.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--centerline-match", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--flow-feature", action="store_true",
                        help="must match the checkpoint: the flow adapter expects the flow observation")
    parser.add_argument("--centralized-critic", action="store_true",
                        help="load a MAPPO checkpoint whose value function uses central_state")
    parser.add_argument("--safety-parity", action=argparse.BooleanOptionalAction, default=True,
                        help="apply the HCC matrix wall recovery and near-target settling protocol")
    args = parser.parse_args()
    if args.max_episodes < 0:
        parser.error("max-episodes must be nonnegative")
    torch.set_num_threads(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_rows = list(iter_rows(split=args.split, clusters=args.clusters))
    paired_entries = None
    if args.scene_list is not None:
        payload = json.loads(args.scene_list.read_text())
        paired_entries = payload.get("rows", [])
        rows = [entry["row"] for entry in paired_entries]
        if not rows:
            raise SystemExit("scene-list contains no rows")
        if any(row.get("split") != args.split or int(row.get("clusters")) != args.clusters for row in rows):
            raise SystemExit("scene-list split or cluster count does not match evaluation")
        selected_indices = [int(entry["manifest_index"]) for entry in payload["rows"]]
        if len(set(selected_indices)) != len(selected_indices):
            raise SystemExit("scene-list contains duplicate manifest indices")
    elif args.index_list is not None:
        indices = [int(x) for x in args.index_list.split(',') if x.strip()]
        if any(i < 0 or i >= len(all_rows) for i in indices):
            raise SystemExit("index-list contains an out-of-range manifest index")
        rows = [all_rows[i] for i in indices]
        selected_indices = indices
    elif args.max_episodes:
        # Keep the pilot spread over the frozen catalogue instead of taking
        # only the first (weak-flow) block of the JSONL file.
        selected_indices = np.linspace(0, len(all_rows) - 1, min(args.max_episodes, len(all_rows)), dtype=int).tolist()
        rows = [all_rows[int(index)] for index in selected_indices]
    else:
        rows = all_rows
        selected_indices = list(range(len(rows)))
    if not rows:
        raise SystemExit("no manifest rows match the requested split and cluster count")
    manifest_path = Path(MANIFEST)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if paired_entries is not None and payload.get("manifest_sha256") != manifest_sha:
        raise ValueError("paired scene list refers to a different manifest")
    anatomies = sorted({row["anatomy"] for row in rows})
    model_cls = {"ppo": PPO, "dqn": DQN, "sac": SAC}[args.kind]
    if args.kind == "sac" and args.action_set != "continuous":
        raise SystemExit("SAC manifest evaluation requires --action-set continuous")
    if args.kind != "sac" and args.action_set == "continuous":
        raise SystemExit("continuous action-set is reserved for SAC evaluation")
    loader = load_navigation_ppo if args.kind == "ppo" else model_cls.load
    model = loader(args.model, device=args.device)
    critic = critic_options(model)
    if args.centralized_critic and not critic["centralized_critic"]:
        parser.error("checkpoint does not contain a centralized critic")
    flow = "flow" in model.observation_space.spaces
    if args.flow_feature and not flow:
        parser.error("checkpoint does not contain flow observations")
    env = IPPOAgentVecEnv(CurriculumVascularEnv(
        seed=4101, clusters=args.clusters, curriculum=False, reward_mode="navigation",
        zero_latency=True, direct_action=True, centerline_match=args.centerline_match), action_set=args.action_set,
        flow_feature=flow, safety_parity=args.safety_parity, exit_mask=False,
        **critic)
    # Holdout rows are permitted only because this is an explicit evaluation
    # runner. They are never used by the training environment or selector.
    env.physical.train_anatomies = anatomies
    from stable_baselines3.common.utils import check_for_correct_spaces
    completed = []
    try:
        check_for_correct_spaces(env, model.observation_space, model.action_space)
        for local_index, row in enumerate(rows):
            obs = env.reset(options=reset_options(row))
            scene = env.reset_infos[0]["scene"]
            targets = env.physical.nav_targets.tolist()
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, infos = env.step(action)
                if bool(done[0]):
                    info = infos[0]
                    metrics = dict(info.get("episode_metrics", {}))
                    completed.append({
                        "manifest_index": int(selected_indices[local_index]),
                        "integration_failure": bool(info.get("integration_failure")),
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
                        "accepted_seed": scene["accepted_seed"],
                        "realized_scenario_hash": scene["scenario_hash"],
                        "navigation_targets": targets,
                        "navigation_success": bool(info.get("navigation_success", metrics.get("navigation_success", False))),
                        "strict_success": bool(metrics.get("strict_success", False)),
                        "wall_after_reach_s": float(metrics["wall_after_reach_s"]),
                        "wall_contact_s": float(metrics.get("wall_contact_s", np.asarray(info.get("agent_wall_contact_s", 0.)).sum())),
                        "lost": int(metrics.get("lost", np.asarray(info.get("agent_lost", 0.)).sum())),
                        "particle_collision_events": float(metrics.get("particle_collision_events", np.asarray(info.get("agent_particle_collision_events", 0.)).sum())),
                        "particle_contact_s": float(metrics.get("particle_contact_s", np.asarray(info.get("agent_particle_contact_s", 0.)).sum())),
                    })
                    if paired_entries is not None:
                        expected = paired_entries[local_index]
                        checks = {
                            "accepted_seed": expected.get("pursuit_accepted_seed"),
                            "realized_scenario_hash": expected.get("pursuit_realized_scenario_hash"),
                            "navigation_targets": expected.get("pursuit_navigation_targets"),
                        }
                        actual = completed[-1]
                        mismatches = {key: (value, actual.get(key)) for key, value in checks.items()
                                      if value is not None and value != actual.get(key)}
                        if mismatches:
                            raise RuntimeError(f"paired scene identity mismatch at row {local_index}: {mismatches}")
                    break
    finally:
        env.close()
    result = {
        "dataset_version": "navigation_dataset_v1",
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": manifest_sha,
        "model": str(args.model),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "algorithm": args.kind,
        "flow_feature": flow,
        **critic,
        "action_set": args.action_set,
        "centerline_match": args.centerline_match,
        "safety_parity": args.safety_parity,
        "exit_mask": False,
        "split": args.split,
        "clusters": args.clusters,
        "scene_list": str(args.scene_list) if args.scene_list else None,
        "protocol": "frozen_manifest_evaluation; holdout rows are never used for selection",
        "overall": aggregate(completed),
        **grouped_aggregates(completed),
        "episodes": completed,
    }
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["overall"], indent=2, ensure_ascii=False))
    if not result["overall"]["complete"]:
        raise SystemExit("Incomplete manifest evaluation. See integration failures in the output.")


if __name__ == "__main__":
    main()
