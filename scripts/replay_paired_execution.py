"""Replay frozen navigation scenes and emit per-step execution traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.navigation_dataset import iter_rows, reset_options
from scripts.train_sb3_curriculum import CurriculumVascularEnv
from scripts.train_sb3_ippo import IPPOAgentVecEnv, load_navigation_ppo


def rows_by_seed(split):
    return {int(row["scene_seed"]): row for row in iter_rows(split=split, clusters=3)}


def fixed_row(row):
    return dict(row, seed=int(row["scene_seed"]), flow=float(row.get("nominal_flow_inlet_mm_s", .025)), latency=0., difficulty=0,
                scenario_family="uniform", dynamic_obstacle_count=int(row.get("dynamic_obstacle_count", 0)))


def replay_learned(model_path, row, out):
    physical = CurriculumVascularEnv(seed=int(row["scene_seed"]), clusters=3,
                                     curriculum=False, reward_mode="navigation", zero_latency=True,
                                     direct_action=True, dataset_rows=[row])
    physical.train_anatomies = [row["anatomy"]]
    env = IPPOAgentVecEnv(physical, action_set="semantic_reduced", flow_feature=True,
                          safety_parity=True, centralized_critic=model_path.name.startswith("mappo_"))
    try:
        model = load_navigation_ppo(model_path, env=env, device="cpu")
        obs = env.reset(options={"dataset_row": row})
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action)
            if bool(done[0]):
                info = infos[0]
                return {"method": out, "scene_seed": int(row["scene_seed"]),
                        "scenario_hash": row["scenario_hash"],
                        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                        "episode_metrics": info.get("episode_metrics", {}),
                        "trace": info.get("execution_trace", [])}
    finally:
        env.close()


def replay_pursuit(row):
    physical = CurriculumVascularEnv(seed=int(row["scene_seed"]), clusters=3,
                                     curriculum=False, reward_mode="navigation", zero_latency=True,
                                     direct_action=True, dataset_rows=[row])
    physical.train_anatomies = [row["anatomy"]]
    try:
        obs, _ = physical.reset(options={"dataset_row": row})
        trace = []
        while True:
            rule = np.asarray(physical.current_rule, dtype=float).reshape(3, 3)
            before = np.asarray(physical.ep.env.positions_mm[:3]).copy()
            physical.step(rule.reshape(-1))
            info = physical.ep.info or {}
            trace.append({"elapsed_s": float(physical.ep.env.elapsed_s), "action_indices": None,
                          "candidate_commands_local": None, "requested_local_velocity": rule.tolist(),
                          "wallguard_settle_local_velocity": rule.tolist(), "sdf_projected_local_velocity": rule.tolist(),
                          "tpg_hold": [False, False, False],
                          "final_sent_command_world": np.asarray(physical.ep.sent_world).tolist(),
                          "position_mm_world": np.asarray(physical.ep.env.positions_mm[:3]).tolist(),
                          "distance_to_wall_mm": np.asarray(getattr(physical.current[0], "radial", [np.nan] * 3)).tolist(),
                          "distance_to_target_mm": np.asarray(physical._navigation_distances()).tolist(),
                          "wall_contact_s_step": np.asarray(info.get("wall_contact_s", [0., 0., 0.])).tolist(),
                          "navigation_reached": int(physical.nav_reached.sum()),
                          "navigation_success": bool(np.all(physical.nav_reached)),
                          "step_wall_contact": bool(np.asarray(info.get("wall_contact_s", 0.)).sum() > 0.),
                          "position_delta_mm": (np.asarray(physical.ep.env.positions_mm[:3]) - before).tolist()})
            if physical.ended:
                break
        return {"method": "pursuit", "scene_seed": int(row["scene_seed"]),
                "scenario_hash": row["scenario_hash"], "model_sha256": None,
                "episode_metrics": physical.ep.info, "trace": trace}
    finally:
        physical.close()


def plot_trace(records, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for record in records:
        trace = record["trace"]
        t = [x["elapsed_s"] for x in trace]
        wall = [float(sum(x.get("wall_contact_s_step", []))) for x in trace]
        distance = [float(np.nanmin(x.get("distance_to_target_mm", [np.nan]))) for x in trace]
        axes[0].plot(t, distance, label=record["method"])
        axes[1].plot(t, np.cumsum(wall), label=record["method"])
    axes[0].set_ylabel("min target distance (mm)")
    axes[1].set_ylabel("cumulative wall contact (s)"); axes[1].set_xlabel("time (s)")
    axes[0].legend(); axes[1].legend(); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("train", "holdout_test"), default="train")
    p.add_argument("--scene-seeds", required=True, help="comma-separated frozen scene_seed values")
    p.add_argument("--ippo-model", type=Path, required=True)
    p.add_argument("--mappo-model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--horizon", type=float, default=300.)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(1)
    catalogue = rows_by_seed(a.split)
    seeds = [int(x) for x in a.scene_seeds.split(",")]
    missing = [seed for seed in seeds if seed not in catalogue]
    if missing:
        raise SystemExit(json.dumps({"missing_scene_seeds": missing}))
    all_records = []
    for seed in seeds:
        row = catalogue[seed]
        records = [replay_pursuit(row), replay_learned(a.ippo_model, row, "IPPO-S41"), replay_learned(a.mappo_model, row, "MAPPO-S41")]
        all_records.extend(records)
        (a.out / f"scene_{seed}.json").write_text(json.dumps({"row": row, "records": records}, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x) + "\n")
        plot_trace(records, a.out / f"scene_{seed}.png")
    summary = {"split": a.split, "scene_seeds": seeds, "records": [{
        "method": x["method"], "scene_seed": x["scene_seed"], "scenario_hash": x["scenario_hash"],
        "steps": len(x["trace"]),
        "navigation_success": x["episode_metrics"].get("navigation_success", x["episode_metrics"].get("is_success", x["trace"][-1].get("navigation_success") if x["trace"] else None)),
        "wall_contact_s": float(np.asarray(x["episode_metrics"].get("wall_contact_s", sum(sum(t.get("wall_contact_s_step", [])) for t in x["trace"])), dtype=float).sum())
    } for x in all_records], "models": {"ippo": str(a.ippo_model), "mappo": str(a.mappo_model)}}
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x) + "\n")


if __name__ == "__main__":
    main()
