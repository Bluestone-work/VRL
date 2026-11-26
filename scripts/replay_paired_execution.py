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

from scripts.navigation_dataset import iter_rows
from scripts.evaluate_classical_manifest import run_row
from stable_baselines3.common.utils import check_for_correct_spaces
from scripts.train_sb3_curriculum import CurriculumVascularEnv
from scripts.train_sb3_ippo import IPPOAgentVecEnv, load_navigation_ppo, critic_options


def rows_by_seed(split):
    return {int(row["scene_seed"]): row for row in iter_rows(split=split, clusters=3)}


def replay_learned(model_path, row, out, horizon=300.):
    physical = CurriculumVascularEnv(seed=int(row["scene_seed"]), clusters=3,
                                     curriculum=False, reward_mode="navigation", zero_latency=True,
                                     direct_action=True, dataset_rows=[row], horizon=horizon)
    physical.train_anatomies = [row["anatomy"]]
    model = load_navigation_ppo(model_path, device="cpu")
    critic = critic_options(model)
    if critic["centralized_critic"] != (out == "MAPPO"):
        raise ValueError(f"{out} checkpoint has the wrong critic type")
    env = IPPOAgentVecEnv(physical, action_set="semantic_reduced",
                          flow_feature="flow" in model.observation_space.spaces,
                          safety_parity=True, record_trace=True, **critic)
    try:
        check_for_correct_spaces(env, model.observation_space, model.action_space)
        obs = env.reset(options={"dataset_row": row})
        scene = env.reset_infos[0]["scene"]
        targets = physical.nav_targets.tolist()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action)
            if bool(done[0]):
                info = infos[0]
                return {"method": out, "scene_seed": int(row["scene_seed"]),
                        "scenario_hash": row["scenario_hash"],
                        "accepted_seed": scene["accepted_seed"],
                        "realized_scenario_hash": scene["scenario_hash"], "navigation_targets": targets,
                        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                        "episode_metrics": info.get("episode_metrics", {}),
                        "trace": info.get("execution_trace", [])}
    finally:
        env.close()


def replay_pursuit(row, horizon=300.):
    physical = CurriculumVascularEnv(seed=int(row["scene_seed"]), clusters=3,
                                     curriculum=False, reward_mode="navigation", zero_latency=True,
                                     direct_action=True, dataset_rows=[row], horizon=horizon)
    physical.train_anatomies = [row["anatomy"]]
    try:
        trace = []
        metrics = run_row(physical, "pursuit", row, trace=trace)
        return {"method": "pursuit", "scene_seed": row["scene_seed"],
                "scenario_hash": row["scenario_hash"], "model_sha256": None,
                **{k: metrics[k] for k in ("accepted_seed", "realized_scenario_hash", "navigation_targets")},
                "episode_metrics": metrics, "trace": trace}
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
    summary_rows = []
    for seed in seeds:
        row = catalogue[seed]
        records = [replay_pursuit(row, a.horizon),
                   replay_learned(a.ippo_model, row, "IPPO", a.horizon),
                   replay_learned(a.mappo_model, row, "MAPPO", a.horizon)]
        for rec in records:
            for key in ("accepted_seed", "realized_scenario_hash", "navigation_targets"):
                if rec[key] != records[0][key]:
                    raise RuntimeError(f"replay identity mismatch for {seed}, {key}")
            trace_wall = sum(sum(step["wall_contact_s_step"]) for step in rec["trace"])
            metrics = rec["episode_metrics"]
            if metrics.get("integration_failure") or not np.isclose(trace_wall, metrics["wall_contact_s"]):
                raise RuntimeError(f"incomplete or inconsistent replay for {seed}, {rec['method']}")
            summary_rows.append({**{k: rec[k] for k in ("method", "scene_seed", "scenario_hash", "model_sha256")},
                                 "steps": len(rec["trace"]), "wall_contact_s": trace_wall,
                                 "navigation_success": metrics["navigation_success"]})
        (a.out / f"scene_{seed}.json").write_text(json.dumps({"row": row, "records": records}, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x) + "\n")
        plot_trace(records, a.out / f"scene_{seed}.png")
    summary = {"split": a.split, "scene_seeds": seeds, "horizon_s": a.horizon, "records": summary_rows,
               "models": {"ippo": str(a.ippo_model), "mappo": str(a.mappo_model)}}
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x) + "\n")


if __name__ == "__main__":
    main()
