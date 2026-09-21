"""Read-only, configuration-matched diagnostics for frozen final policies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from environments.vessel_geometry import resolve_pool
from marl.policy_loader import load_policy


def environment_settings(config):
    return dict(num_robots=config["robots"], num_clots=config["clots"], horizon=config["horizon"],
                robot_radius=config["robot_radius"], obs_mode=config["obs_mode"],
                reward_mode=config["reward_mode"], contact_mode=config["contact_mode"],
                coverage_bonus=config["coverage_bonus"], step_cost=config["step_cost"],
                approach_scale=config["approach_scale"], reward_double_count=config["reward_double_count"],
                control_margin=not config["no_control_margin"], randomize_scenario=False)


def validate_checkpoint(meta, config):
    expected = {key: config[key] for key in ("architecture", "control_mode", "residual_scale", "guidance_speed",
                "critic_value_mode", "obs_mode", "contact_mode", "reward_double_count", "coverage_bonus",
                "step_cost", "approach_scale", "no_control_margin")}
    expected["n_agents"] = config["robots"]
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Checkpoint/config mismatch for {key}: {meta.get(key)!r} != {value!r}")


def episode(policy, env, episode_seed, trace_path=None):
    obs, _ = env.reset(seed=episode_seed)
    n = env.num_robots
    initial = env.clot_initial_mass.copy()
    arrays = {key: [] for key in ("actions", "flow", "occluded_radius", "assignments", "active_per_clot",
                                  "rewards", "wall_contacts", "collision_pairs", "robot_clot_distances")}
    arrays["positions"] = [env.robot_positions.copy()]
    arrays["velocities"] = [env.robot_velocities.copy()]
    arrays["clot_masses"] = [env.clot_masses.copy()]
    inference_seconds = 0.0
    started = time.perf_counter()
    for step in range(env.horizon):
        assignments = env._assigned_clot().copy()
        occluded = env._occluded_radius(env.robot_stations)
        flow = env.tree.flow(env.robot_positions, env.robot_stations, env.flow_speed,
                             env.tube_radius, radius_override=occluded)
        alive_before = env.clot_masses > 0
        tick = time.perf_counter()
        action = policy(obs, env)
        inference_seconds += time.perf_counter() - tick
        obs, reward, terminated, truncated, info = env.step(action)
        distances = np.linalg.norm(env.robot_positions[:, None, :] - env.clot_positions[None, :, :], axis=-1)
        contact = (distances <= env.clot_contact_radius) & alive_before[None, :]
        # Use the pre-step alive mask. Fully cleared clots still count as active contacts on their final step.
        if env.contact_mode == "geodesic":
            from environments.contact_geometry import continuous_route_distance
            geo = np.stack([continuous_route_distance(env.tree, env.robot_positions, env.robot_stations,
                            *env._route(c)) for c in range(env.active_clots)], axis=-1)
            contact &= geo <= env.clot_contact_radius
        arrays["actions"].append(action.copy())
        arrays["flow"].append(flow)
        arrays["occluded_radius"].append(occluded)
        arrays["assignments"].append(assignments)
        arrays["active_per_clot"].append(contact.sum(axis=0))
        arrays["robot_clot_distances"].append(distances)
        arrays["rewards"].append(reward)
        arrays["wall_contacts"].append(info["wall_collisions"])
        arrays["collision_pairs"].append(info["robot_collisions"])
        arrays["positions"].append(env.robot_positions.copy())
        arrays["velocities"].append(env.robot_velocities.copy())
        arrays["clot_masses"].append(env.clot_masses.copy())
        if terminated or truncated:
            break
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("Non-finite diagnostic trace")
    steps = step + 1
    success = bool(info["success"])
    assert success == bool(np.all(env.clot_masses <= 0))
    assert np.all(np.diff(arrays["clot_masses"], axis=0) <= 1e-7)
    record = {
        "episode_seed": int(episode_seed), "scenario": env.active_scenario, "success": float(success),
        "completion_rate": float(success), "removal_rate": float(info["removal_rate"]),
        "steps": steps, "completion_steps": steps if success else None,
        "first_contact_steps": int(info["first_contact_step"]) + 1 if info["first_contact_step"] >= 0 else None,
        "contact_miss": float(info["first_contact_step"] < 0), "return": float(arrays["rewards"].sum()),
        "wall_hits_total": int(arrays["wall_contacts"].sum()),
        "wall_hits_per_step": float(arrays["wall_contacts"].sum() / steps),
        "wall_contact_rate": float(arrays["wall_contacts"].sum() / (n * steps)),
        "robot_collisions_total": int(arrays["collision_pairs"].sum()),
        "collision_rate": float(arrays["collision_pairs"].sum() / (max(n * (n - 1) // 2, 1) * steps)),
        "mean_flow_speed": float(np.linalg.norm(arrays["flow"], axis=-1).mean()),
        "blood_flow_exposure": float(np.linalg.norm(arrays["flow"], axis=-1).sum(axis=0).mean()),
        "mean_action_magnitude": float(np.linalg.norm(arrays["actions"], axis=-1).mean()),
        "action_energy_proxy": float(np.square(arrays["actions"]).sum()),
        "mean_robot_clot_distance_all_pairs": float(arrays["robot_clot_distances"].mean()),
        "allocation_counts": np.bincount(arrays["assignments"][arrays["assignments"] >= 0], minlength=env.active_clots).tolist(),
        "inference_ms_per_step": 1000 * inference_seconds / steps,
        "evaluation_seconds": time.perf_counter() - started,
        "geometry_sha256": hashlib.sha256(env.tree.points.tobytes() + env.tree.radii.tobytes()
                                           + env.tree.branch_ids.tobytes()).hexdigest(),
        "trace": str(trace_path.name) if trace_path else None,
    }
    if trace_path:
        if trace_path.exists():
            raise RuntimeError("Refusing to overwrite trace")
        np.savez_compressed(trace_path, **arrays, tree_points=env.tree.points, tree_radii=env.tree.radii,
                            tree_branch_ids=env.tree.branch_ids, clot_positions=env.clot_positions,
                            clot_initial_mass=initial)
    return record


METRICS = ("success", "completion_rate", "removal_rate", "steps", "completion_steps", "first_contact_steps",
           "contact_miss", "return", "wall_hits_total", "wall_hits_per_step", "wall_contact_rate",
           "robot_collisions_total", "collision_rate", "mean_flow_speed", "blood_flow_exposure",
           "mean_action_magnitude", "action_energy_proxy", "mean_robot_clot_distance_all_pairs", "inference_ms_per_step")


def summarize(records):
    def means(rows):
        out = {}
        for key in METRICS:
            values = [row[key] for row in rows if row[key] is not None]
            out[key] = float(np.mean(values)) if values else None
        return out
    territories = {name: means([r for r in records if r["scenario"] == name])
                   for name in sorted({r["scenario"] for r in records})}
    # Conditional time metrics are episode-weighted over successes/contacts; others use territory macro means.
    macro = means(list(territories.values()))
    conditional = means(records)
    for key in ("completion_steps", "first_contact_steps"):
        macro[key] = conditional[key]
    return {"macro": macro, "per_territory": territories, "episodes": len(records),
            "successful_episodes": int(sum(r["success"] for r in records))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((args.run / "config.json").read_text())
    study = json.loads(args.config.read_text())
    evaluation = study["diagnostic_evaluation"]
    checkpoint = args.run / evaluation["checkpoint"]
    meta = torch.load(checkpoint, map_location="cpu", weights_only=False)["meta"]
    validate_checkpoint(meta, config)
    args.output.mkdir(parents=True, exist_ok=True)
    trace_directory = args.output / "traces"
    trace_directory.mkdir(exist_ok=True)
    records_path = args.output / "episodes.jsonl"
    if records_path.exists():
        raise RuntimeError("Evaluation already started; preserve it and inspect before recovery")
    (args.output / "protocol.json").write_text(json.dumps({"settings": evaluation, "environment": environment_settings(config),
        "policy_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "metric_note": "Event-normalized rates; action energy is a proxy; no clinical units; development validation only"}, indent=2))
    records = []
    with records_path.open("x") as log:
        for index, scenario in enumerate(resolve_pool(config["scenario_pool"])):
            env = Vascular3DMARLEnv(scenario=scenario, scenario_pool=[scenario], **environment_settings(config))
            try:
                policy = load_policy(str(checkpoint), env, device=evaluation["device"])
                for offset in range(evaluation["episodes_per_scenario"]):
                    episode_seed = evaluation["seed_base"] + index * 10000 + offset
                    path = trace_directory / f"{scenario}_{episode_seed}.npz"
                    record = episode(policy, env, episode_seed, path if evaluation["record_traces"] else None)
                    records.append(record)
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                print(json.dumps({"scenario": scenario, "episodes_done": len(records)}), flush=True)
            finally:
                env.close()
    summary = summarize(records)
    summary["split"] = evaluation["split"]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary["macro"], indent=2), flush=True)


if __name__ == "__main__":
    main()
