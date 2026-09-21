"""Evaluate EXP_0005 direct-local policies on the sealed prospective validation manifest.

The evaluator exposes both Frenet policy actions and world-frame actions so the
action path, failure mode, and learned flow response can be audited directly.
The manifest's test split is never opened by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from marl.mappo_advanced import MAPPOAdvanced
from scripts.research_protocol import geometry_hash, initial_identity, trace_hash
from scripts.train_vector_mappo import build_context


METRICS = (
    "success", "removal_rate", "steps", "completion_steps", "first_contact_steps",
    "wall_contact_rate", "collision_rate", "mean_action_magnitude",
    "mean_flow_speed", "blood_flow_exposure", "mean_simultaneously_contacted_clots",
)
HARD_SCENARIOS = ("femoropopliteal_pad", "coronary_rca", "carotid_bifurcation")


def read(path: Path):
    return json.loads(path.read_text())


def environment_settings(cfg):
    return dict(
        num_robots=cfg["robots"], num_clots=cfg["clots"], horizon=cfg["horizon"],
        robot_radius=cfg["robot_radius"], obs_mode=cfg["obs_mode"],
        reward_mode=cfg["reward_mode"], contact_mode=cfg["contact_mode"],
        coverage_bonus=cfg["coverage_bonus"], step_cost=cfg["step_cost"],
        approach_scale=cfg["approach_scale"],
        reward_double_count=cfg["reward_double_count"],
        control_margin=not cfg["no_control_margin"], randomize_scenario=False,
    )


def build_agent(checkpoint: Path, cfg, device: str, expected_mode: str = "local"):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = saved["meta"]
    if meta.get("control_mode") != expected_mode:
        raise AssertionError(f"evaluation expected {expected_mode!r}, received {meta.get('control_mode')!r}")
    if expected_mode == "local" and meta.get("action_semantics") != "direct_local_frenet":
        raise AssertionError("checkpoint is missing direct_local_frenet semantics")
    agent = MAPPOAdvanced(
        n_agents=meta["n_agents"], obs_dim=meta["obs_dim"], action_dim=meta["action_dim"],
        architecture=meta["architecture"], state_dim=meta.get("state_dim", 0),
        hidden_dim=meta.get("hidden_dim", 128), num_layers=meta.get("num_layers", 2),
        num_heads=meta.get("num_heads", 4), sparse_k=meta.get("sparse_k", 8),
        control_mode=expected_mode, residual_scale=meta.get("residual_scale", 0.2),
        guidance_speed=meta.get("guidance_speed", 0.65),
        critic_value_mode=meta.get("critic_value_mode", "v"),
        dropout=meta.get("dropout", 0.0), device=device,
    )
    agent.load(checkpoint, load_optimizers=False)
    agent.actor.eval(); agent.critic.eval()
    return agent


def episode(agent, env, target, trace_path: Path):
    identity = initial_identity(env, target["episode_seed"])
    for key in ("complete_geometry_sha256", "initial_state_sha256", "active_clots", "stations", "branches"):
        if identity[key] != target[key]:
            raise AssertionError(f"prospective identity mismatch for {target['scenario']}/{target['episode_seed']}: {key}")
    obs, _ = env.reset(seed=target["episode_seed"])
    initial = env.clot_initial_mass.copy()
    n, active = env.num_robots, env.active_clots
    arrays = {key: [] for key in (
        "policy_action", "executed_action", "flow", "active_per_clot", "rewards",
        "wall_contacts", "collision_pairs", "robot_clot_distances", "clot_masses",
        "flow_local",
    )}
    arrays["clot_masses"].append(env.clot_masses.copy())
    arrays["positions"] = [env.robot_positions.copy()]
    for step in range(env.horizon):
        ctx = build_context(env, obs)
        state = obs["clot_state"].reshape(-1)
        local_action, _, _ = agent.act(obs["nodes"], ctx, state, deterministic=True)
        executed = agent.env_action(local_action, obs, env)
        stations = env.robot_stations
        flow = env.tree.flow(env.robot_positions, stations, env.flow_speed, env.tube_radius,
                             radius_override=env._occluded_radius(stations))
        tangent, normal, binormal = (env.tree.tangents[stations], env.tree.normals[stations],
                                     env.tree.binormals[stations])
        flow_local = np.stack((np.sum(flow * tangent, axis=1), np.sum(flow * normal, axis=1),
                               np.sum(flow * binormal, axis=1)), axis=1)
        alive_before = env.clot_masses[:active] > 0
        obs, reward, terminated, truncated, info = env.step(executed)
        distances = np.linalg.norm(env.robot_positions[:, None, :] - env.clot_positions[None, :, :], axis=-1)
        contact = (distances[:, :active] <= env.clot_contact_radius) & alive_before[None, :]
        if env.contact_mode == "geodesic":
            from environments.contact_geometry import continuous_route_distance
            geo = np.stack([
                continuous_route_distance(env.tree, env.robot_positions, env.robot_stations,
                                          *env._route(c)) for c in range(active)
            ], axis=-1)
            contact &= geo <= env.clot_contact_radius
        arrays["policy_action"].append(local_action.copy())
        arrays["executed_action"].append(executed.copy())
        arrays["flow"].append(flow.copy())
        arrays["flow_local"].append(flow_local)
        arrays["active_per_clot"].append(contact.sum(axis=0))
        arrays["rewards"].append(reward)
        arrays["wall_contacts"].append(info["wall_collisions"])
        arrays["collision_pairs"].append(info["robot_collisions"])
        arrays["robot_clot_distances"].append(distances[:, :active])
        arrays["clot_masses"].append(env.clot_masses.copy())
        arrays["positions"].append(env.robot_positions.copy())
        if terminated or truncated:
            break
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("non-finite direct-local trace")
    steps = int(arrays["policy_action"].shape[0])
    success = bool(info["success"])
    contact_steps = arrays["active_per_clot"].sum(axis=0)
    total_contact = int(contact_steps.sum())
    wall_rate = float(arrays["wall_contacts"].sum() / (n * steps))
    pair_rate = float(arrays["collision_pairs"].sum() / (max(n * (n - 1) // 2, 1) * steps))
    removal = float(info["removal_rate"])
    labels = []
    if not success:
        if total_contact == 0:
            labels.append("navigation_failure")
        elif total_contact < max(3, int(0.05 * steps * active)):
            labels.append("contact_acquisition_failure")
        if total_contact and total_contact / max(steps * active, 1) < 0.2:
            labels.append("contact_maintenance_failure")
        if removal > 0.9:
            labels.append("completion_tail_failure")
        if wall_rate >= 0.5:
            labels.append("wall_dominated_failure")
        if pair_rate >= 0.1:
            labels.append("robot_crowding_failure")
        if not labels and steps >= env.horizon:
            labels.append("timeout")
    flow_mag = np.linalg.norm(arrays["flow"], axis=-1)
    local_mag = np.linalg.norm(arrays["policy_action"], axis=-1)
    flow_dir = arrays["flow_local"] / np.maximum(flow_mag[..., None], 1e-8)
    action_flow_dot = np.sum(arrays["policy_action"] * flow_dir, axis=-1)
    record = {
        "scenario": target["scenario"], "episode_seed": int(target["episode_seed"]),
        "success": float(success), "removal_rate": removal, "steps": steps,
        "completion_steps": steps if success else None,
        "first_contact_steps": int(np.flatnonzero(arrays["active_per_clot"].sum(axis=1) > 0)[0] + 1)
        if total_contact else None,
        "wall_contact_rate": wall_rate, "collision_rate": pair_rate,
        "mean_action_magnitude": float(local_mag.mean()),
        "mean_flow_speed": float(flow_mag.mean()),
        "blood_flow_exposure": float(flow_mag.sum(axis=0).mean()),
        "mean_simultaneously_contacted_clots": float((arrays["active_per_clot"] > 0).sum(axis=1).mean()),
        "labels": labels, "contact_steps_total": total_contact,
        "policy_action_flow_dot_mean": float(action_flow_dot.mean()),
        "geometry_sha256": geometry_hash(env.tree),
        "trace": trace_path.name,
    }
    np.savez_compressed(trace_path, **arrays, clot_initial_mass=initial)
    record["trace_arrays_sha256"] = trace_hash(trace_path)
    return record


def mean_sd(rows, key):
    values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=float)
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n": int(len(values))}


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="directory containing seed_42/final_policy.pt, ...")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("local", "flow_guided"), default="local")
    parser.add_argument("--checkpoint-glob", default="",
                        help="optional glob with {seed}, used for a fixed reference arm")
    args = parser.parse_args()
    root = args.project_root.resolve(); study = read(args.config.resolve())
    manifest = read(root / study["prospective_validation"]["manifest"])
    if manifest.get("policy_evaluations") != 0:
        raise AssertionError("prospective manifest was evaluated before EXP_0005")
    records_by_scenario = {}
    for row in manifest["records"]:
        records_by_scenario.setdefault(row["scenario"], []).append(row)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    all_summaries, all_records = [], []
    cfg = study["training"]
    for seed in study["seeds"]:
        seed_dir = output / f"seed_{seed}"; seed_dir.mkdir()
        traces = seed_dir / "traces"; traces.mkdir()
        if args.checkpoint_glob:
            matches = sorted(Path().glob(args.checkpoint_glob.format(seed=seed)))
            if len(matches) != 1:
                raise AssertionError(f"expected one checkpoint for seed {seed}, found {matches}")
            checkpoint = matches[0].resolve()
        else:
            checkpoint = args.run_root.resolve() / f"seed_{seed}" / "final_policy.pt"
        agent = None
        per_seed = []
        for scenario, targets in sorted(records_by_scenario.items()):
            env = Vascular3DMARLEnv(scenario=scenario, scenario_pool=[scenario], **environment_settings(cfg))
            try:
                if agent is None:
                    agent = build_agent(checkpoint, cfg, args.device, args.mode)
                for target in targets:
                    trace = traces / f"{scenario}_{target['episode_seed']}.npz"
                    record = episode(agent, env, target, trace)
                    record["training_seed"] = seed
                    per_seed.append(record); all_records.append(record)
            finally:
                env.close()
        summary = {"training_seed": seed, **{key: mean_sd(per_seed, key) for key in METRICS},
                   "failure_labels": {label: sum(label in row["labels"] for row in per_seed)
                                      for label in ("navigation_failure", "contact_acquisition_failure",
                                                    "contact_maintenance_failure", "completion_tail_failure",
                                                    "wall_dominated_failure", "robot_crowding_failure", "timeout")}}
        (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        all_summaries.append(summary)
        write_csv(seed_dir / "episodes.csv", per_seed)
    aggregate = {key: {"mean": float(np.mean([summary[key]["mean"] for summary in all_summaries])),
                        "sample_sd": float(np.std([summary[key]["mean"] for summary in all_summaries], ddof=1))}
                 for key in METRICS}
    per_scenario = []
    for scenario in sorted(records_by_scenario):
        selected = [row for row in all_records if row["scenario"] == scenario]
        per_scenario.append({"scenario": scenario, "episodes": len(selected),
                             **{key: float(np.mean([row[key] for row in selected])) for key in METRICS if key not in ("completion_steps", "first_contact_steps")}})
    flow_rows = []
    for row in all_records:
        with np.load(output / f"seed_{row['training_seed']}" / "traces" / row["trace"]) as trace:
            mag = np.linalg.norm(trace["flow"], axis=-1).reshape(-1)
            act = trace["policy_action"].reshape(-1, 3)
            flow_local = trace["flow_local"].reshape(-1, 3)
            dot = np.sum(act * flow_local / np.maximum(np.linalg.norm(flow_local, axis=1, keepdims=True), 1e-8), axis=1)
            for value, action, scalar in zip(mag, act, dot):
                flow_rows.append({"flow_magnitude": float(value), "u_parallel": float(action[0]),
                                  "u_normal": float(action[1]), "u_binormal": float(action[2]),
                                  "policy_action_flow_dot": float(scalar)})
    q1, q2 = np.quantile([row["flow_magnitude"] for row in flow_rows], [1/3, 2/3])
    flow_summary = {}
    for name, lo, hi in (("low", -np.inf, q1), ("medium", q1, q2), ("high", q2, np.inf)):
        selected = [row for row in flow_rows if lo <= row["flow_magnitude"] < hi]
        flow_summary[name] = {key: float(np.mean([row[key] for row in selected]))
                              for key in ("flow_magnitude", "u_parallel", "u_normal", "u_binormal", "policy_action_flow_dot")}
    result = {"experiment_id": study["experiment_id"], "split": "prospective_validation",
              "sealed_test_accessed": False, "episodes": len(all_records),
              "aggregate_by_training_seed": aggregate, "per_seed": all_summaries,
              "per_scenario": per_scenario, "hard_scenarios": [row for row in per_scenario if row["scenario"] in HARD_SCENARIOS],
              "flow_bins": {"thresholds": [float(q1), float(q2)], "summary": flow_summary},
              "reference_baseline": study["source_baseline"]}
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    write_csv(output / "per_scenario.csv", per_scenario)
    write_csv(output / "flow_action_samples.csv", flow_rows)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter([row["flow_magnitude"] for row in flow_rows], [row["u_parallel"] for row in flow_rows], s=2, alpha=.15)
    axes[0].set(xlabel="flow magnitude", ylabel="u_parallel", title="Flow magnitude vs local axial action")
    axes[1].scatter([row["policy_action_flow_dot"] for row in flow_rows], [row["flow_magnitude"] for row in flow_rows], s=2, alpha=.15)
    axes[1].set(xlabel="policy action dot flow direction", ylabel="flow magnitude", title="Action alignment with flow")
    figure.tight_layout(); figure.savefig(output / "flow_action_diagnostics.png", dpi=160); plt.close(figure)
    print(json.dumps({"episodes": len(all_records), "aggregate": aggregate, "flow_bins": flow_summary}, indent=2))


if __name__ == "__main__":
    main()
