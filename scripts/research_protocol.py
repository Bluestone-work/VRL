"""EXP_0002 read-only evaluation and content-identity checks.

The original environment, policy loader and diagnostic episode are reused unchanged.
No model is invoked while constructing prospective train/validation/test manifests.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from environments.vascular_3d_marl_env import Vascular3DMARLEnv
from environments.vessel_geometry import resolve_pool
from scripts.research_evaluate import environment_settings, validate_checkpoint, episode, summarize, METRICS
from marl.policy_loader import load_policy


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def content_hash(arrays, metadata=None):
    """Explicit field names, shapes and endian-normalized typed bytes; no rounding."""
    value = hashlib.sha256()
    value.update(json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    for name, array in sorted(arrays.items()):
        array = np.asarray(array)
        if array.dtype.hasobject:
            raise TypeError("Object arrays have no stable content identity")
        array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
        header = json.dumps([name, list(array.shape), array.dtype.str], separators=(",", ":")).encode()
        value.update(len(header).to_bytes(8, "little"))
        value.update(header)
        value.update(array.tobytes())
    return value.hexdigest()


def geometry_hash(tree):
    graph = sorted((i, int(j), float(weight)) for i, neighbors in enumerate(tree.station_graph)
                   for j, weight in neighbors)
    arrays = {name: getattr(tree, name) for name in
              ("points", "radii", "branch_ids", "tangents", "normals", "binormals", "arclength")}
    return content_hash(arrays, {"schema": "vascular_geometry_v1", "inlet_station": tree.inlet_station,
        "branches": [asdict(branch) for branch in tree.branches], "station_graph": graph})


def trace_hash(path):
    with np.load(path) as trace:
        return content_hash({key: trace[key] for key in trace.files})


def initial_identity(env, seed):
    env.reset(seed=seed)
    geom = geometry_hash(env.tree)
    state = {name: getattr(env, name) for name in ("robot_positions", "robot_velocities", "robot_stations",
        "clot_positions", "clot_stations", "clot_masses", "clot_initial_mass")}
    return {"complete_geometry_sha256": geom,
            "initial_state_sha256": content_hash(state, {"geometry": geom, "rng": env._rng.bit_generator.state}),
            "active_clots": int(env.active_clots), "stations": env.tree.n_stations,
            "branches": len(env.tree.branches)}


def validate_protocol(meta, config, env, defaults):
    validate_checkpoint(meta, config)
    expected = {key: config[key] for key in ("hidden_dim", "num_layers", "num_heads", "dropout")}
    expected.update(obs_dim=env.observation_space["nodes"].shape[-1], action_dim=3,
                    state_dim=int(np.prod(env.observation_space["clot_state"].shape)))
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Metadata mismatch {key}: {meta.get(key)!r} != {value!r}")
    aliases = {"robots": "num_robots", "clots": "num_clots"}
    for key in ("robots", "clots", "horizon", "robot_radius", "obs_mode", "reward_mode", "contact_mode",
                "coverage_bonus", "step_cost", "approach_scale", "reward_double_count"):
        if getattr(env, aliases.get(key, key)) != config[key]:
            raise ValueError(f"Environment mismatch {key}")
    if env.control_margin != (not config["no_control_margin"]) or env.randomize_scenario:
        raise ValueError("Environment control margin or scenario randomization drift")
    for key, value in defaults.items():
        if getattr(env, key) != value:
            raise ValueError(f"Unrecorded default drift: {key}")


def semantic_equal(left, right):
    keys = [key for key in METRICS if key != "inference_ms_per_step"]
    keys += ["scenario", "episode_seed", "allocation_counts", "geometry_sha256"]
    return {key: [left[key], right[key]] for key in keys if left[key] != right[key]}


def run_evaluation(root, study, seed, device, role, destination):
    parent = root / f"research/runs/EXP_0001/training/seed_{seed}"
    cfg = read(parent / "config.json")
    checkpoint = parent / study["checkpoint"]
    meta = torch.load(checkpoint, map_location="cpu", weights_only=False)["meta"]
    settings = study["smoke"] if role == "smoke" else study["evaluation"]
    count = study["legacy_control"]["episodes_per_scenario"] if role == "legacy" else settings["episodes_per_scenario"]
    seed_base = study["legacy_control"]["seed_base"] if role == "legacy" else study["evaluation"]["seed_base"]
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    traces = dest / "traces"
    traces.mkdir(exist_ok=True)
    write_new(dest / "protocol.json", {"experiment_id": study["experiment_id"], "training_seed": seed,
        "device": device, "role": role, "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint),
        "config_sha256": file_hash(parent / "config.json"), "environment": environment_settings(cfg),
        "fixed_defaults": study["fixed_environment_defaults"], "seed_base": seed_base, "count_per_scenario": count})
    parent_records = {(r["scenario"], r["episode_seed"]): r for r in
        [json.loads(line) for line in (root / f"research/runs/EXP_0001/evaluation/seed_{seed}/episodes.jsonl").read_text().splitlines()]}
    legacy_records = {(r["scenario"], r["episode_seed"]): r for r in read(parent / "summary.json")["final_evaluation"]["episodes"]}
    records, checks = [], []
    with (dest / "episodes.jsonl").open("x") as log, (dest / "repeats.jsonl").open("x") as repeat_log:
        for index, scenario in enumerate(resolve_pool(cfg["scenario_pool"])):
            env = Vascular3DMARLEnv(scenario=scenario, scenario_pool=[scenario], **environment_settings(cfg))
            try:
                validate_protocol(meta, cfg, env, study["fixed_environment_defaults"])
                policy = load_policy(str(checkpoint), env, device=device)
                for offset in range(count):
                    episode_seed = seed_base + index * study["evaluation"]["scenario_stride"] + offset
                    identity = initial_identity(env, episode_seed)
                    trace_path = traces / f"{scenario}_{episode_seed}.npz"
                    record = episode(policy, env, episode_seed, trace_path)
                    record.update(identity, training_seed=seed, device=device, trace_arrays_sha256=trace_hash(trace_path))
                    records.append(record)
                    log.write(json.dumps(record, allow_nan=False) + "\n")
                    log.flush()
                    check = {"scenario": scenario, "episode_seed": episode_seed}
                    if role == "legacy":
                        old = legacy_records[(scenario, episode_seed)]
                        check["legacy_exact"] = all(record[k] == old[k] for k in ("success", "steps", "wall_hits_total")) and abs(record["removal_rate"] - old["removal_rate"]) <= study["gate"]["float_comparison_atol"]
                    elif device == "cpu":
                        old = parent_records[(scenario, episode_seed)]
                        check["parent_metric_differences"] = semantic_equal(record, old)
                        old_path = root / f"research/runs/EXP_0001/evaluation/seed_{seed}/traces" / old["trace"]
                        check["parent_trace_exact"] = record["trace_arrays_sha256"] == trace_hash(old_path)
                    if offset == 0 and role != "legacy" and settings["repeat_first_episode_per_scenario"]:
                        repeat_identity = initial_identity(env, episode_seed)
                        repeat_path = traces / f"{scenario}_{episode_seed}_repeat.npz"
                        repeated = episode(policy, env, episode_seed, repeat_path)
                        repeated.update(repeat_identity, training_seed=seed, device=device, trace_arrays_sha256=trace_hash(repeat_path))
                        repeat_log.write(json.dumps(repeated, allow_nan=False) + "\n")
                        repeat_log.flush()
                        check["repeat_exact"] = not semantic_equal(record, repeated) and identity == repeat_identity and record["trace_arrays_sha256"] == repeated["trace_arrays_sha256"]
                    checks.append(check)
                print(json.dumps({"seed": seed, "device": device, "role": role, "scenario": scenario, "episodes": len(records)}), flush=True)
            finally:
                env.close()
    good = all(c.get("legacy_exact", True) and c.get("parent_trace_exact", True)
               and not c.get("parent_metric_differences") and c.get("repeat_exact", True) for c in checks)
    write_new(dest / "checks.json", {"passed": good, "checks": checks})
    result = summarize(records)
    result.update(device=device, training_seed=seed, role=role)
    write_new(dest / "summary.json", result)
    if not good:
        raise AssertionError("Evaluation protocol consistency failed; all records retained")


def overlaps(groups):
    names = list(groups)
    return {f"{a}__{b}": sorted(set(groups[a]) & set(groups[b]))
            for index, a in enumerate(names) for b in names[index + 1:]}


def run_inventory(root, study, dest):
    dest.mkdir(parents=True, exist_ok=True)
    cfg = read(root / "research/runs/EXP_0001/training/seed_42/config.json")
    prospective = study["prospective_split"]
    manifests, sets = {}, {}
    for split in ("train", "validation", "test"):
        records = []
        for index, scenario in enumerate(resolve_pool(prospective["scenario_pool"])):
            env = Vascular3DMARLEnv(scenario=scenario, scenario_pool=[scenario], **environment_settings(cfg))
            try:
                for offset in range(prospective[split]["episodes_per_scenario"]):
                    episode_seed = prospective[split]["seed_base"] + index * prospective["scenario_stride"] + offset
                    records.append({"split": split, "scenario": scenario, "episode_seed": episode_seed,
                                    **initial_identity(env, episode_seed)})
            finally:
                env.close()
        manifests[split] = records
        sets[split] = [r["complete_geometry_sha256"] for r in records]
        write_new(dest / f"{split}.json", {"name": prospective["name"], "policy_evaluations": 0, "records": records})
        print(json.dumps({"split": split, "registered_geometries": len(records), "policy_evaluations": 0}), flush=True)
    validation = {"legacy_final": [], "development": []}
    for name, base in (("legacy_final", 100000), ("development", study["evaluation"]["seed_base"])):
        for index, scenario in enumerate(resolve_pool(cfg["scenario_pool"])):
            env = Vascular3DMARLEnv(scenario=scenario, scenario_pool=[scenario], **environment_settings(cfg))
            try:
                for offset in range(20):
                    episode_seed = base + index * study["evaluation"]["scenario_stride"] + offset
                    validation[name].append({"scenario": scenario, "episode_seed": episode_seed,
                                              **initial_identity(env, episode_seed)})
            finally:
                env.close()
    write_new(dest / "historical_evaluation_geometries.json", validation)
    saved, unavailable = [], []
    coverage = {}
    for seed in study["seeds"]:
        directory = root / f"research/runs/EXP_0001/training/seed_{seed}"
        by_child = {}
        for path in sorted(directory.glob("*.pt")):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            state = checkpoint.get("training_state", {})
            if "env" not in state:
                unavailable.append({"seed": seed, "checkpoint": path.name, "reason": "No stored environment state"})
                continue
            for index, child in enumerate(state["env"]["children"]):
                generation = int(child["geometry_generation"])
                by_child.setdefault(index, set()).add(generation)
                saved.append({"training_seed": seed, "checkpoint": path.name, "child": index,
                    "transitions": state["transitions"], "generation": generation,
                    "scenario": child["active_scenario"], "complete_geometry_sha256": geometry_hash(child["tree"])})
        coverage[str(seed)] = [{"child": index, "observed_generations": sorted(values),
            "missing_generations_through_last_saved": sorted(set(range(max(values) + 1)) - values)}
            for index, values in sorted(by_child.items())]
    write_new(dest / "saved_training_geometries.json", {"records": saved, "missing_environment_checkpoints": unavailable,
        "coverage": coverage, "full_training_history_available": False})
    groups = {**sets, "saved_historical_train": [r["complete_geometry_sha256"] for r in saved],
              **{name: [r["complete_geometry_sha256"] for r in values] for name, values in validation.items()}}
    overlap = overlaps(groups)
    repeats = {name: len(values) - len(set(values)) for name, values in sets.items()}
    prospective_pass = not any(repeats.values()) and not any(overlap[f"{a}__{b}"] for a, b in
        (("train", "validation"), ("train", "test"), ("validation", "test")))
    historical_overlap = {}
    for seed in study["seeds"]:
        directory = root / f"research/runs/EXP_0001/training/seed_{seed}"
        evaluations = [json.loads(line) for line in (directory / "eval_metrics.jsonl").read_text().splitlines()]
        during = {(r["scenario"], r["episode_seed"]) for item in evaluations for r in item["episodes"]}
        final = {(r["scenario"], r["episode_seed"]) for r in read(directory / "summary.json")["final_evaluation"]["episodes"]}
        historical_overlap[str(seed)] = {"selection_validation_episodes": len(during), "final_validation_episodes": len(final),
            "overlapping_episodes": len(during & final), "overlap_fraction_of_final": len(during & final) / len(final)}
    summary = {"passed": prospective_pass, "geometry_hash_schema": "vascular_geometry_v1; exact ordered content, not graph-isomorphism or near-duplicate detection",
        "counts": {name: len(values) for name, values in groups.items()},
        "unique_counts": {name: len(set(values)) for name, values in groups.items()},
        "prospective_within_split_duplicates": repeats, "overlap_counts": {key: len(value) for key, value in overlap.items()},
        "overlapping_hashes": overlap, "historical_validation_reuse": historical_overlap,
        "full_historical_train_disjointness": "NOT CERTIFIED: missing training generations",
        "prospective_test_policy_evaluations": 0, "unseen_topology_claim": False,
        "prospective_manifest_hashes": {s: file_hash(dest / f"{s}.json") for s in manifests}}
    write_new(dest / "summary.json", summary)
    if not prospective_pass:
        raise AssertionError("Prospective split repeats or overlaps; no resampling permitted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("evaluate", "inventory"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--role", choices=("smoke", "formal", "legacy"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    study = read(args.project_root / "configs/experiments/EXP_0002.json")
    if args.stage == "inventory":
        run_inventory(args.project_root, study, args.output)
    else:
        run_evaluation(args.project_root, study, args.seed, args.device, args.role, args.output)


if __name__ == "__main__":
    main()
