#!/usr/bin/env python3
"""Build and validate the registered navigation_dataset_v1 scene manifest.

The manifest is a deterministic scenario catalogue, not a pre-simulated replay
set. Each row contains all reset parameters needed by a training/evaluation
runner. Scene seeds are deliberately disjoint from configs/evaluation_splits.json
and from the previously accessed multicluster development pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/navigation_dataset_v1.json"
SPLITS = ROOT / "configs/evaluation_splits.json"
OUT_DIR = ROOT / "research/datasets/NAVIGATION_V1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_config():
    cfg = json.loads(CONFIG.read_text())
    splits = json.loads(SPLITS.read_text())
    if cfg["train_anatomies"] != splits["anatomy_holdout_v1"]["train"]:
        raise ValueError("navigation train split diverges from registered anatomy_holdout_v1")
    if cfg["holdout_test_anatomies"] != splits["anatomy_holdout_v1"]["held_out"]:
        raise ValueError("navigation holdout split diverges from registered anatomy_holdout_v1")
    if set(cfg["train_anatomies"]) & set(cfg["holdout_test_anatomies"]):
        raise ValueError("train and holdout anatomies overlap")
    if set(cfg["train_anatomies"] + cfg["holdout_test_anatomies"]) != set(splits["anatomy_order"]):
        raise ValueError("navigation split does not cover all registered anatomies")
    return cfg, splits


def condition_rows(cfg):
    flow_modes = cfg["flow_modes"]
    patterns = cfg["clot_patterns"]
    densities = cfg["obstacle_density"]
    layouts = cfg["obstacle_layouts"]
    rows = []
    condition_id = 0
    for flow_mode, flow in flow_modes.items():
        for clot_pattern, pattern in patterns.items():
            for obstacle_density, obstacle in densities.items():
                for obstacle_layout in layouts:
                    rows.append({
                        "condition_id": condition_id,
                        "flow_mode": flow_mode,
                        "nominal_flow_inlet_mm_s": flow["nominal_flow_inlet_mm_s"],
                        "variation_strength": flow["variation_strength"],
                        "pulsatile": flow["pulsatile"],
                        "clot_pattern": clot_pattern,
                        "environment_clot_initialization": pattern["environment_clot_initialization"],
                        "obstacle_density": obstacle_density,
                        "dynamic_obstacle_count": obstacle["dynamic_obstacle_count"],
                        "obstacle_layout": obstacle_layout,
                    })
                    condition_id += 1
    return rows


def seed_for(cfg, split, anatomy_index, condition_id, replicate):
    ns = cfg["seed_namespace"]
    base = ns[f"{split}_base"]
    return (base + anatomy_index * ns["anatomy_stride"] +
            condition_id * ns["condition_stride"] + replicate * ns["replicate_stride"])


def realize_clot_pattern(pattern, scene_seed):
    if isinstance(pattern, list):
        # Stable, explicit realization. The realization is part of the manifest,
        # so reset code never has to infer it from hidden simulator state.
        return pattern[scene_seed % len(pattern)]
    return pattern


def make_rows(cfg, splits):
    conditions = condition_rows(cfg)
    train = set(cfg["train_anatomies"])
    holdout = set(cfg["holdout_test_anatomies"])
    rows = []
    for split, anatomies, count in (
        ("train", cfg["train_anatomies"], cfg["split_replicates"]["train"]),
        ("validation", cfg["train_anatomies"], cfg["split_replicates"]["validation"]),
        ("holdout_test", cfg["holdout_test_anatomies"], cfg["split_replicates"]["holdout_test"]),
    ):
        for anatomy_index, anatomy in enumerate(splits["anatomy_order"]):
            if anatomy not in anatomies:
                continue
            for condition in conditions:
                for replicate in range(count):
                    scene_seed = seed_for(cfg, split, anatomy_index, condition["condition_id"], replicate)
                    realized = realize_clot_pattern(condition["environment_clot_initialization"], scene_seed)
                    row = {
                        "dataset_version": cfg["dataset_version"],
                        "split": split,
                        "anatomy": anatomy,
                        "anatomy_index": anatomy_index,
                        "clusters": None,
                        "scene_seed": scene_seed,
                        "condition_id": condition["condition_id"],
                        "replicate": replicate,
                        **condition,
                        "environment_clot_initialization_realized": realized,
                        "scenario_hash": sha256_text(canonical({
                            "dataset_version": cfg["dataset_version"],
                            "split": split,
                            "anatomy": anatomy,
                            "condition_id": condition["condition_id"],
                            "replicate": replicate,
                            "scene_seed": scene_seed,
                        })),
                    }
                    for clusters in cfg["cluster_counts"]:
                        item = dict(row)
                        item["clusters"] = clusters
                        item["scenario_hash"] = sha256_text(canonical({**row, "clusters": clusters}))
                        rows.append(item)
    return rows


def validate_rows(rows, cfg, splits):
    expected = (len(cfg["train_anatomies"]) * cfg["split_replicates"]["train"] +
                len(cfg["train_anatomies"]) * cfg["split_replicates"]["validation"] +
                len(cfg["holdout_test_anatomies"]) * cfg["split_replicates"]["holdout_test"]) * \
               len(condition_rows(cfg)) * len(cfg["cluster_counts"])
    if len(rows) != expected:
        raise ValueError(f"row count {len(rows)} != expected {expected}")
    hashes = [r["scenario_hash"] for r in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError("scenario_hash collision")
    seeds_by_split = {}
    for row in rows:
        seeds_by_split.setdefault(row["split"], set()).add(row["scene_seed"])
        if row["clusters"] not in cfg["cluster_counts"]:
            raise ValueError("invalid cluster count")
        if row["flow_mode"] not in cfg["flow_modes"]:
            raise ValueError("invalid flow mode")
        if row["obstacle_density"] not in cfg["obstacle_density"]:
            raise ValueError("invalid obstacle density")
        if row["clot_pattern"] not in cfg["clot_patterns"]:
            raise ValueError("invalid clot pattern")
        if row["environment_clot_initialization_realized"] not in ("historical_sites", "random_branches"):
            raise ValueError("invalid realized clot initialization")
        if row["anatomy"] not in splits["anatomy_order"]:
            raise ValueError("unknown anatomy")
    if set(seeds_by_split["train"]) & set(seeds_by_split["validation"]):
        raise ValueError("train/validation scene seed overlap")
    if (set(seeds_by_split["train"]) | set(seeds_by_split["validation"])) & set(seeds_by_split["holdout_test"]):
        raise ValueError("training and holdout scene seed overlap")
    for row in rows:
        expected_anatomies = cfg["train_anatomies"] if row["split"] != "holdout_test" else cfg["holdout_test_anatomies"]
        if row["anatomy"] not in expected_anatomies:
            raise ValueError("anatomy assigned to wrong split")
    return {
        "rows": len(rows),
        "unique_scenarios": len(hashes),
        "unique_scene_seeds": {k: len(v) for k, v in seeds_by_split.items()},
        "condition_count": len(condition_rows(cfg)),
        "cluster_counts": cfg["cluster_counts"],
        "flow_modes": sorted(cfg["flow_modes"]),
        "obstacle_densities": sorted(cfg["obstacle_density"]),
        "clot_patterns": sorted(cfg["clot_patterns"]),
        "train_anatomies": cfg["train_anatomies"],
        "holdout_test_anatomies": cfg["holdout_test_anatomies"],
    }


def write_dataset(rows, cfg, splits, report):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "manifest.jsonl"
    lines = "".join(canonical(row) + "\n" for row in rows)
    manifest_path.write_text(lines)
    manifest_sha = sha256_text(lines)
    metadata = {
        "dataset_version": cfg["dataset_version"],
        "config_path": str(CONFIG.relative_to(ROOT)),
        "split_source": str(SPLITS.relative_to(ROOT)),
        "manifest": "manifest.jsonl",
        "manifest_sha256": manifest_sha,
        "row_count": len(rows),
        "summary": report,
        "seed_namespace": cfg["seed_namespace"],
        "protocol": cfg["protocol"],
        "note": "Manifest rows are reset scenarios. accepted_seed, scenario_hash from paired_environment, and per-episode metrics must be appended by the runner; this file is frozen before model selection.",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    (OUT_DIR / "VALIDATION.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return manifest_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", action="store_true", help="write the frozen manifest")
    parser.add_argument("--check", action="store_true", help="validate config and generated rows")
    args = parser.parse_args()
    cfg, splits = load_config()
    rows = make_rows(cfg, splits)
    report = validate_rows(rows, cfg, splits)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        print("manifest_sha256=" + write_dataset(rows, cfg, splits, report))
    elif args.check:
        path = OUT_DIR / "manifest.jsonl"
        if not path.exists():
            raise SystemExit("manifest does not exist; run --output first")
        actual = sha256_text(path.read_text())
        expected = sha256_text("".join(canonical(row) + "\n" for row in rows))
        if actual != expected:
            raise SystemExit(f"manifest hash mismatch: existing={actual} expected={expected}")
        metadata_path = OUT_DIR / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("manifest_sha256") != actual:
                raise SystemExit("metadata manifest_sha256 does not match manifest")
        print("existing_manifest_sha256=" + actual)


if __name__ == "__main__":
    main()
