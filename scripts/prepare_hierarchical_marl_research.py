"""Prepare the independent hierarchical MARL research workspace."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("experiments/hierarchical_marl_research_20260906")


def main() -> None:
    config = json.loads((ROOT / "research_config.json").read_text(encoding="utf-8"))
    directories = [
        ROOT / "logs",
        ROOT / "training",
        ROOT / "evaluation",
        ROOT / "media_fast",
        ROOT / "media_slow",
        ROOT / "reports",
        ROOT / "manifests",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for stage in config["stages"]:
        for method in config["methods"]:
            if stage["id"] == "stage_0_contract" and method["id"] not in {
                "nearest_flow_baseline",
                "hungarian_flow",
                "hierarchical_gat_mappo",
            }:
                continue
            for seed in config["seeds"]:
                rows.append({
                    "stage": stage["id"],
                    "method": method["id"],
                    "seed": seed,
                    "robots": stage["robots"],
                    "clots": stage["clots"],
                    "horizon": stage["horizon"],
                    "transitions": stage["transitions"],
                    "gpu": "0" if method["type"] in {"hierarchical_marl", "hierarchical_marl_ablation"} else "1",
                    "status": "planned",
                })

    with (ROOT / "jobs.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "project": config["project"],
        "job_count": len(rows),
        "methods": [method["id"] for method in config["methods"]],
        "stages": [stage["id"] for stage in config["stages"]],
        "media_required": config["media"]["required"],
        "media_formats": config["media"]["formats"],
        "status": "planned_not_started",
    }
    (ROOT / "manifests" / "planning_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
