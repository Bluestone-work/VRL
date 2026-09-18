"""Add source training artifact indexes to the full matrix directory."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("experiments/full_matrix_20260906")


SOURCES = {
    "flow_guided_gat": "experiments/success_study_20260905/training/flow_guided/seed_43",
    "guided_42": "experiments/success_study_20260905/training/guided/seed_42",
    "local_42": "experiments/success_study_20260905/training/local/seed_42",
    "curriculum_gat_43": "experiments/world_model_study_20260905_134953/curriculum/gat_r5_c3/seed_43",
    "baseline_gat_43": "experiments/world_model_study_20260905_134953/baselines/gat_r5_c3/seed_43",
    "edge_bias_gat_43": "experiments/world_model_study_20260905_134953/baselines/edge_bias_gat_r5_c3/seed_43",
    "mlp_43": "experiments/world_model_study_20260905_134953/baselines/mlp_r5_c3/seed_43",
    "pure_43": "experiments/world_model_study_20260905_134953/comparisons/seed_43/pure",
    "mve_43": "experiments/world_model_study_20260905_134953/comparisons/seed_43/mve",
}


def main():
    for label, source_name in SOURCES.items():
        source = Path(source_name)
        target = ROOT / "training" / label
        target.mkdir(parents=True, exist_ok=True)
        artifact_names = ("command.txt", "config.json", "summary.json", "best_policy.pt", "final_policy.pt")
        artifacts = []
        for name in artifact_names:
            path = source / name
            if path.exists():
                artifacts.append({"name": name, "path": str(path), "bytes": path.stat().st_size})
        logs = []
        for log_root in (source.parent.parent / "logs", Path("experiments/success_study_20260905/logs"),
                         Path("experiments/world_model_study_20260905_134953/logs")):
            if log_root.exists():
                logs.extend(str(path) for path in sorted(log_root.glob("*.log"))
                            if label.split("_")[0] in path.name or source.name in path.name)
        (target / "source_artifacts.json").write_text(json.dumps({
            "method": label, "training_status": "completed_before_full_matrix",
            "source_directory": str(source), "artifacts": artifacts,
            "candidate_logs": sorted(set(logs)),
            "note": "Full matrix reuses this trained checkpoint; it does not claim a new training run.",
        }, indent=2, ensure_ascii=False))

    for label in ("flow_controller", "flow_spread"):
        target = ROOT / "training" / label
        target.mkdir(parents=True, exist_ok=True)
        (target / "source_artifacts.json").write_text(json.dumps({
            "method": label, "training_status": "not_applicable_controller",
            "source_directory": None, "artifacts": [], "candidate_logs": [],
            "note": "Deterministic controller ablation; no learned checkpoint is expected.",
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
