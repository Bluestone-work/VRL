"""Validate study completeness and generate artifact/source SHA256 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_transitions(path: Path):
    return sum(
        int(json.loads(line)["transitions"])
        for line in path.read_text().splitlines() if line
    )


def load_summaries(paths, expected_transitions, expected_added=None):
    records = []
    for path in paths:
        data = json.loads(path.read_text())
        if int(data["real_transitions"]) < expected_transitions(path, data):
            raise RuntimeError(f"insufficient real transitions: {path}")
        if expected_added is not None and int(data["added_real_transitions"]) != expected_added:
            raise RuntimeError(f"incorrect added transition budget: {path}")
        dataset_total = manifest_transitions(path.parent / "dataset" / "manifest.jsonl")
        if dataset_total != int(data["dataset_transitions"]):
            raise RuntimeError(
                f"dataset/summary transition mismatch: {path} "
                f"{dataset_total} != {data['dataset_transitions']}"
            )
        records.append(data)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-since", default="2026-09-05T12:00:00")
    args = parser.parse_args()
    root = Path(args.study_root).resolve()
    repo = Path(args.repo_root).resolve()

    baseline_paths = sorted((root / "baselines").glob("*_r5_c3/seed_*/summary.json"))
    curriculum_paths = sorted((root / "curriculum").glob("gat_r5_c3/seed_*/summary.json"))
    comparison_paths = sorted((root / "comparisons").glob("seed_*/*/summary.json"))
    if (len(baseline_paths), len(curriculum_paths), len(comparison_paths)) != (9, 3, 6):
        raise RuntimeError(
            "expected 9 baseline, 3 curriculum and 6 comparison summaries, got "
            f"{len(baseline_paths)}, {len(curriculum_paths)}, {len(comparison_paths)}"
        )

    baselines = load_summaries(
        baseline_paths, lambda _path, _data: 500_000
    )
    curricula = load_summaries(
        curriculum_paths, lambda _path, _data: 1_000_000
    )
    comparisons = load_summaries(
        comparison_paths,
        lambda _path, data: int(data["initial_real_transitions"]) + 250_000,
        expected_added=250_048,
    )

    world_metrics = json.loads((root / "world_model" / "metrics.json").read_text())
    world_dataset = json.loads((root / "world_model" / "dataset.json").read_text())
    report_metrics = json.loads((root / "reports" / "aggregate_metrics.json").read_text())
    media_metrics = json.loads((root / "artifacts" / "media_validation.json").read_text())
    if not world_metrics["gate"]["passed"]:
        raise RuntimeError("world-model gate did not pass")
    if int(world_dataset["merged_transitions"]) != 1_500_000:
        raise RuntimeError("world-model dataset is not the planned 1.5M hard transitions")
    if not report_metrics["complete"]:
        raise RuntimeError("aggregate report is incomplete")
    if not media_metrics["passed"] or media_metrics["scenario_count"] != 20:
        raise RuntimeError("media validation is incomplete")

    final_pytest = root / "logs" / "pytest_posttraining_attempt_3_after_gui_fix.log"
    final_gui = root / "logs" / "watch_gui_best_mve_attempt_4_stride5.log"
    if "126 passed, 1 skipped" not in final_pytest.read_text():
        raise RuntimeError("final pytest result is missing")
    if "ep 5:" not in final_gui.read_text():
        raise RuntimeError("interactive GUI did not finish five episodes")

    exit_codes = (root / "logs" / "training_exit_codes.txt").read_text().splitlines()
    crash_count = sum(line.endswith((" 132", " 139")) for line in exit_codes)
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "baseline_runs": len(baselines),
        "curriculum_runs": len(curricula),
        "equal_budget_runs": len(comparisons),
        "world_model_gate_passed": True,
        "world_model_dataset_transitions": world_dataset["merged_transitions"],
        "world_model_test_geometries": world_metrics["split"]["test"]["geometries"],
        "scenario_mp4": media_metrics["scenario_count"],
        "policy_mp4": media_metrics["policy_mp4_count"],
        "policy_gif": media_metrics["policy_gif_count"],
        "final_pytest": "126 passed, 1 skipped",
        "interactive_gui_episodes": 5,
        "recorded_native_crashes": crash_count,
        "log_files": len(list((root / "logs").glob("*"))),
    }
    (root / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    (root / "logs" / "audit_finalize.log").write_text(
        json.dumps(summary, indent=2) + "\nvalidation=passed\n"
    )

    source_since = datetime.fromisoformat(args.source_since).timestamp()
    source_paths = []
    for base in (repo, repo / "environments", repo / "marl", repo / "scripts", repo / "tests"):
        iterator = base.glob("*.py") if base == repo else base.rglob("*")
        for path in iterator:
            if (
                path.is_file() and path.suffix in (".py", ".sh")
                and "__pycache__" not in path.parts
                and path.stat().st_mtime >= source_since
            ):
                source_paths.append(path.resolve())
    source_paths = sorted(set(source_paths))
    source_lines = [
        f"{sha256(path)}  {path.relative_to(repo)}"
        for path in source_paths
    ]
    (root / "SOURCE_SHA256SUMS").write_text("\n".join(source_lines) + "\n")

    excluded = {"SHA256SUMS"}
    artifact_paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in excluded
    )
    artifact_lines = [
        f"{sha256(path)}  {path.relative_to(root)}"
        for path in artifact_paths
    ]
    (root / "SHA256SUMS").write_text("\n".join(artifact_lines) + "\n")
    print(json.dumps({
        **summary,
        "source_manifest_files": len(source_paths),
        "artifact_manifest_files": len(artifact_paths),
    }, indent=2))


if __name__ == "__main__":
    main()
