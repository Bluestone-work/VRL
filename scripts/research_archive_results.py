"""Commit compact completed research evidence on a separate results branch.

The user's current branch and index and the frozen execution worktree are untouched.
Large checkpoints, traces and training logs remain local, covered by the hash manifest.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "research/runs/EXP_0001"
    result = json.loads((run / "analysis/aggregate.json").read_text())
    parent = result["git_commit"]
    branch = "research/EXP_0001-results"
    existing = subprocess.run(["git", "show-ref", "--verify", "--quiet", "refs/heads/" + branch], cwd=root)
    if existing.returncode == 0:
        raise RuntimeError("Results branch already exists; refusing to replace it")
    paths = [p for p in (root / "research").rglob("*.md") if "runs" not in p.relative_to(root).parts]
    paths.extend((root / "research/figures").glob("EXP_0001_*.png"))
    paths.extend([root / "research/EXPERIMENT_REGISTRY.csv", root / "scripts/research_report.py",
                  root / "scripts/research_check_replay.py", root / "scripts/research_verify_artifacts.py", Path(__file__)])
    paths.extend(p for p in (run / "analysis").iterdir() if p.suffix in (".json", ".csv"))
    paths.extend(p for p in (run / "provenance").rglob("*") if p.is_file())
    paths.extend(p for p in (run / "checks").iterdir() if p.suffix in (".json", ".py", ".xml", ".log"))
    paths.extend(run / name for name in ("training_gate.json", "evaluation_gate.json"))
    for seed in result["seeds"]:
        for name in ("config.json", "summary.json", "health_check.json", "attempt_1.json", "attempt_1.health.summary.json"):
            paths.append(run / f"training/seed_{seed}" / name)
        for name in ("protocol.json", "summary.json", "episodes.jsonl", "attempt_1.json"):
            paths.append(run / f"evaluation/seed_{seed}" / name)
    paths.extend(p for p in (run / "smoke").iterdir() if p.suffix == ".json")
    relative = sorted({str(p.relative_to(root)) for p in paths})
    for name in relative:
        if not (root / name).is_file():
            raise FileNotFoundError(name)
    with tempfile.TemporaryDirectory(prefix="vascular_results_index_") as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        subprocess.run(["git", "read-tree", parent], cwd=root, env=env, check=True)
        subprocess.run(["git", "add", "-f", "--", *relative], cwd=root, env=env, check=True)
        tree = subprocess.check_output(["git", "write-tree"], cwd=root, env=env, text=True).strip()
        commit = subprocess.run(["git", "commit-tree", tree, "-p", parent], cwd=root, env=env, text=True,
            input="research EXP_0001: archive baseline results, diagnostics, figures and phase review\n",
            capture_output=True, check=True).stdout.strip()
        subprocess.run(["git", "update-ref", "refs/heads/" + branch, commit, "0" * 40], cwd=root, check=True)
    record = {"execution_commit": parent, "results_commit": commit, "results_branch": branch,
              "archived_files": len(relative), "large_artifacts": "Local checkpoints/traces/logs; see analysis/artifact_manifest.json"}
    with (run / "results_archive.json").open("x") as file:
        json.dump(record, file, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
