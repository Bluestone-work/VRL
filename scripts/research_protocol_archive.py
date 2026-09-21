"""Archive EXP_0002 compact evidence using a private Git index."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

from scripts.research_protocol import read, write_new, file_hash
from scripts.research_protocol_run import assert_frozen, git


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "research/runs/EXP_0002"
    study = read(root / "configs/experiments/EXP_0002.json")
    _, parent = assert_frozen(root, study, run)
    assert read(run / "analysis/aggregate.json")["gate"]["passed"]
    branch = study["results_branch"]
    if subprocess.run(["git", "show-ref", "--verify", "--quiet", "refs/heads/" + branch], cwd=root).returncode == 0:
        raise RuntimeError("Results branch exists; refusing to replace")
    files = [p for p in (root / "research").rglob("*.md") if "runs" not in p.relative_to(root).parts]
    files += list((root / "research/figures").glob("EXP_0002_*.png"))
    files += [root / "research/EXPERIMENT_REGISTRY.csv", root / "scripts/research_protocol_report.py", Path(__file__)]
    files += [p for p in run.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".csv", ".xml", ".log")]
    relative = sorted({str(p.relative_to(root)) for p in files})
    with tempfile.TemporaryDirectory(prefix="vascular_exp0002_results_") as temp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / "index"))
        subprocess.run(["git", "read-tree", parent], cwd=root, env=env, check=True)
        subprocess.run(["git", "add", "-f", "--", *relative], cwd=root, env=env, check=True)
        tree = git(root, "write-tree", env=env)
        commit = subprocess.run(["git", "commit-tree", tree, "-p", parent], cwd=root, env=env,
            input="research EXP_0002: archive paired evaluation, sealed split manifests and protocol review\n",
            text=True, capture_output=True, check=True).stdout.strip()
        subprocess.run(["git", "update-ref", "refs/heads/" + branch, commit, "0" * 40], cwd=root, check=True)
    assert_frozen(root, study, run)
    result = {"execution_commit": parent, "results_commit": commit, "results_branch": branch,
              "archived_files": len(relative), "excluded_large_files": "NPZ traces stay local; artifact hashes archived",
              "analysis_manifest_sha256": file_hash(run / "analysis/artifact_manifest.json")}
    write_new(run / "results_archive.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
