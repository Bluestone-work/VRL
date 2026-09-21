"""Freeze and execute EXP_0002 without changing the user's branch/index."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile

from scripts.research_baseline import source_files, run_child
from scripts.research_protocol import read, write_new, file_hash


def git(root, *args, env=None):
    return subprocess.check_output(["git", *args], cwd=root, env=env, text=True).strip()


def prepare(root, study, run):
    frozen = Path(study["snapshot_directory"])
    if frozen.exists() or (run / "provenance/snapshot.json").exists():
        raise RuntimeError("Existing snapshot; refusing overwrite")
    parent = root / "research/runs/EXP_0001"
    inventory = read(parent / "provenance/source_inventory.json")
    for row in inventory:
        if row["path"].startswith(("environments/", "marl/", "scripts/train")):
            if file_hash(root / row["path"]) != row["sha256"]:
                raise AssertionError(f"Baseline source changed: {row['path']}")
    files = sorted(set(source_files(root)) | {"configs/experiments/EXP_0002.json", "research/EXPERIMENT_REGISTRY.csv"})
    contents = [{"path": name, "sha256": file_hash(root / name)} for name in files]
    branch = git(root, "branch", "--show-current")
    index_path = Path(git(root, "rev-parse", "--git-path", "index"))
    if not index_path.is_absolute():
        index_path = root / index_path
    index_hash = file_hash(index_path)
    with tempfile.TemporaryDirectory(prefix="vascular_exp0002_index_") as temp:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temp) / "index"))
        subprocess.run(["git", "read-tree", study["parent_results_commit"]], cwd=root, env=env, check=True)
        subprocess.run(["git", "add", "-f", "--", *files], cwd=root, env=env, check=True)
        tree = git(root, "write-tree", env=env)
        commit = subprocess.run(["git", "commit-tree", tree, "-p", study["parent_results_commit"]],
            input="research EXP_0002: freeze paired device evaluation and split protocol\n", cwd=root,
            env=env, text=True, capture_output=True, check=True).stdout.strip()
    subprocess.run(["git", "worktree", "add", "-b", study["snapshot_branch"], str(frozen), commit], cwd=root, check=True)
    p = run / "provenance"
    write_new(p / "source_inventory.json", contents)
    write_new(p / "snapshot.json", {"git_commit": commit, "directory": str(frozen),
        "parent": study["parent_results_commit"], "created_at": datetime.now(timezone.utc).isoformat(),
        "original_branch": branch, "original_index_path": str(index_path), "original_index_sha256": index_hash,
        "config_sha256": file_hash(root / "configs/experiments/EXP_0002.json")})
    write_new(p / "config.json", study)
    sources = []
    for seed in study["seeds"]:
        directory = parent / f"training/seed_{seed}"
        paths = list(directory.glob("*.pt")) + [directory / "config.json", directory / "summary.json", directory / "eval_metrics.jsonl"]
        paths += [parent / f"evaluation/seed_{seed}/episodes.jsonl", parent / f"evaluation/seed_{seed}/summary.json"]
        sources.extend({"path": str(path.relative_to(root)), "sha256": file_hash(path)} for path in paths)
    write_new(p / "parent_inputs.json", sources)
    machine = {"captured_at": datetime.now(timezone.utc).isoformat(), "variables": study["environment_variables"]}
    for key, command in {"python": [study["python"], "--version"], "pip_freeze": [study["python"], "-m", "pip", "freeze"],
                         "gpu": ["nvidia-smi", "-q"], "cpu": ["lscpu"], "uname": ["uname", "-a"]}.items():
        machine[key] = subprocess.check_output(command, cwd=root, text=True)
    write_new(p / "machine.json", machine)
    print(json.dumps({"git_commit": commit, "frozen_files": len(contents), "parent_input_files": len(sources)}), flush=True)


def assert_frozen(root, study, run):
    info = read(run / "provenance/snapshot.json")
    frozen = Path(info["directory"])
    assert git(frozen, "rev-parse", "HEAD") == info["git_commit"]
    for row in read(run / "provenance/source_inventory.json"):
        if file_hash(frozen / row["path"]) != row["sha256"]:
            raise AssertionError(f"Frozen source changed: {row['path']}")
    assert file_hash(root / "configs/experiments/EXP_0002.json") == info["config_sha256"]
    for row in read(run / "provenance/parent_inputs.json"):
        if file_hash(root / row["path"]) != row["sha256"]:
            raise AssertionError(f"Parent artifact changed: {row['path']}")
    assert git(root, "branch", "--show-current") == info["original_branch"]
    assert file_hash(info["original_index_path"]) == info["original_index_sha256"]
    return frozen, info["git_commit"]


def child(root, study, run, role, seed, device):
    frozen, _ = assert_frozen(root, study, run)
    dest = run / role / device.replace(":", "_") / f"seed_{seed}"
    command = [study["python"], frozen / "scripts/research_protocol.py", "evaluate", "--project-root", root,
               "--seed", seed, "--device", device, "--role", role, "--output", dest]
    code = run_child(command, dest, "attempt_1", study, frozen)
    if code:
        raise RuntimeError(f"{role} failed; retained {dest}")
    assert read(dest / "checks.json")["passed"]


def execute(root, study, run, stage):
    frozen, commit = assert_frozen(root, study, run)
    gate = run / f"{stage}_gate.json"
    if gate.exists():
        raise RuntimeError(f"Stage {stage} already complete; refusing rerun")
    if stage == "checks":
        command = [study["python"], "-m", "pytest", "tests", "-q", "--junitxml=" + str(run / "checks/pytest.xml")]
        if run_child(command, run / "checks", "pytest", study, frozen):
            raise RuntimeError("Regression failed")
    elif stage == "smoke":
        assert read(run / "checks_gate.json")["passed"]
        for device in study["smoke"]["devices"]:
            for seed in study["smoke"]["seeds"]:
                child(root, study, run, "smoke", seed, device)
    else:
        assert read(run / "smoke_gate.json")["passed"]
        def lane(device):
            for seed in study["seeds"]:
                child(root, study, run, "formal", seed, device)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(lane, device) for device in study["devices"]]
            errors = []
            for job in jobs:
                try:
                    job.result()
                except Exception as error:
                    errors.append(repr(error))
            if errors:
                raise RuntimeError("; ".join(errors))
        for seed in study["seeds"]:
            child(root, study, run, "legacy", seed, study["legacy_control"]["devices_by_seed"][str(seed)])
        command = [study["python"], frozen / "scripts/research_protocol.py", "inventory", "--project-root", root,
                   "--output", run / "splits"]
        if run_child(command, run / "splits", "attempt_1", study, frozen):
            raise RuntimeError("Split audit failed; records preserved")
    assert_frozen(root, study, run)
    write_new(gate, {"passed": True, "git_commit": commit, "time": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "checks", "smoke", "formal", "verify"))
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    study = read(root / "configs/experiments/EXP_0002.json")
    run = root / "research/runs/EXP_0002"
    if args.stage == "prepare":
        prepare(root, study, run)
    elif args.stage == "verify":
        print(assert_frozen(root, study, run))
    else:
        execute(root, study, run, args.stage)


if __name__ == "__main__":
    main()
