"""Versioned, append-only execution of preregistered baseline reproduction.

No policy, environment, reward, or hyperparameter tuning is performed here.
Run prepare, checks, smoke, train, evaluate, in that order.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output(command, cwd, env=None):
    return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()


def source_files(root):
    paths = set(output(["git", "ls-files"], root).splitlines())
    for parent in (root, root / "environments", root / "marl", root / "scripts", root / "tests"):
        paths.update(str(p.relative_to(root)) for p in parent.iterdir()
                     if p.is_file() and p.suffix in (".py", ".md", ".sh", ".json", ".txt"))
    paths.update(str(p.relative_to(root)) for p in (root / "research").rglob("*.md"))
    paths.add("configs/experiments/EXP_0001.json")
    return sorted(p for p in paths if (root / p).is_file() and not p.startswith(
        ("experiments/", "vascular_marl_local/", "research/runs/")))


def prepare(root, config, run):
    provenance = run / "provenance"
    if (provenance / "snapshot.json").exists():
        raise RuntimeError("Already frozen; refusing to replace experiment provenance")
    snapshot = Path(config["snapshot_directory"])
    if snapshot.exists():
        raise RuntimeError(f"Snapshot destination already exists: {snapshot}")
    provenance.mkdir(parents=True, exist_ok=True)
    files = source_files(root)
    inventory = []
    for name in files:
        path = root / name
        row = {"path": name, "sha256": sha(path), "bytes": path.stat().st_size}
        if path.suffix == ".py":
            text = path.read_text()
            tree = ast.parse(text, filename=name)
            row["lines"] = len(text.splitlines())
            row["definitions"] = [{"name": node.name, "line": node.lineno,
                                   "kind": type(node).__name__} for node in ast.walk(tree)
                                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        inventory.append(row)
    save(provenance / "source_inventory.json", inventory)
    parent = output(["git", "rev-parse", "HEAD"], root)
    status = output(["git", "status", "--porcelain=v1"], root)
    with tempfile.TemporaryDirectory(prefix="vascular_research_index_") as temporary:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(temporary) / "index"))
        subprocess.run(["git", "read-tree", "HEAD"], cwd=root, env=env, check=True)
        subprocess.run(["git", "add", "-f", "--", *files], cwd=root, env=env, check=True)
        tree = output(["git", "write-tree"], root, env)
        result = subprocess.run(["git", "commit-tree", tree, "-p", parent], cwd=root, env=env,
                                input="research EXP_0001: freeze audited working source and baseline protocol\n",
                                text=True, capture_output=True, check=True)
        commit = result.stdout.strip()
    subprocess.run(["git", "worktree", "add", "-b", config["snapshot_branch"], str(snapshot), commit],
                   cwd=root, check=True)
    save(provenance / "snapshot.json", {"created_at": now(), "git_commit": commit, "parent_commit": parent,
         "branch": config["snapshot_branch"], "directory": str(snapshot), "original_status": status,
         "config_sha256": sha(root / "configs/experiments/EXP_0001.json")})
    shutil.copy2(root / "configs/experiments/EXP_0001.json", provenance / "experiment_config.json")
    reference = []
    for path in sorted((root / "experiments/ladder_stage1").glob("geodesic_v_seed*/summary.json")):
        folder = provenance / "reference" / path.parent.name
        folder.mkdir(parents=True)
        for name in ("summary.json", "config.json", "ladder_config.json", "command.txt"):
            shutil.copy2(path.parent / name, folder / name)
        reference.append({"directory": str(path.parent), "summary_sha256": sha(path),
                          "checkpoint_sha256": sha(path.parent / "final_policy.pt")})
    save(provenance / "reference.json", reference)
    metadata = {"captured_at": now(), "packages": {}, "python": output([config["python"], "--version"], root),
                "cpu_affinity": config["cpu_affinity"], "env": config["environment_variables"]}
    for package in ("numpy", "torch", "gymnasium", "pybullet", "pillow", "pytest", "matplotlib", "scipy"):
        metadata["packages"][package] = importlib.metadata.version(package)
    for key, command in {"gpu": ["nvidia-smi", "-q"], "cpu": ["lscpu"], "memory": ["free", "-h"],
                         "pip_freeze": [config["python"], "-m", "pip", "freeze"], "uname": ["uname", "-a"]}.items():
        metadata[key] = output(command, root)
    save(provenance / "machine.json", metadata)
    print(json.dumps({"snapshot_commit": commit, "source_files": len(files), "directory": str(snapshot)}), flush=True)


def assert_frozen(root, config, run):
    snapshot = Path(config["snapshot_directory"])
    record = json.loads((run / "provenance/snapshot.json").read_text())
    if output(["git", "rev-parse", "HEAD"], snapshot) != record["git_commit"]:
        raise RuntimeError("Snapshot HEAD changed")
    for row in json.loads((run / "provenance/source_inventory.json").read_text()):
        if sha(snapshot / row["path"]) != row["sha256"]:
            raise RuntimeError(f"Frozen source changed: {row['path']}")
    if sha(root / "configs/experiments/EXP_0001.json") != record["config_sha256"]:
        raise RuntimeError("Registered configuration changed")
    return snapshot, record["git_commit"]


def run_child(command, directory, label, config, cwd, extra_env=None):
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / f"{label}.json"
    if record_path.exists():
        raise RuntimeError(f"Refusing to overwrite prior attempt: {record_path}")
    cpus = ",".join(map(str, config["cpu_affinity"]))
    command = ["taskset", "-c", cpus, *map(str, command)]
    env = dict(os.environ, **config["environment_variables"], PYTHONPATH=str(cwd))
    env.pop("DISPLAY", None)
    env.update(extra_env or {})
    record = {"started_at": now(), "command": command, "cwd": str(cwd), "status": "running"}
    save(record_path, record)
    started = time.monotonic()
    with (directory / f"{label}.stdout.log").open("x") as stdout, (directory / f"{label}.stderr.log").open("x") as stderr:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
    record.update(finished_at=now(), seconds=time.monotonic() - started, returncode=result.returncode,
                  status="complete" if result.returncode == 0 else "failed")
    save(record_path, record)
    print(json.dumps({"job": str(directory), **record}, ensure_ascii=False), flush=True)
    return result.returncode


def checks(root, config, run):
    snapshot, commit = assert_frozen(root, config, run)
    code = run_child([config["python"], "-m", "pytest", "tests/", "-q", "--junitxml=" + str(run / "checks/pytest.xml")],
                     run / "checks", "pytest", config, snapshot)
    if code:
        raise RuntimeError("Regression checks failed; see retained logs")
    save(run / "checks/gate.json", {"passed": True, "git_commit": commit, "time": now()})


def trainer_command(config, snapshot, directory, seed, device, smoke=False):
    settings = dict(config["training"])
    if smoke:
        settings.update({k: v for k, v in config["smoke"].items() if k != "seed"})
    command = [config["python"], snapshot / "scripts/research_train_entry.py",
               "--run-dir", directory, "--seed", str(seed), "--device", device]
    for key, value in settings.items():
        if value is True:
            command.append("--" + key)
        elif value is not False and value is not None:
            command.extend(["--" + key, str(value)])
    return command


def finite_tree(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return True


def inspect_run(directory, expected):
    import torch
    summary = json.loads((directory / "summary.json").read_text())
    if summary["real_transitions"] != expected or not finite_tree(summary):
        raise AssertionError("Incorrect budget or non-finite summary")
    records = [json.loads(line) for line in (directory / "update_metrics.jsonl").read_text().splitlines()]
    if not records or not all(finite_tree(row) for row in records):
        raise AssertionError("Missing updates or non-finite metrics")
    gradient_count = 0
    for path in directory.glob("attempt_*.health.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not row["finite"] or not finite_tree(row):
                raise AssertionError("Non-finite gradients")
            gradient_count += 1
    if not gradient_count:
        raise AssertionError("No observed gradients")
    final = torch.load(directory / "final_policy.pt", map_location="cpu", weights_only=False)
    initial = torch.load(directory / "initial_policy.pt", map_location="cpu", weights_only=False)
    for key in ("actor", "critic"):
        if not all(torch.isfinite(tensor).all() for tensor in final[key].values()):
            raise AssertionError("Non-finite checkpoint parameters")
    changed = any(not torch.equal(value, initial["actor"][name]) for name, value in final["actor"].items())
    if not changed:
        raise AssertionError("Actor parameters never changed")
    episodes = summary["final_evaluation"]["episodes"]
    recomputed = {key: sum(row[key] for row in episodes) / len(episodes) for key in ("success", "removal_rate")}
    for key, value in recomputed.items():
        if abs(value - summary["final_evaluation"]["macro"][key]) > 1e-7:
            raise AssertionError("Episode metrics do not reproduce summary")
    return {"passed": True, "updates": len(records), "observed_gradient_calls": gradient_count,
            "actor_changed": changed, "checkpoint_sha256": sha(directory / "final_policy.pt"),
            "recomputed": recomputed, "episodes": len(episodes),
            "max_pre_update_ratio_error": max(row.get("pre_update_ratio_max_error", 0) for row in records)}


def training_job(root, config, run, seed, device, smoke=False):
    snapshot, commit = assert_frozen(root, config, run)
    directory = run / ("smoke" if smoke else f"training/seed_{seed}")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "health_check.json").exists():
        raise RuntimeError("Run already inspected; refusing performance rerun")
    save(directory / "execution_provenance.json", {"git_commit": commit, "seed": seed, "device": device,
         "experiment_id": config["experiment_id"], "role": "smoke" if smoke else "formal"})
    command = trainer_command(config, snapshot, directory, seed, device, smoke)
    # EXP_0001 deliberately stops on failure; recovery needs inspection before using the retry budget.
    code = run_child(command, directory, "attempt_1", config, snapshot,
                     {"RESEARCH_HEALTH_PATH": str(directory / "attempt_1.health.jsonl")})
    if code:
        raise RuntimeError(f"Training failed, preserved {directory}")
    result = inspect_run(directory, config["smoke"]["timesteps"] if smoke else config["training"]["timesteps"])
    save(directory / "health_check.json", result)
    return result


def train(root, config, run, smoke=False):
    gate = run / ("checks/gate.json" if smoke else "smoke/health_check.json")
    if not gate.exists() or not json.loads(gate.read_text())["passed"]:
        raise RuntimeError(f"Prerequisite gate absent: {gate}")
    if smoke:
        return training_job(root, config, run, config["smoke"]["seed"], config["devices"][0], True)
    # One queue per GPU; fixed assignment, independent of performance.
    queues = {device: config["seeds"][index::len(config["devices"])] for index, device in enumerate(config["devices"])}
    def queue(device, seeds):
        return [training_job(root, config, run, seed, device) for seed in seeds]
    results = []
    with ThreadPoolExecutor(max_workers=len(queues)) as pool:
        futures = [pool.submit(queue, device, seeds) for device, seeds in queues.items()]
        for future in as_completed(futures):
            results.extend(future.result())
    save(run / "training_gate.json", {"passed": True, "results": results, "time": now()})


def evaluate(root, config, run):
    snapshot, commit = assert_frozen(root, config, run)
    if not (run / "training_gate.json").exists():
        raise RuntimeError("Formal training gate absent")
    def one(seed):
        directory = run / f"evaluation/seed_{seed}"
        code = run_child([config["python"], snapshot / "scripts/research_evaluate.py", "--run",
                          run / f"training/seed_{seed}", "--config", root / "configs/experiments/EXP_0001.json",
                          "--output", directory], directory, "attempt_1", config, snapshot)
        if code:
            raise RuntimeError(f"Evaluation failed for seed {seed}")
        return {"seed": seed, "status": "complete"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(one, config["seeds"]))
    save(run / "evaluation_gate.json", {"passed": True, "git_commit": commit, "results": results, "time": now()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "checks", "smoke", "train", "evaluate"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads((root / "configs/experiments/EXP_0001.json").read_text())
    run = root / "research/runs/EXP_0001"
    if args.stage == "prepare": prepare(root, config, run)
    elif args.stage == "checks": checks(root, config, run)
    elif args.stage == "smoke": train(root, config, run, smoke=True)
    elif args.stage == "train": train(root, config, run)
    else: evaluate(root, config, run)


if __name__ == "__main__":
    main()
