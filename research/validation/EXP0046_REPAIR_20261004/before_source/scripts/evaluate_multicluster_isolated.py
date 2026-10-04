"""Episode-isolated runner for EXP0046.

The reference physical integrator can segfault after repeated long rollouts in
one interpreter.  This wrapper starts one evaluator process per episode, keeps
successful JSONL rows, and records a structured crash row for the failed seed.
It is intentionally diagnostic and never writes to the sealed-test ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from scripts.safe_metrics import aggregate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("single_sequential", "multi_parallel"), required=True)
    p.add_argument("--clusters", type=int, default=2)
    p.add_argument("--d-min-mm", type=float, default=2.)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-base", type=int, default=1300000000)
    p.add_argument("--anatomy", default="mca_m1_lvo")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--noise", type=float, default=.025)
    p.add_argument("--duration-s", type=float, default=5.)
    p.add_argument("--timeout-s", type=float, default=120.)
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows, crashes = [], []
    evaluator = Path(__file__).with_name("evaluate_multicluster.py")
    with tempfile.TemporaryDirectory(prefix="exp0046_parts_", dir=args.out.parent) as part_dir:
        for index in range(args.episodes):
            seed = args.seed_base + index
            part = Path(part_dir) / f"episode_{index:04d}.jsonl"
            command = [
                sys.executable, str(evaluator), "--method", args.method,
                "--clusters", str(args.clusters), "--d-min-mm", str(args.d_min_mm),
                "--episodes", "1", "--seed-base", str(seed), "--anatomy", args.anatomy,
                "--out", str(part), "--noise", str(args.noise),
                "--duration-s", str(args.duration_s),
            ]
            if args.config is not None:
                command.extend(["--config", str(args.config)])
            env = os.environ.copy()
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            try:
                completed = subprocess.run(
                    command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, timeout=args.timeout_s,
                )
            except subprocess.TimeoutExpired as error:
                crashes.append(dict(seed=seed, status="timeout", detail=str(error)))
                print(json.dumps(crashes[-1], ensure_ascii=False), flush=True)
                continue
            if completed.returncode != 0 or not part.exists() or part.stat().st_size == 0:
                crashes.append(dict(seed=seed, status="process_exit", returncode=completed.returncode,
                                    stderr=completed.stderr[-1000:]))
                print(json.dumps(crashes[-1], ensure_ascii=False), flush=True)
                continue
            try:
                row = json.loads(part.read_text(encoding="utf-8").splitlines()[0])
            except (ValueError, IndexError) as error:
                crashes.append(dict(seed=seed, status="invalid_output", detail=str(error)))
                continue
            rows.append(row)
            print(json.dumps({"seed": seed, "status": "ok", "safe_success": row.get("safe_success"),
                              "removal": row.get("removal")}, ensure_ascii=False), flush=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = {
        "method": args.method, "clusters": args.clusters, "min_spacing_mm": args.d_min_mm,
        "requested_episodes": args.episodes, "completed_episodes": len(rows),
        "crashed_episodes": len(crashes), "crashes": crashes,
        "aggregate": aggregate(rows), "duration_s": args.duration_s,
        "episode_isolated": True, "development_feasibility": True,
        "local_observation": args.method == "multi_parallel", "sealed_test_used": False,
    }
    args.out.with_name(args.out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
