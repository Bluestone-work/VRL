"""Run the registered EXP0046 development cells with process isolation."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/validation/EXP0046_ISOLATED_MATRIX_20261004"
RUNNER = ROOT / "scripts/evaluate_multicluster_isolated.py"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [("single_sequential", 1, None)] + [
        ("multi_parallel", n, d) for n in (2, 3) for d in (1., 2., 4.)
    ]
    records = []
    for index, (method, clusters, dmin) in enumerate(cases):
        label = f"{method}_n{clusters}" if dmin is None else f"{method}_n{clusters}_d{int(dmin)}"
        out = OUT / f"{label}.jsonl"
        command = [
            sys.executable, str(RUNNER), "--method", method,
            "--clusters", str(clusters), "--episodes", "5",
            "--seed-base", str(1301000000 + index * 100), "--duration-s", "5",
            "--d-min-mm", str(2. if dmin is None else dmin), "--out", str(out),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True)
        records.append({"label": label, "returncode": result.returncode, "output": str(out)})
    (OUT / "matrix_status.json").write_text(
        __import__("json").dumps({"diagnostic_only": True, "sealed_test_used": False,
                                  "cases": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
