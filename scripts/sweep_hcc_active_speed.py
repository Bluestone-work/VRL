"""Compute an identifiability sweep from the calibrated HCC flow report.

This is an analytic bound from the current parabolic profile and finite-radius
wall-margin formula. It is not an experimental magnetic calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("calibration", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--speeds", type=float, nargs="+", default=[1., 10., 50., 100., 150., 200., 250.])
    a = p.parse_args()
    report = json.loads(a.calibration.read_text())
    rows = []
    for flow_case in ("legacy", "reference_healthy_inlet"):
        for scene in report[flow_case]:
            required = [1.0 - float(goal["optimistic_wall_margin_mm_s"])
                        for goal in scene["goals"]]
            for active in a.speeds:
                rows.append(dict(flow_case=flow_case, seed=scene["seed"],
                                 active_speed_mm_s=float(active),
                                 required_active_speed_mm_s=required,
                                 optimistic_targets_reachable=int(sum(active >= x for x in required)),
                                 all_targets_optimistically_reachable=bool(all(active >= x for x in required))))
    out = dict(version="hcc_active_speed_identifiability_v1",
               caveat="Analytic bound only. It does not identify magnetic field strength, magnetic force, cluster drag, or wall lubrication from missing experiments.",
               rows=rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(a.out), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
