"""Evaluate fixed three-lesion sequential/parallel navigation with pursuit.

The experiment reuses the vascular simulator, image sensor, target assignment,
WallGuard, spacing shield, and route execution from ``LysisEpisode``.  Only the
task layer changes: three named lesions are assigned as N=1 sequential, N=2
split (one cluster visits two), or N=3 parallel.  Every lesion is counted once
when its assigned cluster first enters the simulator's navigation tolerance.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_lysis import LysisEpisode, SettleGuard
from marl.vessel_sdf import VesselSDF, project_local_commands


ASSIGNMENTS = {
    1: [[0, 1, 2]],
    2: [[0, 1], [2]],
    3: [[0], [1], [2]],
}


def run_episode(clusters: int, seed: int, anatomy: str, horizon: float, tau: float):
    assignment = ASSIGNMENTS[clusters]
    ep = LysisEpisode(clusters, anatomy, seed, horizon=horizon, sensing="image",
                      shield=True, tpg=True, navigation=True)
    guard = SettleGuard(ep)
    sdf = VesselSDF(ep.sensor)
    est = ep.observe()
    current = np.asarray([seq[0] for seq in assignment], dtype=np.int64)
    next_index = np.zeros(clusters, dtype=np.int64)
    reached = np.zeros(3, dtype=bool)
    first_time = np.full(3, np.nan, dtype=float)
    wall = 0.0
    pair = 0.0
    collisions = 0
    started = time.monotonic()
    while True:
        targets = current.copy()
        ep.navigation_targets = targets.copy()
        rule = ep.ctl.act(targets, est)
        hold = np.zeros(clusters, dtype=bool)
        local = guard(ep, est, targets, rule, hold)
        local, _ = project_local_commands(local, est, targets, hold, ep.ctl.frames(est),
                                          sdf, ep.ctl.body)
        done, outcome = ep.step(est, local, hold)
        wall += float(np.asarray(outcome["wall"], dtype=float).sum())
        pair += float(getattr(ep, "pair", 0.0))
        collisions += int(ep.info.get("episode_particle_collision_events", 0))
        distances = ep.env._target_distances()[np.arange(clusters), current]
        for i, sequence in enumerate(assignment):
            if next_index[i] >= len(sequence):
                continue
            lesion = int(sequence[next_index[i]])
            if np.isfinite(distances[i]) and distances[i] <= .30:
                if not reached[lesion]:
                    reached[lesion] = True
                    first_time[lesion] = float(ep.env.elapsed_s)
                next_index[i] += 1
                if next_index[i] < len(sequence):
                    current[i] = sequence[next_index[i]]
        if reached.all() or done or not ep.env.active[:clusters].all():
            break
        est = ep.observe()
    lost = int(np.count_nonzero(~ep.env.active[:clusters]))
    finite = first_time[np.isfinite(first_time)]
    span = float(np.max(finite) - np.min(finite)) if len(finite) == 3 else None
    result = {
        "method": "pursuit",
        "anatomy": anatomy,
        "clusters": clusters,
        "seed": int(seed),
        "assignment": assignment,
        "lesion_reached": reached.astype(int).tolist(),
        "first_arrival_s": [None if not np.isfinite(x) else float(x) for x in first_time],
        "all_three_reached": bool(reached.all()),
        "arrival_span_s": span,
        "tau_s": float(tau),
        "synchronized_success": bool(reached.all() and span is not None and span <= tau and lost == 0),
        "last_arrival_s": None if len(finite) != 3 else float(np.max(finite)),
        "wall_contact_s": wall,
        "spacing_pair_s": pair,
        "particle_collision_events": collisions,
        "lost": lost,
        "elapsed_s": float(ep.env.elapsed_s),
        "timeout": bool(not reached.all() and ep.env.elapsed_s >= horizon),
        "seconds": float(time.monotonic() - started),
    }
    ep.close()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", default="1,2,3")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--seed", type=int, default=2944200000)
    p.add_argument("--anatomy", default="mca_m1_lvo")
    p.add_argument("--horizon", type=float, default=300.)
    p.add_argument("--tau", type=float, default=30.)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for n in [int(x) for x in a.clusters.split(",")]:
        for offset in range(a.count):
            row = run_episode(n, a.seed + offset, a.anatomy, a.horizon, a.tau)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {"episodes": len(rows), "tau_s": a.tau, "rows": rows, "by_clusters": {}}
    for n in sorted({r["clusters"] for r in rows}):
        group = [r for r in rows if r["clusters"] == n]
        summary["by_clusters"][str(n)] = {
            "episodes": len(group),
            "all_three_rate": float(np.mean([r["all_three_reached"] for r in group])),
            "synchronized_rate": float(np.mean([r["synchronized_success"] for r in group])),
            "last_arrival_s": float(np.nanmean([r["last_arrival_s"] if r["last_arrival_s"] is not None else np.nan for r in group])),
            "arrival_span_s": float(np.nanmean([r["arrival_span_s"] if r["arrival_span_s"] is not None else np.nan for r in group])),
            "wall_contact_s": float(np.mean([r["wall_contact_s"] for r in group])),
            "lost_rate": float(np.mean([r["lost"] > 0 for r in group])),
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
