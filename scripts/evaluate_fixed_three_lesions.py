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

from scripts.three_target_allocation import geodesic_allocation, linear_sum_allocation
from scripts.benchmark_lysis import LysisEpisode, SettleGuard
from marl.vessel_sdf import VesselSDF, project_local_commands


ASSIGNMENTS = {
    1: [[0, 1, 2]],
    2: [[0, 1], [2]],
    3: [[0], [1], [2]],
}


def entity_pair_contact_seconds(ep):
    """Read the episode-level contact accumulator once at termination."""
    return float(getattr(ep, "pair", 0.0))


def classify_episode(reached, lost, *, terminated=False, truncated=False, elapsed_s=None, horizon=None):
    """Classify the terminal boundary using the simulator's signal first.

    ``elapsed_s`` is retained only for old environments which did not expose
    ``truncated``.  ``isclose`` prevents a floating point value such as
    299.9999999999997 from being treated as an unfinished episode.
    """
    reached = bool(reached)
    lost = bool(lost)
    at_deadline = bool(truncated)
    if not terminated and not truncated and elapsed_s is not None and horizon is not None:
        at_deadline = bool(np.isclose(float(elapsed_s), float(horizon), rtol=0., atol=1e-9) or
                           float(elapsed_s) > float(horizon))
    if reached:
        return "last_step_success" if at_deadline else "success"
    if lost:
        return "lost_before_deadline" if not at_deadline else "lost_at_deadline"
    return "timeout" if at_deadline else "incomplete"


def run_episode(clusters: int, seed: int, anatomy: str, horizon: float, tau: float, tpg: bool = True,
                allocation: str = "fixed", ccbs: bool = False, trace=None):
    ep = LysisEpisode(clusters, anatomy, seed, horizon=horizon, sensing="image",
                      shield=True, tpg=False, navigation=True)
    if ccbs and not tpg:
        raise ValueError("CCBS requires the common coordination gate")
    from scripts.benchmark_multicluster import station_geodesic
    distance = station_geodesic(ep.env)
    if allocation in ("geodesic", "linear_sum"):
        planner = geodesic_allocation if allocation == "geodesic" else linear_sum_allocation
        assignment = planner(distance, ep.env.robot_stations[:clusters],
                             ep.env.clot_stations[:3])
    elif allocation == "fixed":
        assignment = ASSIGNMENTS[clusters]
    else:
        raise ValueError(f"unknown allocation: {allocation}")
    # Navigation mode deliberately disables the pre-operative coordinator while
    # it is constructing its frozen target assignment. Re-enable it here with
    # the fixed lesion visit order so TPG-on and TPG-off remain paired.
    if tpg and clusters > 1:
        ep.use_tpg = True
        ep.coord = ep._build_coord(assignment, dwell_s=0.)
        original_gate = ep.coord.gate
        ep.coord.gate_calls = 0
        def counted_gate(*args, **kwargs):
            ep.coord.gate_calls += 1
            return original_gate(*args, **kwargs)
        ep.coord.gate = counted_gate
    ccbs_plan = None
    if ccbs and clusters > 1:
        from marl.ccbs_adapter import CCBSPlanner
        ccbs_plan = CCBSPlanner(ep.coord, deadline_s=horizon).plan()
        if not ccbs_plan.has_plan:
            ep.coord.gate = lambda positions, active: np.ones(clusters, dtype=bool)
    route_lengths = []
    for i, tour in enumerate(assignment):
        stations = [ep.env.robot_stations[i], *[ep.env.clot_stations[j] for j in tour]]
        route_lengths.append(float(sum(distance[a, b] for a, b in zip(stations, stations[1:]))))
    guard = SettleGuard(ep)
    sdf = VesselSDF(ep.sensor)
    est = ep.observe()
    current = np.asarray([seq[0] for seq in assignment], dtype=np.int64)
    next_index = np.zeros(clusters, dtype=np.int64)
    reached = np.zeros(3, dtype=bool)
    first_time = np.full(3, np.nan, dtype=float)
    wall = 0.0
    active_cluster_s = 0.0
    gate_calls = 0
    waiting_transitions = 0
    waiting_robot_s = 0.0
    tpg_hold_steps = 0
    previous_hold = np.zeros(clusters, dtype=bool)
    terminated = truncated = False
    started = time.monotonic()
    while True:
        targets = current.copy()
        ep.navigation_targets = targets.copy()
        rule = ep.ctl.act(targets, est)
        hold = ep.hold(est) if tpg and clusters > 1 else np.zeros(clusters, dtype=bool)
        gate_calls += 1 if (tpg and clusters > 1) else 0
        tpg_hold_steps += int(np.count_nonzero(hold & ~previous_hold))
        waiting_transitions += int(np.count_nonzero(hold & ~previous_hold))
        previous_hold = hold.copy()
        local = guard(ep, est, targets, rule, hold)
        local, _ = project_local_commands(local, est, targets, hold, ep.ctl.frames(est),
                                          sdf, ep.ctl.body)
        done, outcome = ep.step(est, local, hold)
        terminated = bool(outcome.get("terminated", False))
        truncated = bool(outcome.get("truncated", False))
        waiting_robot_s += float(np.count_nonzero(hold & outcome["active_before"])) * float(ep.info["step_duration_s"])
        wall += float(np.asarray(outcome["wall"], dtype=float).sum())
        active_cluster_s += float(np.count_nonzero(outcome["active_before"])) * float(ep.info["step_duration_s"])
        if trace is not None:
            trace.append(dict(time_s=float(ep.env.elapsed_s), positions_mm=ep.env.positions_mm[:clusters].tolist(),
                              targets=targets.tolist(), active=ep.env.active[:clusters].tolist(),
                              wall_s=np.asarray(outcome["wall"]).tolist(), hold=hold.tolist()))
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
    termination_class = classify_episode(
        reached.all(), lost > 0, terminated=terminated, truncated=truncated,
        elapsed_s=ep.env.elapsed_s, horizon=horizon)
    spacing = ep.spacing.summary()
    collisions = int(ep.info.get("episode_particle_collision_events", 0))
    finite = first_time[np.isfinite(first_time)]
    span = float(np.max(finite) - np.min(finite)) if len(finite) == 3 else None
    result = {
        "method": "pursuit",
        "tpg": bool(tpg and not ccbs),
        "coordinator": "ccbs" if ccbs else ("tpg" if tpg else "off"),
        "planning": ccbs_plan.metadata() if ccbs_plan is not None else None,
        "conflict_zones": len(ep.coord.zones) if ep.coord is not None else None,
        "route_lengths_mm": route_lengths,
        "resource_budget": {
            "horizon_s": horizon, "control_dt_s": ep.env.config.control_dt_s,
            "per_cluster_speed_mm_s": ep.env.config.robot_speed_mm_s,
            "available_cluster_s": clusters * horizon, "used_cluster_s": active_cluster_s,
            "deployment": "nested predeployed starts", "release_modeled": False,
            "active_lysis_mass_per_s": ep.env.config.lysis_mass_per_s,
            "payload_mass": None, "magnetic_power": None,
        },
        "anatomy": anatomy,
        "clusters": clusters,
        "seed": int(seed),
        "assignment": assignment,
        "allocation": allocation,
        "scenario_hash": ep.manifest["scenario_hash"],
        "accepted_seed": ep.manifest["accepted_seed"],
        "selected_start_indices": ep.manifest["selected_start_indices"],
        "deadline_scored_last_arrival_s": float(np.max(finite)) if reached.all() else float(horizon),
        "lesion_reached": reached.astype(int).tolist(),
        "first_arrival_s": [None if not np.isfinite(x) else float(x) for x in first_time],
        "all_three_reached": bool(reached.all()),
        "arrival_span_s": span,
        "tau_s": float(tau),
        "synchronized_success": bool(reached.all() and span is not None and span <= tau and lost == 0),
        "last_arrival_s": None if len(finite) != 3 else float(np.max(finite)),
        "wall_contact_s": wall,
        "spacing_pair_s": entity_pair_contact_seconds(ep),
        "spacing_violation_pair_s": float(spacing["spacing_violation_pair_s"]),
        "spacing_violation_steps": int(spacing["spacing_violation_steps"]),
        "entity_pair_contact_s": entity_pair_contact_seconds(ep),
        "tpg_gate_calls": int(gate_calls),
        "tpg_trigger_count": int(tpg_hold_steps),
        "waiting_transitions": int(waiting_transitions),
        "waiting_robot_s": float(waiting_robot_s),
        "particle_collision_events": collisions,
        "lost": lost,
        "elapsed_s": float(ep.env.elapsed_s),
        "termination_class": termination_class,
        "reached_at_deadline": bool(reached.all() and termination_class == "last_step_success"),
        "last_step_success": bool(termination_class == "last_step_success"),
        "timeout": bool(termination_class == "timeout"),
        "seconds": float(time.monotonic() - started),
    }
    ep.close()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--allocation", choices=("fixed", "geodesic", "linear_sum"), default="fixed")
    p.add_argument("--clusters", default="1,2,3")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--seed", type=int, default=2944200000)
    p.add_argument("--anatomy", default="mca_m1_lvo")
    p.add_argument("--horizon", type=float, default=300.)
    p.add_argument("--tau", type=float, default=30.)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tpg", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ccbs", action="store_true", help="fixed-route conflict search with the common gate")
    p.add_argument("--trace-seeds", type=int, nargs="*", default=[])
    a = p.parse_args()
    rows = []
    for n in [int(x) for x in a.clusters.split(",")]:
        for offset in range(a.count):
            seed = a.seed + offset
            trace = [] if seed in a.trace_seeds else None
            row = run_episode(n, seed, a.anatomy, a.horizon, a.tau, tpg=a.tpg,
                              allocation=a.allocation, ccbs=a.ccbs, trace=trace)
            if trace is not None:
                path = a.out.with_name(f"{a.out.stem}_n{n}_s{seed}_trace.json")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"episode": row, "trace": trace}) + "\n")
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {"episodes": len(rows), "tau_s": a.tau, "rows": rows, "by_clusters": {}}
    for n in sorted({r["clusters"] for r in rows}):
        group = [r for r in rows if r["clusters"] == n]
        summary["by_clusters"][str(n)] = {
            "episodes": len(group),
            "deadline_scored_last_arrival_s": float(np.mean([r["deadline_scored_last_arrival_s"] for r in group])),
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
