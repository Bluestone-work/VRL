# Global layer: graph-based swarm configuration and planning

Code: `marl/v3/vessel_graph.py`, `marl/v3/configuration.py`; parameters: `configs/v3/assumptions.json`.

## Input

Only preoperative information: centreline stations and their adjacency, healthy lumen radius,
target (lesion) stations, and a nominal flow solution (healthy tree, 0.05 mm/s inlet). The realised
flow regime, patient-level actuation response, image latency and obstacle motion are unknown to the
planner. CT/MR segmentation is not part of this work; the simulator's geometry stands in for it.

## Graph G = (V, E, W)

* **Station graph** (used for planning). V = every simulator station; E = centreline adjacency.
  Per edge a→b: length; minimum lumen radius; direction relative to flow (downstream = away from the
  root); nominal mean flow speed u; travel time L / max(0.1, 0.9·1.0 + s·u) with s = ±1; risk time
  (lumen clearance < 3 body radii, or upstream against u > 0.3 mm/s); passability (radius > body).
* **Key graph** (reporting): root, bifurcations, leaves, targets and release candidates, with aggregated
  segment attributes (`VesselGraph.key_graph`).
* On a tree the route between two stations is unique, so "global path planning" reduces to choosing
  release points and visiting orders; shared segments and their timing are what remains to coordinate.

## Decision variables and model

N (1 ≤ N ≤ n_max = 4), release station per cluster, ordered targets per cluster, catheter release order.
The catheter enters at the root at t = 0, travels at 2 mm/s, and releases at most once per 3 s; a
cluster is released only when every released cluster is ≥ 2.5 mm from the tip. Release candidates are
the root and every bifurcation whose whole root path has lumen radius ≥ 0.6 mm; a cluster may be
released only on the root path of its first target (the catheter is advanced along the route).

Passage order is resolved by a prioritised simulation on the planning model (0.5 s step): a cluster
waits when its next move would bring it within d_safe = 2 mm of a released cluster; no progress for
60 s is a deadlock. All priority orders are tried for N ≤ 3, four heuristic orders for N = 4.

## Objective (lexicographic)

1. feasibility: all targets covered, passable routes, no deadlock, completion before the deadline;
2. J = T_pred + 1.0·W_pred + 0.5·R_pred (predicted deadline-scored completion, waiting cluster-seconds,
   risk seconds);
3. N* = smallest N with J*(N) ≤ 1.1·min_N J*(N);
4. total predicted cluster-time.

## Search

* `exact`: enumerates every N, ordered partition of the targets, release station per cluster and
  catheter order; ranks them by the admissible bound LB = T_free + 0.5·R (waiting only delays) and
  simulates in increasing bound order until LB ≥ incumbent. The certificate `exact_within_model` means
  optimal for this planning model and candidate set; a capped run is labelled `best_first_truncated`.
* `local` (proposed, scalable): farthest-point seeding on geodesic distance, nearest-seed assignment,
  best release station and visiting order per group, relocate/swap moves, three restarts, 250
  evaluations per N.

## Outputs per configuration

`Configuration` (`marl/v3/configuration.py`): N, tours, release stations, release order and planned
release times, station routes, predicted T / W / R / cluster-time / deployment time, planned potential
conflicts without coordination (layer A), shared-segment length, certificate, planning time, J for
every N, and `configuration_hash` (bound to `base_task_hash`).

Results: `CONFIGURATION_RESULTS/` and `ACCEPTANCE_REPORT.md`.
