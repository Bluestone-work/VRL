# EXP_0008_CONNECTIVITY_ALLOCATION — Corridor-Overlap-Aware Task Allocation

## Preregistration

| Field | Value |
|---|---|
| Experiment ID | `EXP_0008_CONNECTIVITY_ALLOCATION` |
| Parent | `EXP_0005_DIRECT_LOCAL` |
| Research question | Does a connectivity/contiguity-inspired allocator reduce unnecessary shared corridors while respecting clot capacity, vs `nearest` and `flow_spread`? |
| Main variable | Task allocation rule only |
| Invariants | reward unchanged; the allocator emits **targets only** — the final 3D action remains the Direct Local GAT-MAPPO output (`direct_local_frenet`) |

## Reference and adaptation

C²-Explorer keeps a communicating team connected and spreads robots over *contiguous* regions. The vascular adaptation: each clot is a task, and assignments should minimise **unnecessary** corridor sharing. Strictly disjoint paths are not the goal — the trunk is unavoidable — the goal is to avoid excess sharing and congestion. C²-Explorer's communication-connectivity constraint has no vascular counterpart and is replaced by corridor overlap. This is an adaptation, not an implementation of C²-Explorer.

## Cost terms (per robot–clot pair)

| Term | Meaning |
|---|---|
| geodesic distance | along-vessel travel, normalised by total arclength |
| path overlap | fraction of the candidate corridor already used by previously assigned routes |
| edge congestion | penalty for edges used by >1 assigned route (the first trunk user is free) |
| flow cost | fraction of local flow opposing the route direction |
| switching penalty | cost of abandoning the robot's previous target (anti-thrash) |
| clot capacity | soft overflow beyond `lysis_saturation` (4 robots); hard respect while an unsaturated clot exists |

Solver: greedy sequential allocation in increasing geodesic-distance order. Overlap couples rows (row *i*'s cost depends on rows < *i*), so a global Hungarian assignment is not applicable; greedy is O(R·C) route lookups and deterministic.

## Application

`env.set_task_assignments(...)` / vector-env equivalent overrides the nearest-clot rule with a live-clot guard (a planner cannot steer robots at a cleared clot); auto-reset invalidates stale rows. The route next-hop field can serve as a corridor hint; the planner never emits actions.

## Measured (static scenes, 14 anatomical scenarios × 20 episodes, 8 robots, 3 clots)

| Allocator | Mean route-overlap cost |
|---|---|
| nearest | 3.433 |
| flow_spread | 2.856 |
| **connectivity_aware** | **2.787** |

connectivity_aware beats `nearest` on 14/14 scenarios and `flow_spread` on 8/14 (distance pressure dominates the rest). 5-robot snapshot: 1.959 / 1.649 / 1.653.

## Validation

`tests/test_connectivity_allocator.py`: capacity respected when an alternative exists; single live clot → all robots assigned (totality); overlap ≤ nearest everywhere and ≤ flow_spread × 1.02 on average; switching penalty stabilises assignments (≥5/6 stable); dispatch validation; vector-env override + reset guards.

## Status

Allocator implemented and measured on static scenes. **No closed-loop success claim**: that requires training/evaluation runs with the allocator in the loop.
