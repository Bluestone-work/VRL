# EXP_0007_SEPARATED_INIT — Geodesic Farthest-Point Spawn

## Preregistration

| Field | Value |
|---|---|
| Experiment ID | `EXP_0007_SEPARATED_INIT` |
| Parent | `EXP_0005_DIRECT_LOCAL` |
| Research question | Does geodesic FPS initialization spread the swarm while satisfying joint Euclidean + geodesic separation, with a bounded recorded relaxation? |
| Main variable | `initialization_mode="separated"` (spawn rule only) |
| Invariants | reward, observation, physics, action semantics unchanged; `legacy` mode bit-identical |

## Mechanism

- Candidates: all stations on real branches, excluding the distal cap. Seed station drawn uniformly from the proximal third.
- Selection: geodesic farthest-point sampling — each new spawn maximises the minimum geodesic distance to the already-chosen ones, with a joint **Euclidean** constraint (min 8 robot radii, dimensionless relative to the device) and **geodesic** constraint (min 0.12 of total arclength).
- **Relaxation ladder** (bounded, recorded, no retry loop): R0 strict → R1 both minimums halved → R2 geodesic-only → R3 unconstrained FPS (always terminates). The level used and the achieved minima are reported per reset (`info["separated_relaxation_level"]`, `separated_min_euclidean`, `separated_min_geodesic`).
- Vector env: same rule for the shared tree, per-reset level exposed in step info.

## Scope statement

The Euclidean minimum is a **geometric proxy for disjoint local control regions**. This repository has no magnetic-field model; this experiment does not claim to simulate one.

## Validation

- `tests/test_separated_initialization.py`: constraints hold at level 0 across scenarios; infeasible constraints relax through bounded recorded levels without deadlock; separated spread beats the legacy trunk cluster; legacy mode unchanged; vector env mode works.
- Measured (14 anatomical scenarios × 8 episodes, 5 robots, default constraints): **112/112 resets at level 0**; min pairwise Euclidean 0.0715 (required 0.0176); min geodesic fraction 0.196 (required 0.12).
- Smoke: `scripts/run_scaling_smokes.sh separated_init`.

## Status

Capability implemented and tested. No performance claim — success impact requires new training runs.
