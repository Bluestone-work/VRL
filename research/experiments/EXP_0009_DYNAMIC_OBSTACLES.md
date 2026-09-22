# EXP_0009_DYNAMIC_OBSTACLES — Blood-Cell-Inspired Intravascular Particles

## Preregistration

| Field | Value |
|---|---|
| Experiment ID | `EXP_0009_DYNAMIC_OBSTACLES` |
| Parent | `EXP_0005_DIRECT_LOCAL` |
| Research question | Can optional dynamic obstacles that advect with the local flow be added without perturbing legacy runs? |
| Main variable | `dynamic_intravascular_particles` flag (default **off**) |
| Invariants | reward unchanged (v1 obstacles exert a separation **impulse**, not a penalty); robot physics and action semantics unchanged |

## Model (first version, honest scope)

- Advection by the **same** `tree.flow` field the robots see, with the same occluded-radius override — one flow model for robots and obstacles, so obstacle motion responds to clot occlusion exactly as robot flow does.
- Small random drift scaled to the local flow speed — the off-axis tumbling proxy.
- Projected back into the lumen every step (same `tree.project` the robots use) — no wall tunnelling.
- Size: `radius_ratio` (default 1.6) × `robot_radius` — dimensionless, follows the device scale.
- **This is not a red-blood-cell model**: no haemorheology, no deformability, no near-wall lift, no rouleaux. "Blood-cell-inspired" is the ceiling of the claim; the config and code comments say so explicitly.

## Records

Per step: `particle_collisions` (overlap count per robot), `particle_clearance` (nearest distance minus contact distance), `particle_relative_speed` (|robot − nearest particle velocity|). `state_dict` round-trips particle state and its dedicated RNG for exact resume.

## Reproducibility guarantee

A **dedicated particle RNG** means enabling the flag leaves the env's own RNG stream untouched (verified: identical state and identical next draws). With the flag off, both envs reproduce legacy behaviour bit-identically. EXP_0001–EXP_0005 are therefore unaffected by default.

## Validation

`tests/test_dynamic_particles.py`: displacement aligns with local flow (mean cosine > 0.8); lumen containment over 10 steps; dimensionless size; RNG-stream preservation; flag-off bit-identity; info records present and finite; separation impulse at exact overlap; state round-trip. Smoke: `scripts/run_scaling_smokes.sh dynamic_obstacles` (448 transitions, 16 particles, finite PPO update).

## Status

Capability implemented and tested. No performance claim — collision/success impact requires new training runs with the flag enabled.
