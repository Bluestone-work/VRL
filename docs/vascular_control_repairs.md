# Vascular control repair protocol

The default finite-vessel policy is `endcap_mode="clamp"`. The root inlet and
terminal branch outlets are exposed caps. Child branch starts and anastomosis
links remain junctions, so a projection cannot create a wall at a fork. The
`open` mode reports an outlet crossing and preserves axial departure for an
explicit open-boundary experiment.

`integration_mode="legacy"` preserves the original one-step update. The
optional `adaptive` mode chooses enough substeps that the predicted displacement
does not exceed `max_substep_displacement`, checks geometry after every substep,
divides active displacement and flow by the substep count, and scales Brownian
noise by the inverse square root of that count.

Every step reports `local_flow_speed`, `available_control_speed`,
`axial_flow_speed`, `control_authority_ratio`, `cannot_reverse_flow`,
`terminal_stuck_steps`, `wall_projection_count`, `distance_to_target`, and
`geodesic_progress`. These diagnostics do not terminate an episode. Normal
truncation, successful termination, and physical projection events remain
separate fields.

`geometric_v2` keeps the original 36 geometric features unchanged and appends
log-scaled flow speed, signed route-aligned flow, control margin, time,
remaining clot mass, and crowding. The old 36-D semantics remain available with
`--obs-mode geometric`.

The PPO head uses a reparameterized tanh-Gaussian entropy estimate with the
Jacobian correction by default. `--entropy-mode legacy_cross_entropy` is an
explicit ablation that reproduces the old `-log_prob` term.

## Fixed-scene diagnosis

```bash
python -m pytest -q tests/test_vascular_control_repairs.py tests/test_vessel_geometry.py tests/test_vector_env.py
```

## Controlled training matrix

All commands below use the same seed, robot count, clot count, budget, scenario
pool, and evaluation settings. Evaluation environments always use full flow.

```bash
# A. Original Direct Local MAPPO
python scripts/train_vector_mappo.py --control-mode local --obs-mode geometric --integration-mode legacy --entropy-mode legacy_cross_entropy --seed 42 --robots 5 --clots 3 --timesteps 500000 --scenario-pool anatomical --run-dir runs/A

# B. Boundary and numerical repair
python scripts/train_vector_mappo.py --control-mode local --obs-mode geometric --integration-mode adaptive --seed 42 --robots 5 --clots 3 --timesteps 500000 --scenario-pool anatomical --run-dir runs/B

# C. B plus geometric_v2
python scripts/train_vector_mappo.py --control-mode local --obs-mode geometric_v2 --integration-mode adaptive --seed 42 --robots 5 --clots 3 --timesteps 500000 --scenario-pool anatomical --run-dir runs/C

# D. C plus corrected PPO entropy
python scripts/train_vector_mappo.py --control-mode local --obs-mode geometric_v2 --integration-mode adaptive --entropy-mode squashed_mc --seed 42 --robots 5 --clots 3 --timesteps 500000 --scenario-pool anatomical --run-dir runs/D

# E. D plus configurable flow curriculum
python scripts/train_vector_mappo.py --control-mode local --obs-mode geometric_v2 --integration-mode adaptive --entropy-mode squashed_mc --flow-curriculum --flow-boundaries 0.2 0.45 0.7 --flow-scales 0.2 0.5 0.75 1.0 --seed 42 --robots 5 --clots 3 --timesteps 500000 --scenario-pool anatomical --run-dir runs/E
```

The smoke path can be exercised on CPU with `--timesteps 16 --n-envs 4
--n-steps 2 --n-epochs 1 --batch-size 8 --skip-evaluation --no-dataset`.
