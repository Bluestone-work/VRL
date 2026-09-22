#!/usr/bin/env bash
# Smoke tests for the direct-local scaling features (EXP_0006-0009).
#
# Each smoke runs a tiny real training loop end-to-end (forward, PPO update,
# checkpoint save/reload) on CPU with the feature under test enabled, then a
# control run reproducing legacy behaviour. Uses the same taskset exclusion
# as the research protocol (cores 6/7 are unstable on this machine).
#
# Usage: bash scripts/run_scaling_smokes.sh [output_dir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/wj/miniconda3/envs/v/bin/python}"
OUT="${1:-$ROOT/research/runs/SCALING_SMOKES}"
AFFINITY="${AFFINITY:-0-5,8-23}"
STEPS="${STEPS:-448}"
NENVS="${NENVS:-14}"

mkdir -p "$OUT"

run() {
  local name="$1"; shift
  echo "=== smoke: $name ==="
  taskset -c "$AFFINITY" env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 "$PYTHON" "$ROOT/scripts/train_vector_mappo.py" \
    --architecture gat --control-mode local \
    --n-envs "$NENVS" --clots 3 --horizon 60 \
    --timesteps "$STEPS" --n-steps 32 --n-epochs 2 --batch-size 256 \
    --scenario-pool anatomical --tree-resample-interval 0 \
    --robot-radius 0.0011 --obs-mode geometric --reward-mode milestone \
    --contact-mode geodesic --critic-value-mode v --dropout 0.0 \
    --seed 4242 --device cpu --skip-evaluation --no-dataset \
    --save-interval "$STEPS" --eval-interval "$STEPS" \
    --log-dir "$OUT/$name" \
    "$@" | tail -3
}

# EXP_0006_VARIABLE_N: 8-slot tensors, 5 real agents, capacity 10.
run "variable_n" --robots 8 --active-robots 5 --max-agents 10

# EXP_0006 control: fixed-N, all slots real.
run "variable_n_control" --robots 5

# EXP_0007_SEPARATED_INIT: separated geodesic FPS spawn.
run "separated_init" --robots 5 --initialization-mode separated

# EXP_0007 control: legacy trunk spawn.
run "separated_init_control" --robots 5 --initialization-mode legacy

# EXP_0009_DYNAMIC_OBSTACLES: particles on.
run "dynamic_obstacles" --robots 5 --dynamic-particles --particle-count 16

# EXP_0009 control: particles off.
run "dynamic_obstacles_control" --robots 5

# EXP_0008_CONNECTIVITY_ALLOCATION is exercised by the unit tests
# (tests/test_connectivity_allocator.py); no training smoke is needed
# because the allocator changes no training-path code by itself.

echo "=== all scaling smokes passed ==="