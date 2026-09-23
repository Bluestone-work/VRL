#!/usr/bin/env bash
# EXP_0010–EXP_0013: 1M-transition training runs for the four scaling
# capabilities, each with its locked control arm.
#
#   EXP_0010  variable-N          8-slot/5-active masked policy vs fixed 5
#   EXP_0011  separated init      geodesic FPS spawn (stochastic per reset)
#   EXP_0012  dynamic obstacles   particle density ladder: sparse 8 /
#                                  moderate 16 / dense 32
#   EXP_0013  connectivity alloc  connectivity_aware allocator vs env nearest
#
# Protocol: hyperparameters/reward/action semantics identical to EXP_0005
# (research/experiments/EXP_0005_DIRECT_LOCAL.md). Seeds 42/43/44, 1M real
# transitions each, cuda:0 (GPU 0), GPU 1 left free. Sequential inside tmux.
# Checkpoints every 100k; update_metrics.jsonl per run. Re-invocation skips
# completed seeds (final_policy.pt present).
#
# Usage: bash research/scripts/run_exp0010_0013.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/wj/miniconda3/envs/v/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
OUT="$ROOT/research/runs"
AFFINITY="${AFFINITY:-0-5,8-23}"

cd "$ROOT"

BASE=(--architecture gat --control-mode local --residual-scale 0.2
      --guidance-speed 0.65 --n-envs 64 --clots 3 --horizon 300
      --timesteps 1000000 --n-steps 128 --n-epochs 5 --batch-size 2048
      --scenario-pool anatomical --tree-resample-interval 900
      --robot-radius 0.0011 --obs-mode geometric --reward-mode milestone
      --contact-mode geodesic --critic-value-mode v --dropout 0.0
      --coverage-bonus 0.2 --step-cost 0.0 --approach-scale 0.1
      --reward-double-count on --hidden-dim 128 --num-layers 2
      --num-heads 4 --lr-actor 0.0003 --lr-critic 0.001 --gamma 0.99
      --gae-lambda 0.95 --clip-epsilon 0.2 --entropy-coef 0.01
      --value-clip 10.0 --value-loss-coef 0.5 --max-grad-norm 0.5
      --log-std-init 0.0 --device "$DEVICE" --skip-evaluation
      --eval-interval 100000 --eval-episodes 5 --final-eval-episodes 20
      --eval-seed 100000 --save-interval 100000
      --dataset-shard-size 32768 --no-dataset)

run_seed() {
  local exp="$1" seed="$2"; shift 2
  local dir="$OUT/$exp/seed_$seed"
  if [[ -f "$dir/final_policy.pt" ]]; then
    echo "=== $exp seed $seed already complete, skipping ==="
    return 0
  fi
  echo "=== $exp seed $seed: start $(date '+%F %T') ==="
  mkdir -p "$dir"
  taskset -c "$AFFINITY" env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 "$PYTHON" scripts/train_vector_mappo.py \
    "${BASE[@]}" --seed "$seed" --run-dir "$dir" \
    --log-dir "$OUT/$exp" "$@" \
    > "$dir/train_stdout.log" 2>&1
  echo "=== $exp seed $seed: done $(date '+%F %T') rc=$? ==="
}

for SEED in 42 43 44; do
  # EXP_0010_VARIABLE_N — masked 8-slot policy, 5 real agents.
  run_seed EXP_0010_VARIABLE_N "$SEED" \
    --robots 8 --active-robots 5 --max-agents 10
  # Control — legacy fixed 5 (matches EXP_0005 arm).
  run_seed EXP_0010_CONTROL "$SEED" --robots 5
done

for SEED in 42 43 44; do
  # EXP_0011_SEPARATED_INIT — stochastic geodesic FPS spawn per reset.
  run_seed EXP_0011_SEPARATED_INIT "$SEED" \
    --robots 5 --initialization-mode separated
  # Control — legacy trunk spawn.
  run_seed EXP_0011_CONTROL "$SEED" --robots 5
done

# EXP_0012_DYNAMIC_OBSTACLES — density ladder (sparse/8, moderate/16,
# dense/32 particles). Shared control: particles off (legacy physics).
for SEED in 42 43 44; do
  run_seed EXP_0012_PARTICLES_SPARSE "$SEED" \
    --robots 5 --dynamic-particles --particle-count 8
  run_seed EXP_0012_PARTICLES_MODERATE "$SEED" \
    --robots 5 --dynamic-particles --particle-count 16
  run_seed EXP_0012_PARTICLES_DENSE "$SEED" \
    --robots 5 --dynamic-particles --particle-count 32
  run_seed EXP_0012_CONTROL "$SEED" --robots 5
done

for SEED in 42 43 44; do
  # EXP_0013_CONNECTIVITY_ALLOC — allocator in the loop, per-step replan.
  run_seed EXP_0013_CONNECTIVITY_ALLOC "$SEED" \
    --robots 5 --task-allocator connectivity_aware
  # Control — env's own nearest-clot rule (legacy path).
  run_seed EXP_0013_CONTROL "$SEED" --robots 5
done

echo "=== EXP_0010–EXP_0013 all complete $(date '+%F %T') ==="
