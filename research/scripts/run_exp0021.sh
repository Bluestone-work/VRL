#!/usr/bin/env bash
# EXP_0021 driver: matched base vs repair (no-progress early truncation K=50).
# Authorized 2026-09-27 by the user (choice: variable (b), then "2" = launch).
# Two arms × 3 seeds × 3M transitions; GRU predictor per seed reused from the
# EXP16-20 formal batch (motion_fit_s4x, gate-passed); checkpoints at 1M/2M/3M;
# dev-validation-v2 evaluation (140 episodes per checkpoint).
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/wj/miniconda3/envs/v/bin/python"
OUT="$ROOT/research/runs/EXP_0021_TAIL_REPAIR_20260927a"
GRU="$ROOT/research/runs/EXP16_20_FORMAL_20260927a/jobs"
mkdir -p "$OUT"
AFF="0-5,8-23"
export PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1

run_train() {  # arm seed device
  local arm="$1" seed="$2" device="$3"
  local run="$OUT/${arm}_s${seed}"
  [ -f "$run/summary.json" ] && { echo "skip train $arm/$seed (done)" >> "$OUT/driver.log"; return; }
  taskset -c $AFF "$PY" "$ROOT/scripts/exp21_worker.py" train \
    --arm "$arm" --seed "$seed" --device "$device" --out "$run" \
    --predictor "$GRU/motion_fit_s${seed}_attempt1/model.pt" \
    --n-envs 56 --timesteps 3000000 --checkpoint-interval 1000000 \
    --epochs 5 --stall-limit 50 \
    >> "$OUT/driver.log" 2>&1
  echo "train $arm/$seed rc=$? $(date -Is)" >> "$OUT/driver.log"
}

run_eval() {  # arm seed transitions checkpoint
  local arm="$1" seed="$2" ts="$3" ckpt="$4"
  local run="$OUT/eval_${arm}_s${seed}_${ts}"
  [ -f "$run/summary.json" ] && { echo "skip eval $arm/$seed/$ts (done)" >> "$OUT/driver.log"; return; }
  taskset -c $AFF "$PY" "$ROOT/scripts/exp21_worker.py" evaluate \
    --arm "$arm" --seed "$seed" --device cuda:0 --out "$run" \
    --checkpoint "$ckpt" \
    --predictor "$GRU/motion_fit_s${seed}_attempt1/model.pt" \
    --stall-limit 50 --eval-episodes 10 \
    >> "$OUT/driver.log" 2>&1
  echo "eval $arm/$seed/$ts rc=$? $(date -Is)" >> "$OUT/driver.log"
}

echo "EXP_0021 relaunch with GRU predictor $(date -Is)" >> "$OUT/driver.log"

# Train: repair on GPU0, base on GPU1 (per-seed alternation, serial per line).
run_train repair 42 cuda:0
run_train base   42 cuda:1
run_train repair 43 cuda:0
run_train base   43 cuda:1
run_train repair 44 cuda:0
run_train base   44 cuda:1

# Evaluations after training (fixed cuda:0).
for arm in repair base; do
  for seed in 42 43 44; do
    for ts in 1003520 2007040 3000032; do
      ck="$OUT/${arm}_s${seed}/policy_${ts}.pt"
      [ -f "$ck" ] && run_eval "$arm" "$seed" "$ts" "$ck"
    done
  done
done

echo "EXP_0021 complete $(date -Is)" >> "$OUT/driver.log"
