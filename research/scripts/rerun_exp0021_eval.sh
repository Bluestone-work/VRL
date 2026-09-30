#!/usr/bin/env bash
# EXP_0021 evaluation rerun (fixed action semantics, 2026-09-28).
# The first evaluation pass fed raw Frenet proposals to env.step (world-frame
# velocity semantics) — 18 eval dirs quarantined in eval_invalid_action_semantics_20260928/.
# Training is untouched (training always used direct_local_action); this rerun
# re-evaluates the existing checkpoints with the corrected worker only.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/wj/miniconda3/envs/v/bin/python"
OUT="$ROOT/research/runs/EXP_0021_TAIL_REPAIR_20260927a"
GRU="$ROOT/research/runs/EXP16_20_FORMAL_20260927a/jobs"
AFF="0-5,8-23"
export PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1

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

echo "EXP_0021 eval rerun (fixed Frenet transform) $(date -Is)" >> "$OUT/driver.log"

for arm in repair base; do
  for seed in 42 43 44; do
    for ts in 1003520 2007040 3000032; do
      ck="$OUT/${arm}_s${seed}/policy_${ts}.pt"
      [ -f "$ck" ] && run_eval "$arm" "$seed" "$ts" "$ck"
    done
  done
done
echo "EXP_0021 eval rerun complete $(date -Is)" >> "$OUT/driver.log"
