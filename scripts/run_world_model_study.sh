#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export PY=/home/wj/miniconda3/envs/v/bin/python

study_root="${1:-experiments/world_model_study_20260905_134953}"
out_dir="$study_root/world_model"
mkdir -p "$study_root/logs"

if [[ -f "$out_dir/metrics.json" ]] && "$PY" -c \
  "import json; assert json.load(open('$out_dir/metrics.json'))['gate']['passed']" \
  2>/dev/null; then
  printf 'world_model complete\n' >> "$study_root/logs/training_exit_codes.txt"
  exit 0
fi

attempt=1
while [[ -e "$study_root/logs/train_world_model_attempt_${attempt}.log" ]]; do
  attempt=$((attempt + 1))
done
if [[ -d "$out_dir" ]]; then
  mv "$out_dir" "$study_root/world_model_failed_attempt_${attempt}"
fi

log="$study_root/logs/train_world_model_attempt_${attempt}.log"
env -u DISPLAY PYTHONPATH=. PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" scripts/train_world_model.py \
  --dataset-dir \
    "$study_root/curriculum/gat_r5_c3/seed_42/dataset" \
    "$study_root/curriculum/gat_r5_c3/seed_43/dataset" \
    "$study_root/curriculum/gat_r5_c3/seed_44/dataset" \
  --dataset-start-fraction 0.5 \
  --out-dir "$out_dir" --epochs 40 --patience 10 \
  --ensemble-size 5 --hidden-dim 192 --batch-size 4096 \
  --seed 31415 --device cuda:1 2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
printf 'world_model attempt_%s train_exit %s\n' "$attempt" "$status" \
  >> "$study_root/logs/training_exit_codes.txt"
if (( status != 0 )); then
  exit "$status"
fi
if ! "$PY" -c \
  "import json; assert json.load(open('$out_dir/metrics.json'))['gate']['passed']" \
  2>/dev/null; then
  printf 'world_model attempt_%s gate_failed\n' "$attempt" \
    >> "$study_root/logs/training_exit_codes.txt"
  exit 4
fi
printf 'world_model attempt_%s gate_passed\n' "$attempt" \
  >> "$study_root/logs/training_exit_codes.txt"
