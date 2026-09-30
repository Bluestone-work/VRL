#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/wj/miniconda3/envs/v/bin/python"
OUT="$ROOT/research/runs/EXP_0015_DYNAMIC_RISK_GAT"
mkdir -p "$OUT"

run_one() {
  local seed="$1" device="$2"
  local run="$OUT/seed_${seed}"
  mkdir -p "$run"
  env PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    taskset -c 0-5,8-23 "$PY" "$ROOT/scripts/train_vector_mappo.py" \
      --run-dir "$run" --device "$device" \
      --architecture adaptive_edge_gat --control-mode local \
      --obs-mode geometric_dynamic --contact-mode geodesic --critic-value-mode v \
      --task-allocator risk_aware_connectivity --dynamic-particles \
      --particle-count 24 --particle-radius-ratio 1.6 --particle-lateral-drift 0.15 \
      --timesteps 1000000 --n-envs 64 --n-steps 128 --batch-size 2048 \
      --n-epochs 5 --robots 5 --clots 3 --horizon 300 \
      --scenario-pool anatomical --tree-resample-interval 900 \
      --no-dataset --skip-evaluation --save-interval 100000 --seed "$seed" \
      > "$OUT/seed_${seed}.stdout.log" 2>&1
  echo "seed=${seed} device=${device} exit=$?" >> "$OUT/launcher.log"
}

date -Is > "$OUT/launcher.log"
run_one 42 cuda:0 & p42=$!
run_one 43 cuda:1 & p43=$!
wait "$p42" "$p43"
run_one 44 cuda:0
date -Is >> "$OUT/launcher.log"
