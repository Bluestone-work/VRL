#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PY=/home/wj/miniconda3/envs/v/bin/python

mode="${1:?control mode required}"
seed="${2:?seed required}"
gpu="${3:-0}"
budget="${4:-500000}"
study="${5:-experiments/success_study_20260905}"
run_dir="$study/training/${mode}/seed_${seed}"
mkdir -p "$run_dir" "$study/logs"

for attempt in $(seq 1 12); do
    resume=()
    latest="$(find "$run_dir" -maxdepth 1 -name 'checkpoint_*.pt' | sort -V | tail -n 1)"
    if [[ -n "$latest" ]]; then
        resume=(--resume "$latest")
    fi
    log="$study/logs/train_${mode}_${seed}_$(date +%Y%m%d_%H%M%S)_attempt_${attempt}.log"
    printf '%s\n' "mode=$mode seed=$seed budget=$budget resume=$latest" >> "$study/logs/training_runs.log"
    if env -u DISPLAY PYTHONPATH=. PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" \
        "$PY" scripts/train_vector_mappo.py \
        --architecture gat --control-mode "$mode" \
        --seed "$seed" --n-envs 64 --robots 5 --clots 3 --horizon 300 \
        --timesteps "$budget" --n-steps 128 --n-epochs 5 --batch-size 2048 \
        --scenario-pool anatomical --robot-radius 0.0011 --tree-resample-interval 900 \
        --eval-interval 100000 --eval-episodes 3 --final-eval-episodes 3 \
        --eval-seed 400000 --save-interval 50000 --no-dataset \
        --run-dir "$run_dir" "${resume[@]}" > "$log" 2>&1; then
        printf 'complete %s %s %s\n' "$mode" "$seed" "$attempt" >> "$study/logs/training_runs.log"
        exit 0
    else
        code=$?
        printf 'failed %s %s %s exit=%s\n' "$mode" "$seed" "$attempt" "$code" >> "$study/logs/training_runs.log"
        if [[ "$code" != 139 && "$code" != 134 && "$code" != 132 && "$code" != 135 ]]; then
            tail -n 30 "$log" >&2
            exit "$code"
        fi
    fi
done
exit 1
