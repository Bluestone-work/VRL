#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export PY=/home/wj/miniconda3/envs/v/bin/python

study_root="${1:-experiments/world_model_study_20260905_134953}"
mkdir -p "$study_root/logs" "$study_root/baselines"

jobs=(
  "gat 42" "gat 43" "gat 44"
  "edge_bias_gat 42" "edge_bias_gat 43" "edge_bias_gat 44"
  "mlp 42" "mlp 43" "mlp 44"
)

run_one() {
  local architecture="$1"
  local seed="$2"
  local gpu="$3"
  local run_dir="$study_root/baselines/${architecture}_r5_c3/seed_${seed}"
  local attempt=0
  local attempts_this_run=0
  local max_attempts_this_run=8

  # Continue numbering across runner invocations so recovery evidence is never
  # overwritten after a native crash exhausts one batch of attempts.
  local previous_logs=("$study_root/logs/train_${architecture}_seed_${seed}_attempt_"*.log)
  if [[ -e "${previous_logs[0]}" ]]; then
    for previous_log in "${previous_logs[@]}"; do
      local previous_attempt
      previous_attempt="$(basename "$previous_log" | sed -E 's/.*_attempt_([0-9]+)\.log/\1/')"
      if [[ "$previous_attempt" =~ ^[0-9]+$ ]] && (( previous_attempt > attempt )); then
        attempt="$previous_attempt"
      fi
    done
  fi

  while (( attempts_this_run < max_attempts_this_run )); do
    if [[ -f "$run_dir/summary.json" ]] && "$PY" -c \
      "import json; assert json.load(open('$run_dir/summary.json'))['real_transitions'] >= 500000" \
      2>/dev/null; then
      printf '%s %s complete %s\n' "$architecture" "$seed" "$attempt" \
        >> "$study_root/logs/training_exit_codes.txt"
      return 0
    fi

    attempt=$((attempt + 1))
    attempts_this_run=$((attempts_this_run + 1))
    local resume_args=()
    local latest_checkpoint=""
    if [[ -d "$run_dir" ]]; then
      latest_checkpoint="$(find "$run_dir" -maxdepth 1 -name 'checkpoint_*.pt' | sort -V | tail -1)"
    fi
    if [[ -n "$latest_checkpoint" ]]; then
      local keep
      keep="$(basename "$latest_checkpoint" | sed -E 's/checkpoint_([0-9]+)\.pt/\1/')"
      if [[ -f "$run_dir/dataset/manifest.jsonl" ]]; then
        "$PY" scripts/recover_transition_dataset.py \
          --dataset-dir "$run_dir/dataset" --keep-transitions "$keep" \
          > "$study_root/logs/recover_${architecture}_seed_${seed}_attempt_${attempt}.log"
      fi
      resume_args=(--resume "$latest_checkpoint")
    elif [[ -f "$run_dir/dataset/manifest.jsonl" ]]; then
      "$PY" scripts/recover_transition_dataset.py \
        --dataset-dir "$run_dir/dataset" --keep-transitions 0 \
        > "$study_root/logs/recover_${architecture}_seed_${seed}_attempt_${attempt}.log"
    fi

    local log="$study_root/logs/train_${architecture}_seed_${seed}_attempt_${attempt}.log"
    env -u DISPLAY PYTHONPATH=. PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$PY" scripts/train_vector_mappo.py \
      --architecture "$architecture" \
      --n-envs 64 --robots 5 --clots 3 --horizon 300 \
      --timesteps 500000 --n-steps 128 --n-epochs 5 --batch-size 4096 \
      --scenario-pool anatomical --tree-resample-interval 900 \
      --robot-radius 0.0011 --reward-mode milestone \
      --device "cuda:${gpu}" --seed "$seed" \
      --eval-interval 100000 --eval-episodes 5 --final-eval-episodes 20 \
      --eval-seed 100000 --save-interval 100000 \
      --dataset-shard-size 32768 --log-dir "$study_root/baselines" \
      "${resume_args[@]}" 2>&1 | tee "$log"
    local status="${PIPESTATUS[0]}"
    printf '%s %s attempt_%s %s\n' "$architecture" "$seed" "$attempt" "$status" \
      >> "$study_root/logs/training_exit_codes.txt"
    if (( status == 0 )); then
      return 0
    fi
  done
  return 1
}

failed=0
for ((index=0; index<${#jobs[@]}; index+=2)); do
  read -r architecture_a seed_a <<< "${jobs[index]}"
  run_one "$architecture_a" "$seed_a" 0 &
  pid_a=$!

  pid_b=""
  if (( index + 1 < ${#jobs[@]} )); then
    read -r architecture_b seed_b <<< "${jobs[index + 1]}"
    run_one "$architecture_b" "$seed_b" 1 &
    pid_b=$!
  fi

  wait "$pid_a" || failed=1
  if [[ -n "$pid_b" ]]; then
    wait "$pid_b" || failed=1
  fi
done

exit "$failed"
