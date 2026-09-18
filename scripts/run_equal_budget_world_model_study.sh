#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export PY=/home/wj/miniconda3/envs/v/bin/python

study_root="${1:-experiments/world_model_study_20260905_134953}"
world_model="${2:-$study_root/world_model/best_world_model.pt}"
world_metrics="$(dirname "$world_model")/metrics.json"
mkdir -p "$study_root/logs" "$study_root/comparisons"

if [[ ! -f "$world_model" || ! -f "$world_metrics" ]]; then
  printf 'missing world model or metrics: %s\n' "$world_model" >&2
  exit 2
fi
if ! "$PY" -c \
  "import json; assert json.load(open('$world_metrics'))['gate']['passed']" 2>/dev/null; then
  printf 'world-model validation gate did not pass: %s\n' "$world_metrics" >&2
  exit 3
fi

uncertainty_threshold="$($PY -c \
  "import json; print(json.load(open('$world_metrics'))['one_step']['uncertainty_p95'])")"

run_arm() {
  local seed="$1"
  local arm="$2"
  local gpu="$3"
  local curriculum_dir="$study_root/curriculum/gat_r5_c3/seed_${seed}"
  local source=""
  local run_dir="$study_root/comparisons/seed_${seed}/${arm}"
  local attempt=0
  local attempts_this_run=0
  local max_attempts_this_run=12
  local previous_logs=("$study_root/logs/train_comparison_${arm}_seed_${seed}_attempt_"*.log)

  source="$($PY -c \
    "import json,pathlib; d=pathlib.Path('$curriculum_dir'); rows=[json.loads(x) for x in (d/'eval_metrics.jsonl').read_text().splitlines() if x]; valid=[r for r in rows if (d/f\"checkpoint_{int(r['transitions'])}.pt\").exists()]; best=max(valid,key=lambda r:(r['macro']['success'],r['macro']['removal_rate'])); print(d/f\"checkpoint_{int(best['transitions'])}.pt\")")"
  if [[ -z "$source" || ! -f "$source" ]]; then
    printf 'missing validation-selected curriculum checkpoint for seed %s\n' "$seed" >&2
    return 2
  fi
  local source_transitions target_transitions
  source_transitions="$(basename "$source" | sed -E 's/checkpoint_([0-9]+)\.pt/\1/')"
  target_transitions=$((source_transitions + 250000))
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
      "import json; d=json.load(open('$run_dir/summary.json')); assert d['real_transitions'] >= $target_transitions and d['added_real_transitions'] >= 250000" \
      2>/dev/null; then
      printf 'comparison_%s %s complete %s\n' "$arm" "$seed" "$attempt" \
        >> "$study_root/logs/training_exit_codes.txt"
      return 0
    fi

    attempt=$((attempt + 1))
    attempts_this_run=$((attempts_this_run + 1))
    local resume="$source"
    local latest_checkpoint=""
    if [[ -d "$run_dir" ]]; then
      latest_checkpoint="$(find "$run_dir" -maxdepth 1 -name 'checkpoint_*.pt' | sort -V | tail -1)"
    fi
    if [[ -n "$latest_checkpoint" ]]; then
      resume="$latest_checkpoint"
      local checkpoint_transitions origin_transitions keep
      checkpoint_transitions="$(basename "$latest_checkpoint" | sed -E 's/checkpoint_([0-9]+)\.pt/\1/')"
      origin_transitions="$($PY -c \
        "import json; print(json.load(open('$run_dir/run_origin.json'))['initial_real_transitions'])")"
      keep=$((checkpoint_transitions - origin_transitions))
      if [[ -f "$run_dir/dataset/manifest.jsonl" ]]; then
        "$PY" scripts/recover_transition_dataset.py \
          --dataset-dir "$run_dir/dataset" --keep-transitions "$keep" \
          > "$study_root/logs/recover_comparison_${arm}_seed_${seed}_attempt_${attempt}.log"
      fi
    elif [[ -f "$run_dir/dataset/manifest.jsonl" ]]; then
      "$PY" scripts/recover_transition_dataset.py \
        --dataset-dir "$run_dir/dataset" --keep-transitions 0 \
        > "$study_root/logs/recover_comparison_${arm}_seed_${seed}_attempt_${attempt}.log"
    fi

    local model_args=()
    if [[ "$arm" == "mve" ]]; then
      model_args=(
        --world-model "$world_model"
        --imagination-horizon 3 --imagination-blend 0.25
        --imagination-uncertainty "$uncertainty_threshold"
      )
    fi
    local log="$study_root/logs/train_comparison_${arm}_seed_${seed}_attempt_${attempt}.log"
    env -u DISPLAY PYTHONPATH=. PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$PY" scripts/train_vector_mappo.py \
      --architecture gat --run-dir "$run_dir" --resume "$resume" \
      --n-envs 64 --robots 5 --clots 3 --horizon 300 \
      --timesteps "$target_transitions" --n-steps 128 --n-epochs 5 --batch-size 4096 \
      --scenario-pool anatomical --tree-resample-interval 900 \
      --robot-radius 0.0011 --reward-mode milestone \
      --device "cuda:${gpu}" --seed "$seed" \
      --eval-interval 100000 --eval-episodes 5 --final-eval-episodes 20 \
      --eval-seed 300000 --save-interval 100000 \
      --dataset-shard-size 32768 \
      "${model_args[@]}" 2>&1 | tee "$log"
    local status="${PIPESTATUS[0]}"
    printf 'comparison_%s %s attempt_%s %s\n' "$arm" "$seed" "$attempt" "$status" \
      >> "$study_root/logs/training_exit_codes.txt"
    if (( status == 0 )); then
      return 0
    fi
  done
  return 1
}

failed=0
for seed in 42 43 44; do
  run_arm "$seed" pure 0 &
  pure_pid=$!
  run_arm "$seed" mve 1 &
  mve_pid=$!
  wait "$pure_pid" || failed=1
  wait "$mve_pid" || failed=1
done
exit "$failed"
