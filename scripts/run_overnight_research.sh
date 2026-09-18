#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export PY=/home/wj/miniconda3/envs/v/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1

ROOT="${1:-experiments/overnight_research_20260905}"
mkdir -p "$ROOT/logs" "$ROOT/training" "$ROOT/world_models" "$ROOT/media"

run_training() {
  local label="$1" architecture="$2" control="$3" seed="$4" gpu="$5" budget="$6"
  local run_dir="$ROOT/training/${label}/seed_${seed}"
  local log_prefix="$ROOT/logs/train_${label}_seed_${seed}"
  mkdir -p "$run_dir"
  local curriculum_args=()
  if [[ "$label" == "curriculum_flow" ]]; then
    curriculum_args=(--curriculum --curriculum-boundaries 0.2 0.5 --curriculum-difficulties 0.2 0.6 1.0)
  fi
  if [[ -f "$run_dir/summary.json" ]] && "$PY" -c "import json; assert json.load(open('$run_dir/summary.json'))['real_transitions'] >= $budget" 2>/dev/null; then
    echo "already complete $label $seed" >> "$ROOT/logs/orchestrator.log"
    return 0
  fi
  local latest="$(find "$run_dir" -maxdepth 1 -name 'checkpoint_*.pt' | sort -V | tail -n 1)"
  local resume=()
  if [[ -n "$latest" ]]; then resume=(--resume "$latest"); fi
  for attempt in $(seq 1 12); do
    local log="${log_prefix}_attempt_${attempt}.log"
    printf '[%s] start label=%s seed=%s gpu=%s attempt=%s resume=%s\n' "$(date '+%F %T')" "$label" "$seed" "$gpu" "$attempt" "$latest" >> "$ROOT/logs/orchestrator.log"
    env -u DISPLAY CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. \
      "$PY" scripts/train_vector_mappo.py \
      --architecture "$architecture" --control-mode "$control" \
      --seed "$seed" --device cuda:0 --n-envs 64 --robots 5 --clots 3 --horizon 300 \
      --timesteps "$budget" --n-steps 128 --n-epochs 5 --batch-size 2048 \
      --scenario-pool anatomical --robot-radius 0.0011 --tree-resample-interval 900 \
      --eval-interval 100000 --eval-episodes 3 --final-eval-episodes 5 \
      --skip-evaluation \
      --eval-seed 400000 --save-interval 100000 --dataset-shard-size 32768 \
      --run-dir "$run_dir" "${curriculum_args[@]}" "${resume[@]}" > "$log" 2>&1
    local status=$?
    printf '[%s] finish label=%s seed=%s attempt=%s exit=%s\n' "$(date '+%F %T')" "$label" "$seed" "$attempt" "$status" >> "$ROOT/logs/orchestrator.log"
    if (( status == 0 )); then return 0; fi
    if [[ "$status" != 139 && "$status" != 134 && "$status" != 132 && "$status" != 135 ]]; then
      tail -n 40 "$log" >> "$ROOT/logs/orchestrator.log"
      return "$status"
    fi
    latest="$(find "$run_dir" -maxdepth 1 -name 'checkpoint_*.pt' | sort -V | tail -n 1)"
    resume=(); if [[ -n "$latest" ]]; then resume=(--resume "$latest"); fi
  done
  return 1
}

run_mve() {
  local source="$ROOT/training/world_gat/seed_43/best_policy.pt"
  local world_model="$ROOT/world_models/world_gat_model/best_world_model.pt"
  local run_dir="$ROOT/training/world_mve/seed_43"
  mkdir -p "$run_dir"
  [[ -f "$source" && -f "$world_model" ]] || return 2
  if [[ -f "$run_dir/summary.json" ]] && "$PY" -c "import json; assert json.load(open('$run_dir/summary.json'))['real_transitions'] >= 750000" 2>/dev/null; then return 0; fi
  local latest="$(find "$run_dir" -maxdepth 1 -name 'checkpoint_*.pt' | sort -V | tail -n 1)"
  local resume="$source"
  [[ -n "$latest" ]] && resume="$latest"
  for attempt in $(seq 1 8); do
    local log="$ROOT/logs/train_world_mve_seed_43_attempt_${attempt}.log"
    env -u DISPLAY CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
      "$PY" scripts/train_vector_mappo.py \
      --architecture gat --control-mode world --seed 43 --device cuda:0 \
      --n-envs 64 --robots 5 --clots 3 --horizon 300 --timesteps 750000 \
      --n-steps 128 --n-epochs 5 --batch-size 2048 --scenario-pool anatomical \
      --robot-radius 0.0011 --tree-resample-interval 900 --eval-interval 100000 \
      --eval-episodes 3 --final-eval-episodes 5 --eval-seed 400000 \
      --skip-evaluation \
      --save-interval 100000 --run-dir "$run_dir" --resume "$resume" \
      --world-model "$world_model" --imagination-horizon 3 \
      --imagination-blend 0.25 --imagination-uncertainty 0.05 > "$log" 2>&1
    local status=$?
    if (( status == 0 )); then return 0; fi
    latest="$(find "$run_dir" -maxdepth 1 -name 'checkpoint_*.pt' | sort -V | tail -n 1)"
    [[ -n "$latest" ]] && resume="$latest"
    if [[ "$status" != 139 && "$status" != 134 && "$status" != 132 && "$status" != 135 ]]; then return "$status"; fi
  done
  return 1
}

run_world_model() {
  local label="$1" dataset="$2" gpu="$3" action_field="$4"
  local out="$ROOT/world_models/$label"
  local log="$ROOT/logs/world_model_${label}.log"
  if [[ -f "$out/metrics.json" ]]; then return 0; fi
  for attempt in $(seq 1 6); do
    env -u DISPLAY CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. \
      "$PY" scripts/train_world_model.py \
      --dataset-dir "$dataset" --action-field "$action_field" --out-dir "$out" \
      --epochs 40 --patience 10 --ensemble-size 5 --hidden-dim 192 \
      --batch-size 4096 --seed 31415 --device cuda:0 > "${log}_attempt_${attempt}" 2>&1
    local status=$?
    if (( status == 0 )) && "$PY" -c "import json; assert json.load(open('$out/metrics.json'))['gate']['passed']" 2>/dev/null; then
      echo "world model complete $label" >> "$ROOT/logs/orchestrator.log"
      return 0
    fi
    if [[ -d "$out" ]]; then mv "$out" "${out}_failed_${attempt}"; fi
    if [[ "$status" != 139 && "$status" != 134 && "$status" != 132 && "$status" != 135 ]]; then break; fi
  done
  return 1
}

run_training flow_guided_gat gat flow_guided 43 0 1000000 & p1=$!
run_training edge_bias_flow edge_bias_gat flow_guided 43 1 1000000 & p2=$!
wait "$p1" || echo 'flow_guided_gat failed' >> "$ROOT/logs/orchestrator.log"
wait "$p2" || echo 'edge_bias_flow failed' >> "$ROOT/logs/orchestrator.log"

run_training world_gat gat world 43 0 500000 & p3=$!
run_training curriculum_flow gat flow_guided 44 1 1000000 & p4=$!
wait "$p3" || echo 'world_gat failed' >> "$ROOT/logs/orchestrator.log"
wait "$p4" || echo 'curriculum_flow failed' >> "$ROOT/logs/orchestrator.log"

run_world_model world_gat_model "$ROOT/training/world_gat/seed_43/dataset" 1 action || true
run_mve || echo 'world_mve failed' >> "$ROOT/logs/orchestrator.log"

echo "research phase complete $(date '+%F %T')" >> "$ROOT/logs/orchestrator.log"
