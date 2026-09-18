#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export PY=/home/wj/miniconda3/envs/v/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

ROOT="${ROOT:-experiments/hierarchical_marl_research_20260906}"
TASK_DEVICE="${TASK_DEVICE:-cpu}"
INITIALIZATION_MODE="${INITIALIZATION_MODE:-stratified}"
SCENARIO_OFFSET="${SCENARIO_OFFSET:-0}"
SCENARIO_STRIDE="${SCENARIO_STRIDE:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
METHODS="${METHODS:-nearest_flow,hungarian_flow,hierarchical_v2_seed44}"
SCENARIOS="${SCENARIOS:-all}"

mkdir -p "$ROOT/media_fast" "$ROOT/media_slow" "$ROOT/logs/media"

if [[ "$SCENARIOS" == "all" ]]; then
  mapfile -t scenario_list < <("$PY" - <<'PY'
from environments.vessel_geometry import ALL_SCENARIOS
print("\n".join(ALL_SCENARIOS))
PY
)
else
  IFS=',' read -r -a scenario_list <<< "$SCENARIOS"
fi

method_args() {
  case "$1" in
    nearest_flow) printf '%s\n' --controller flow_guided --task-allocator none ;;
    hungarian_flow) printf '%s\n' --controller flow_guided --task-allocator hungarian ;;
    hierarchical_v2_seed44)
      printf '%s\n' --controller flow_guided --task-allocator learned \
        --task-checkpoint "$ROOT/training/hierarchical_allocator_v2/seed_44/best_allocator.pt" \
        --device "$TASK_DEVICE"
      ;;
    hierarchical_v3_seed46)
      printf '%s\n' --policy experiments/success_study_20260905/selection/flow_guided_43.pt \
        --architecture gat --task-allocator learned \
        --task-checkpoint "$ROOT/training/hierarchical_allocator_v3/seed_46/best_allocator.pt" \
        --device "$TASK_DEVICE"
      ;;
    *) return 2 ;;
  esac
}

render_one() {
  local method="$1" scenario="$2" extension="$3" speed="$4"
  local output_root="$ROOT/media_${speed}"
  local output="$output_root/$method/$scenario/$method.$extension"
  local logdir="$ROOT/logs/media/$method/$scenario"
  mkdir -p "$logdir"
  [[ -s "$output" ]] && return 0
  local max_frames=300 stride=2 fps=12
  if [[ "$speed" == slow ]]; then
    max_frames=1500
    stride=1
    fps=5
  fi
  mapfile -t allocator_args < <(method_args "$method")
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    env -u DISPLAY PYTHONPATH=. \
      "$PY" make_gif.py "${allocator_args[@]}" \
      --scenario "$scenario" --seed 920000 --robots 5 --clots 3 \
      --horizon 1500 --robot-radius 0.0011 --reward-mode milestone \
      --initialization-mode "$INITIALIZATION_MODE" \
      --views triple --layout bottom --wall-opacity 0.10 --spin 0.0 \
      --max-frames "$max_frames" --stride "$stride" --fps "$fps" \
      --width 1280 --height 720 --out "$output" \
      >"$logdir/${speed}_${extension}_attempt_${attempt}.log" 2>&1
    local status=$?
    if (( status == 0 )); then return 0; fi
    if [[ "$status" != 132 && "$status" != 134 && "$status" != 135 && "$status" != 139 ]]; then
      break
    fi
  done
  printf '%s\t%s\t%s\t%s\n' "$method" "$scenario" "$speed" "$extension" \
    >> "$ROOT/logs/media_failed.tsv"
  return 1
}

scenario_index=0
for scenario in "${scenario_list[@]}"; do
  if (( scenario_index % SCENARIO_STRIDE != SCENARIO_OFFSET )); then
    scenario_index=$((scenario_index + 1))
    continue
  fi
  IFS=',' read -r -a method_list <<< "$METHODS"
  for method in "${method_list[@]}"; do
    for speed in fast slow; do
      for extension in gif mp4; do
        render_one "$method" "$scenario" "$extension" "$speed" || true
      done
    done
  done
  scenario_index=$((scenario_index + 1))
done

"$PY" - "$ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = {}
for speed in ("fast", "slow"):
    files = sorted((root / f"media_{speed}").glob("*/*/*"))
    manifest[speed] = {
        "gif_count": sum(path.suffix == ".gif" for path in files),
        "mp4_count": sum(path.suffix == ".mp4" for path in files),
        "files": [str(path.relative_to(root)) for path in files if path.suffix in {".gif", ".mp4"}],
    }
(root / "manifests" / "media_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)
print(json.dumps({key: {name: value for name, value in data.items() if name.endswith("count")} for key, data in manifest.items()}))
PY
