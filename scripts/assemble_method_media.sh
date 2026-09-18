#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export PY=/home/wj/miniconda3/envs/v/bin/python
ROOT="${1:-experiments/overnight_research_20260906}"
OUT="$ROOT/media_methods"
mkdir -p "$OUT" "$ROOT/logs/media_methods"

render() {
  local label="$1" policy="$2" controller="$3" scenario="$4" extension="$5"
  local out="$OUT/${label}_${scenario}.${extension}"
  [[ -f "$out" ]] && return 0
  for attempt in $(seq 1 8); do
    local log="$ROOT/logs/media_methods/${label}_${scenario}_${extension}_attempt_${attempt}.log"
    local args=(--scenario "$scenario" --seed 750001 --robots 5 --clots 3 --horizon 300
      --robot-radius 0.0011 --reward-mode milestone --views triple --wall-opacity 0.10
      --spin 0.15 --max-frames 180 --stride 2 --fps 12 --width 1280 --height 720 --out "$out")
    [[ -n "$policy" ]] && args+=(--policy "$policy" --architecture gat)
    [[ -z "$policy" ]] && args+=(--controller "$controller")
    env -u DISPLAY PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$PY" make_gif.py "${args[@]}" > "$log" 2>&1
    status=$?
    if (( status == 0 )); then return 0; fi
    if [[ "$status" != 139 && "$status" != 134 && "$status" != 132 && "$status" != 135 ]]; then return "$status"; fi
  done
  return 1
}

methods=(
  "flow_guided_gat|experiments/success_study_20260905/selection/flow_guided_43.pt|"
  "old_mve|experiments/world_model_study_20260905_134953/comparisons/seed_43/mve/best_policy.pt|"
  "old_pure|experiments/world_model_study_20260905_134953/comparisons/seed_43/pure/best_policy.pt|"
  "old_gat|experiments/world_model_study_20260905_134953/baselines/gat_r5_c3/seed_43/best_policy.pt|"
  "old_edge_bias|experiments/world_model_study_20260905_134953/baselines/edge_bias_gat_r5_c3/seed_43/best_policy.pt|"
  "flow_spread||flow_spread"
)
for item in "${methods[@]}"; do
  IFS='|' read -r label policy controller <<< "$item"
  for scenario in mca_m1_lvo femoropopliteal_pad; do
    render "$label" "$policy" "$controller" "$scenario" mp4 || true
    render "$label" "$policy" "$controller" "$scenario" gif || true
  done
done

"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = [
    {"path": str(path.resolve()), "bytes": path.stat().st_size}
    for path in sorted(root.iterdir()) if path.suffix in (".mp4", ".gif")
]
(root / "manifest.json").write_text(json.dumps({"count": len(files), "files": files}, indent=2))
print(json.dumps({"count": len(files)}))
PY
