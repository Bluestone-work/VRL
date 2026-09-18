#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export PY=/home/wj/miniconda3/envs/v/bin/python
ROOT="${1:-experiments/overnight_research_20260905}"
mkdir -p "$ROOT/media" "$ROOT/logs"

methods=(flow_guided_gat edge_bias_flow curriculum_flow world_gat world_mve)
scenarios=(mca_m1_lvo femoropopliteal_pad)

for method in "${methods[@]}"; do
  policy="$ROOT/training/$method/seed_43/best_policy.pt"
  [[ "$method" == "curriculum_flow" ]] && policy="$ROOT/training/$method/seed_44/best_policy.pt"
  [[ -f "$policy" ]] || { echo "missing $policy" >> "$ROOT/logs/media_skips.log"; continue; }
  for scenario in "${scenarios[@]}"; do
    for extension in mp4 gif; do
      out="$ROOT/media/${method}_${scenario}.${extension}"
      [[ -f "$out" ]] && continue
      for attempt in $(seq 1 6); do
        log="$ROOT/logs/media_${method}_${scenario}_${extension}_attempt_${attempt}.log"
        env -u DISPLAY PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
          "$PY" make_gif.py --policy "$policy" --architecture gat --scenario "$scenario" \
          --seed 750001 --robots 5 --clots 3 --horizon 300 --robot-radius 0.0011 \
          --reward-mode milestone --views triple --wall-opacity 0.10 --spin 0.15 \
          --max-frames 180 --stride 2 --fps 12 --width 1280 --height 720 \
          --out "$out" > "$log" 2>&1
        status=$?
        if (( status == 0 )); then break; fi
        if [[ "$status" != 139 && "$status" != 134 && "$status" != 132 && "$status" != 135 ]]; then break; fi
      done
    done
  done
done

"$PY" - "$ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "media").glob("*")):
    if path.suffix in (".mp4", ".gif"):
        rows.append({"path": str(path.resolve()), "bytes": path.stat().st_size})
(root / "media_manifest.json").write_text(json.dumps({"files": rows, "count": len(rows)}, indent=2))
print(json.dumps({"count": len(rows)}))
PY
