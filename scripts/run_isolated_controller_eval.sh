#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export PY=/home/wj/miniconda3/envs/v/bin/python
ROOT="${1:-experiments/overnight_research_20260906/controller_pilots}"
CONTROLLER="${2:-flow_spread}"
mkdir -p "$ROOT/$CONTROLLER/episodes" "$ROOT/logs"
scenarios=(pulmonary_saddle coronary_lm_bifurcation coronary_rca mca_m1_lvo ica_siphon ica_terminus_t carotid_bifurcation basilar_vertebral sma_embolism femoropopliteal_pad renal_artery iliac_may_thurner popliteal_calf_dvt cerebral_venous_sinus)

for index in "${!scenarios[@]}"; do
  scenario="${scenarios[$index]}"
  for episode in $(seq 0 19); do
    seed=$((400000 + index * 10000 + episode))
    output="$ROOT/$CONTROLLER/episodes/${scenario}_${seed}.jsonl"
    [[ -f "$output" ]] && continue
    success=0
    for attempt in $(seq 1 15); do
      env -u DISPLAY PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$PY" scripts/eval_single_episode.py --scenario "$scenario" --seed "$seed" \
        --controller "$CONTROLLER" --output "$output" \
        > "$ROOT/logs/${CONTROLLER}_${scenario}_${seed}_attempt_${attempt}.log" 2>&1
      status=$?
      if (( status == 0 )); then success=1; break; fi
      if [[ "$status" != 139 && "$status" != 134 && "$status" != 132 && "$status" != 135 ]]; then break; fi
    done
    (( success == 1 )) || echo "$CONTROLLER $scenario $seed failed after retries" >> "$ROOT/logs/isolated_failures.log"
  done
done

"$PY" - "$ROOT/$CONTROLLER" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "episodes").glob("*.jsonl")):
    rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
by = {}
for row in rows:
    by.setdefault(row["scenario"], []).append(row)
keys = ("success", "removal_rate", "return", "wall_hits_total", "wall_hits_per_step", "robot_collisions_total", "steps", "contact_miss")
summary = {
    "episodes": len(rows),
    "successes": sum(row["success"] for row in rows),
    "macro": {key: sum(row[key] for row in rows) / max(len(rows), 1) for key in keys},
    "per_territory": {
        scenario: {key: sum(row[key] for row in values) / len(values) for key in keys}
        for scenario, values in by.items()
    },
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY
