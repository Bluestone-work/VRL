#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export PY=/home/wj/miniconda3/envs/v/bin/python
ROOT="${1:-experiments/full_matrix_20260906}"
EPISODES="${EPISODES:-20}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"
WORKER="${WORKER:-all}"
mkdir -p "$ROOT" "$ROOT/logs" "$ROOT/methods" "$ROOT/media" "$ROOT/training"

cat > "$ROOT/methods.tsv" <<'EOF'
flow_guided_gat|experiments/success_study_20260905/selection/flow_guided_43.pt|policy|flow_guided_43|flow_guided_gat
guided_42|experiments/success_study_20260905/selection/guided_42.pt|policy|guided_42|guided_42
local_42|experiments/success_study_20260905/selection/local_42.pt|policy|local_42|local_42
curriculum_gat_43|experiments/world_model_study_20260905_134953/curriculum/gat_r5_c3/seed_43/best_policy.pt|policy|curriculum_gat_43|curriculum_gat_43
baseline_gat_43|experiments/world_model_study_20260905_134953/baselines/gat_r5_c3/seed_43/best_policy.pt|policy|baseline_gat_43|baseline_gat_43
edge_bias_gat_43|experiments/world_model_study_20260905_134953/baselines/edge_bias_gat_r5_c3/seed_43/best_policy.pt|policy|edge_bias_gat_43|edge_bias_gat_43
mlp_43|experiments/world_model_study_20260905_134953/baselines/mlp_r5_c3/seed_43/best_policy.pt|policy|mlp_43|mlp_43
pure_43|experiments/world_model_study_20260905_134953/comparisons/seed_43/pure/best_policy.pt|policy|pure_43|pure_43
mve_43|experiments/world_model_study_20260905_134953/comparisons/seed_43/mve/best_policy.pt|policy|mve_43|mve_43
flow_controller||controller|flow_controller|flow_guided
flow_spread||controller|flow_spread|flow_spread
EOF

cat > "$ROOT/scenarios.txt" <<'EOF'
straight
bifurcation
anastomosis
stenotic
multilevel
mca_stroke
pulmonary_saddle
coronary_lm_bifurcation
coronary_rca
iliac_may_thurner
popliteal_calf_dvt
mca_m1_lvo
ica_siphon
ica_terminus_t
carotid_bifurcation
basilar_vertebral
cerebral_venous_sinus
sma_embolism
femoropopliteal_pad
renal_artery
EOF

case "$WORKER" in
  first) awk 'NR % 2 == 1' "$ROOT/methods.tsv" > "$ROOT/methods.first.tsv"; worker_file="$ROOT/methods.first.tsv" ;;
  second) awk 'NR % 2 == 0' "$ROOT/methods.tsv" > "$ROOT/methods.second.tsv"; worker_file="$ROOT/methods.second.tsv" ;;
  all) cp "$ROOT/methods.tsv" "$ROOT/methods.all.tsv"; worker_file="$ROOT/methods.all.tsv" ;;
  *) echo "WORKER must be first, second, or all" >&2; exit 2 ;;
esac

is_crash() {
  case "$1" in 132|134|135|136|139|143) return 0 ;; *) return 1 ;; esac
}

radius_for() {
  case "$1" in
    straight|bifurcation|anastomosis|stenotic|multilevel|mca_stroke) echo 0.0045 ;;
    *) echo 0.0011 ;;
  esac
}

sha256_of() {
  "$PY" - "$1" <<'PY'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
print(h.hexdigest())
PY
}

write_training_record() {
  local label="$1" checkpoint="$2" kind="$3" source="$4" out="$ROOT/training/$label"
  mkdir -p "$out"
  "$PY" - "$out/record.json" "$label" "$checkpoint" "$kind" "$source" <<'PY'
import json, pathlib, sys
path, label, checkpoint, kind, source = sys.argv[1:]
data = {
    "method": label,
    "training_status": "completed_reused_checkpoint" if checkpoint else "controller_no_training",
    "checkpoint": checkpoint or None,
    "checkpoint_sha256": None,
    "source": source,
    "kind": kind,
    "note": "Existing validated training artifact reused; no new training claim is made by this matrix run.",
}
if checkpoint:
    import hashlib
    h = hashlib.sha256()
    with open(checkpoint, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    data["checkpoint_sha256"] = h.hexdigest()
pathlib.Path(path).write_text(json.dumps(data, indent=2))
PY
}

run_episode() {
  local label="$1" checkpoint="$2" kind="$3" controller="$4" scenario="$5" episode="$6"
  local index="$7" seed=$((700000 + index * 10000 + episode))
  local method_dir="$ROOT/methods/$label"
  local output="$method_dir/episodes/${scenario}_${seed}.jsonl"
  local radius
  radius="$(radius_for "$scenario")"
  mkdir -p "$method_dir/episodes" "$method_dir/logs"
  [[ -s "$output" ]] && return 0
  local attempt status=1
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    local log="$method_dir/logs/${scenario}_${seed}_attempt_${attempt}.log"
    local args=(scripts/eval_single_episode.py --scenario "$scenario" --seed "$seed"
      --robots 5 --clots 3 --horizon 300 --robot-radius "$radius" --output "$output")
    if [[ "$kind" == policy ]]; then
      args+=(--policy "$checkpoint")
    else
      args+=(--controller "$controller")
    fi
    env -u DISPLAY PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$PY" "${args[@]}" > "$log" 2>&1
    status=$?
    if (( status == 0 )); then break; fi
    if ! is_crash "$status"; then break; fi
  done
  if (( status != 0 )); then
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$scenario" "$seed" "$status" "$attempt" >> "$ROOT/logs/failed_episodes.tsv"
    return 1
  fi
  return 0
}

aggregate_method() {
  local label="$1" method_dir="$ROOT/methods/$label"
  "$PY" - "$method_dir" "$label" "$EPISODES" <<'PY'
import json, pathlib, sys
root, label, episodes = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rows = []
for path in sorted((root / "episodes").glob("*.jsonl")):
    for line in path.read_text().splitlines():
        if line.strip(): rows.append(json.loads(line))
by = {}
for row in rows: by.setdefault(row["scenario"], []).append(row)
keys = ("success", "removal_rate", "return", "steps", "wall_hits_total", "wall_hits_per_step", "robot_collisions_total", "contact_miss")
per = {scenario: {key: sum(row[key] for row in values) / len(values) for key in keys}
       for scenario, values in sorted(by.items())}
macro = {key: sum(row[key] for row in rows) / max(len(rows), 1) for key in keys}
summary = {
    "method": label, "episodes_requested_per_scenario": episodes,
    "episodes_completed": len(rows), "scenarios_completed": len(by),
    "scenarios_expected": 20, "successes": sum(row["success"] for row in rows),
    "macro_completed_episodes": macro, "per_scenario": per,
    "complete": len(rows) == 20 * episodes and len(by) == 20,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({"method": label, "episodes": len(rows), "complete": summary["complete"]}))
PY
}

index=0
  while IFS='|' read -r label checkpoint kind source controller; do
  [[ -z "$label" ]] && continue
  write_training_record "$label" "$checkpoint" "$kind" "$source"
  scenario_index=0
  while IFS= read -r scenario; do
    [[ -z "$scenario" ]] && continue
    for episode in $(seq 0 $((EPISODES - 1))); do
      run_episode "$label" "$checkpoint" "$kind" "$controller" "$scenario" "$episode" "$scenario_index" || true
    done
    scenario_index=$((scenario_index + 1))
  done < "$ROOT/scenarios.txt"
  aggregate_method "$label"
done < "$worker_file"

"$PY" - "$ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
items = []
for record in sorted((root / "training").glob("*/record.json")):
    items.append(json.loads(record.read_text()))
summary = {
    "episodes_per_scenario": int(__import__("os").environ.get("EPISODES", "20")),
    "methods_registered": len(items), "methods": items,
    "failed_episode_log": str((root / "logs" / "failed_episodes.tsv").resolve()),
}
(root / "training_manifest.json").write_text(json.dumps(summary, indent=2))
PY
echo "matrix worker=$WORKER finished root=$ROOT"
