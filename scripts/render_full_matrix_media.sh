#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export PY=/home/wj/miniconda3/envs/v/bin/python
ROOT="${1:-experiments/full_matrix_20260906}"
WORKER="${WORKER:-all}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
mkdir -p "$ROOT/media" "$ROOT/logs/media"

if [[ ! -f "$ROOT/methods.tsv" || ! -f "$ROOT/scenarios.txt" ]]; then
  echo "missing matrix registry; run run_full_matrix_study.sh first" >&2
  exit 2
fi
case "$WORKER" in
  first) awk 'NR % 2 == 1' "$ROOT/methods.tsv" > "$ROOT/media.first.tsv"; method_file="$ROOT/media.first.tsv" ;;
  second) awk 'NR % 2 == 0' "$ROOT/methods.tsv" > "$ROOT/media.second.tsv"; method_file="$ROOT/media.second.tsv" ;;
  all) cp "$ROOT/methods.tsv" "$ROOT/media.all.tsv"; method_file="$ROOT/media.all.tsv" ;;
  *) echo "WORKER must be first, second, or all" >&2; exit 2 ;;
esac

radius_for() {
  case "$1" in
    straight|bifurcation|anastomosis|stenotic|multilevel|mca_stroke) echo 0.0045 ;;
    *) echo 0.0011 ;;
  esac
}

is_crash() {
  case "$1" in 132|134|135|136|139|143) return 0 ;; *) return 1 ;; esac
}

render_one() {
  local label="$1" checkpoint="$2" kind="$3" source="$4" controller="$5" scenario="$6" extension="$7"
  local out="$ROOT/media/$label/$scenario/$label.$extension"
  local logdir="$ROOT/media/$label/$scenario/logs"
  local radius
  radius="$(radius_for "$scenario")"
  mkdir -p "$logdir"
  [[ -s "$out" ]] && return 0
  local attempt status=1
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    local log="$logdir/${extension}_attempt_${attempt}.log"
    local args=(--scenario "$scenario" --seed 750001 --robots 5 --clots 3 --horizon 300
      --robot-radius "$radius" --reward-mode milestone --views triple --wall-opacity 0.10
      --spin 0.15 --max-frames 180 --stride 2 --fps 12 --width 1280 --height 720 --out "$out")
    if [[ "$kind" == policy ]]; then
      args+=(--policy "$checkpoint")
    else
      args+=(--controller "$controller")
    fi
    env -u DISPLAY PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$PY" make_gif.py "${args[@]}" > "$log" 2>&1
    status=$?
    if (( status == 0 )); then return 0; fi
    if ! is_crash "$status"; then break; fi
  done
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$scenario" "$extension" "$status" "$attempt" \
    >> "$ROOT/logs/media_failed.tsv"
  return 1
}

while IFS='|' read -r label checkpoint kind source controller; do
  [[ -z "$label" ]] && continue
  while IFS= read -r scenario; do
    [[ -z "$scenario" ]] && continue
    render_one "$label" "$checkpoint" "$kind" "$source" "$controller" "$scenario" gif || true
    render_one "$label" "$checkpoint" "$kind" "$source" "$controller" "$scenario" mp4 || true
  done < "$ROOT/scenarios.txt"
done < "$method_file"

"$PY" - "$ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
methods = {}
for path in sorted((root / "media").glob("*/*/*")):
    if path.suffix not in {".gif", ".mp4"}: continue
    label, scenario = path.parent.parent.name, path.parent.name
    methods.setdefault(label, {"gif": [], "mp4": []})[path.suffix[1:]].append(str(path.relative_to(root)))
manifest = {
    "methods": methods,
    "gif_count": sum(len(item["gif"]) for item in methods.values()),
    "mp4_count": sum(len(item["mp4"]) for item in methods.values()),
    "expected_methods": 11, "expected_scenarios": 20,
}
(root / "media_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps({"gif_count": manifest["gif_count"], "mp4_count": manifest["mp4_count"]}))
PY
echo "media worker=$WORKER finished root=$ROOT"
