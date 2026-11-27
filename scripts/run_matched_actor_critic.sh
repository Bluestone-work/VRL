#!/usr/bin/env bash
# Matched three-cluster local navigation experiment, one training seed per invocation.
set -euo pipefail
cd "$(dirname "$0")/.."
seed=${1:-41}
steps=${2:-50176}
out=${3:-runs/matched_actor_critic}
mode=${4:-full}
case "$mode" in
  full) ;;
  smoke) out="$out/smoke"; [[ "$steps" == 256 ]] || { echo "Use 256 steps in smoke mode" >&2; exit 1; } ;;
  *) echo "Use full or smoke" >&2; exit 1 ;;
esac
[[ "$seed" =~ ^[0-9]+$ && "$steps" =~ ^[1-9][0-9]*$ ]] && (( steps % 256 == 0 )) || {
  echo "Use an integer seed and a positive step count divisible by the PPO rollout length (256)." >&2
  exit 1
}
scenes=research/COMPARISON_20261125_REPAIRED
if [[ "$mode" == smoke ]]; then
  smoke_scenes="$out/steps${steps}/s${seed}/manifests"
  python - "$scenes" "$smoke_scenes" <<'PYTHON'
import json, sys
from pathlib import Path
src, dst = map(Path, sys.argv[1:])
dst.mkdir(parents=True, exist_ok=True)
for label in ("train180", "holdout100"):
    name = f"paired_{label}.json"
    data = json.loads((src / name).read_text())
    data["rows"] = data["rows"][:1]
    (dst / name).write_text(json.dumps(data, indent=2) + "\n")
PYTHON
  scenes="$smoke_scenes"
fi
for algo in ppo a2c mappo; do
  dir="$out/steps${steps}/s${seed}/$algo"
  if [[ ! -f "$dir/TRAINING_COMPLETE.json" ]]; then
    if [[ -e "$dir" ]]; then
      echo "Incomplete run in $dir. Preserve it and choose a fresh output root for a retry." >&2
      exit 1
    fi
    python scripts/train_sb3_ippo.py --algo "$algo" --seed "$seed" --clusters 3 \
      --timesteps "$steps" --action-set semantic_reduced --mask-invalid --flow-feature \
      --safety-parity --eval-count 0 --device "${TRAIN_DEVICE:-auto}" --out "$dir"
  fi
  python - "$dir" "$seed" "$steps" "$algo" <<'PY'
import json, sys
from pathlib import Path
d, seed, steps, algo = sys.argv[1:]
d, steps = Path(d), int(steps)
cfg = json.loads((d / "config.json").read_text())
done = json.loads((d / "TRAINING_COMPLETE.json").read_text())
settings = dict(seed=int(seed), physical_steps=steps, clusters=3, optimiser=algo,
                flow_feature=True, safety_parity=True, invalid_action_mask=True,
                action_set="semantic_reduced", critic_identity=algo == "mappo",
                warm_start_from=None, centerline_match=False, exit_mask=False, curriculum=True)
for k,v in settings.items():
    if k not in cfg or cfg[k] != v:
        raise SystemExit(f"{d}: {k}={cfg.get(k)!r}, expected {v!r}")
if done["agent_steps"] != steps*3 or done["environment_steps"] != steps or not (d / "final_model.zip").is_file():
    raise SystemExit(f"{d}: checkpoint or realized interaction budget mismatch: {done}")
PY
  kind=ppo; extra=()
  if [[ "$algo" == a2c ]]; then kind=a2c; fi
  if [[ "$algo" == mappo ]]; then extra+=(--centralized-critic); fi
  for split in train holdout_test; do
    if [[ "$split" == train ]]; then label=train180; n=180; else label=holdout100; n=100; fi
    if [[ "$mode" == smoke ]]; then n=1; fi
    python scripts/evaluate_navigation_chunks.py --model "$dir/final_model.zip" \
      --kind "$kind" --action-set semantic_reduced --flow-feature "${extra[@]}" \
      --split "$split" --clusters 3 --chunk-size "${EVAL_CHUNK_SIZE:-1}" --total "$n" --scene-list "$scenes/paired_${label}.json" \
      --out "$dir/paired_${label}.json"
  done
done
python scripts/report_matched_actor_critic.py "$out/steps${steps}/s${seed}" --manifests "$scenes"
