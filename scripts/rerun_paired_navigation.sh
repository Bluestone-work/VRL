#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
seed=${1:-41}
out=${2:-research/paired_navigation_corrected/mappo_v2_s${seed}}
timesteps=${3:-50000}
scenes=research/COMPARISON_20261125_REPAIRED
ippo="research/COMPARISON_20261125/models/ippo_s${seed}_final_model.zip"
[[ -f "$ippo" ]] || { echo "Missing comparison checkpoint: $ippo" >&2; exit 1; }
if [[ ! -e "$out/final_model.zip" ]]; then
    if [[ -e "$out/config.json" ]]; then
        echo "Found a partial training run in $out. Use a new output directory to train again." >&2
        exit 1
    fi
    python scripts/train_sb3_ippo.py --algo mappo --clusters 3 --seed "$seed" \
        --timesteps "$timesteps" --action-set semantic_reduced --mask-invalid --flow-feature \
        --safety-parity --out "$out"
fi
python - "$out" "$seed" "$timesteps" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text())
expected = dict(optimiser="mappo", seed=int(sys.argv[2]), physical_steps=int(sys.argv[3]), clusters=3,
                action_set="semantic_reduced", invalid_action_mask=True, flow_feature=True,
                safety_parity=True, critic_identity=True, curriculum=True, exit_mask=False)
for key, value in expected.items():
    if config.get(key) != value:
        raise SystemExit(f"{key}: expected {value!r}, found {config.get(key)!r} in {root}")
if not (root / "TRAINING_COMPLETE.json").is_file():
    raise SystemExit(f"Missing training completion record in {root}")
PY
for split in train holdout_test; do
    if [[ "$split" == train ]]; then label=train180; count=180; else label=holdout100; count=100; fi
    python scripts/evaluate_navigation_chunks.py --model "$out/final_model.zip" \
        --kind ppo --action-set semantic_reduced --flow-feature --centralized-critic \
        --split "$split" --clusters 3 --total "$count" --scene-list "$scenes/paired_${label}.json" \
        --out "$out/paired_${label}.json"
    python scripts/evaluate_navigation_chunks.py --model "$ippo" \
        --kind ppo --action-set semantic_reduced --flow-feature \
        --split "$split" --clusters 3 --total "$count" --scene-list "$scenes/paired_${label}.json" \
        --out "$out/ippo_${label}.json"
done
python scripts/summarize_repaired_comparison.py --rerun-dir "$out"
