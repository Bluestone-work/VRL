#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PY=/home/wj/miniconda3/envs/v/bin/python
study="${1:-experiments/success_study_20260905}"
mkdir -p "$study/selection" "$study/logs"

evaluate() {
    local label="$1"
    shift
    local previous_attempt=0
    for previous_log in "$study/logs/eval_${label}_attempt_"*.log; do
        [[ -f "$previous_log" ]] || continue
        local index="${previous_log##*_}"
        index="${index%.log}"
        if (( index > previous_attempt )); then previous_attempt="$index"; fi
    done
    for retry in $(seq 1 12); do
        local attempt=$((previous_attempt + retry))
        if env -u DISPLAY PYTHONPATH=. PYTHONFAULTHANDLER=1 \
            OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            "$PY" scripts/eval_success_study.py \
            --episodes 20 --seed 700000 --output "$study/heldout/$label" "$@" \
            > "$study/logs/eval_${label}_attempt_${attempt}.log" 2>&1; then
            return 0
        else
            code=$?
            printf '%s attempt=%s exit=%s\n' "$label" "$attempt" "$code" >> "$study/logs/evaluation_retries.log"
            if [[ "$code" != 139 && "$code" != 134 && "$code" != 132 && "$code" != 135 ]]; then
                return "$code"
            fi
        fi
    done
    return 1
}

select_and_evaluate() {
    local mode="$1"
    local seed="$2"
    local run_dir="$study/training/$mode/seed_$seed"
    while [[ ! -f "$run_dir/summary.json" ]]; do
        sleep 15
    done
    "$PY" - "$run_dir" "$study/selection/${mode}_${seed}" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
prefix = Path(sys.argv[2])
records = [json.loads(line) for line in (run_dir / 'eval_metrics.jsonl').read_text().splitlines()]
selected = max(records, key=lambda row: (row['macro']['success'], row['macro']['removal_rate']))
source = run_dir / 'best_policy.pt'
checkpoint = prefix.with_suffix('.pt')
shutil.copy2(source, checkpoint)
prefix.with_suffix('.json').write_text(json.dumps({
    'source': str(source), 'checkpoint': str(checkpoint),
    'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'selection_rule': 'maximum validation success, then removal; never held-out scores',
    'validation_seed': 400000, 'validation_episodes_per_territory': 3,
    'selected_evaluation': selected,
}, indent=2))
PY
    evaluate "${mode}_${seed}" --policy "$study/selection/${mode}_${seed}.pt"
}

evaluate flow_controller --controller flow_guided &
controller_pid=$!
select_and_evaluate local 42 &
local_pid=$!
select_and_evaluate guided 42 &
guided_pid=$!
wait "$controller_pid"
wait "$local_pid"
wait "$guided_pid"

select_and_evaluate flow_guided 42 &
pid42=$!
select_and_evaluate flow_guided 43 &
pid43=$!
select_and_evaluate flow_guided 44 &
pid44=$!
wait "$pid42"
wait "$pid43"
wait "$pid44"
printf 'all held-out evaluations complete\n'
