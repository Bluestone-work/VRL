#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
seed=${1:?usage: $0 SEED}
out=${2:-research/paired_navigation_corrected/mappo_v2_s${seed}}
bash scripts/rerun_paired_navigation.sh "$seed" "$out"
