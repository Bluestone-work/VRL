#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
upstream=${MR_NAV_ROOT:-external/mr-nav}
dataset=${MR_NAV_DATASET:-$upstream/dataset/vascular_env_2d_sim/unique_simulated_vessel_2d}
exec python scripts/evaluate_mr_nav_2d.py --upstream "$upstream" --dataset "$dataset" \
    --out "${1:-research/mr_nav_2d_paired.json}"
