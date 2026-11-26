#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out_root=${1:-research/tpg_conflict}
mkdir -p "$out_root"
for allocation in geodesic linear_sum; do
  for gate in --tpg --no-tpg; do
    python scripts/evaluate_fixed_three_lesions.py --clusters 1,2,3 --allocation "$allocation" "$gate" \
      --out "$out_root/${allocation}_${gate#--}.json"
  done
done
