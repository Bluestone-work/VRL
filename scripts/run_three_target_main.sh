#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out_root=${1:-research/three_target_main}
mkdir -p "$out_root"
for clusters in 1 2 3; do
  python scripts/evaluate_fixed_three_lesions.py --clusters "$clusters" --allocation geodesic --tpg \
    --out "$out_root/geodesic_n${clusters}_tpg.json"
done
