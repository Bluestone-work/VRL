#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="/home/wj/miniconda3/envs/v/bin/python"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

declare -a METHODS=(
  "EXP0120_IPPO_SEMANTIC_1M_20261016|EXP0124_IPPO_SEMANTIC_3M_20261016|ippo|semantic"
  "EXP0121_IPPO_ATOMIC_1M_20261016|EXP0125_IPPO_ATOMIC_3M_20261016|ippo|atomic"
  "EXP0122_IPPO_CONTINUOUS_1M_20261016|EXP0126_IPPO_CONTINUOUS_3M_20261016|ippo|continuous"
  "EXP0123_ISAC_CONTINUOUS_1M_20261016|EXP0127_ISAC_CONTINUOUS_3M_20261016|isac|continuous"
)

echo "[$(date -Is)] waiting for all four 1M runs to finish"
while :; do
  ready=1
  for spec in "${METHODS[@]}"; do
    IFS='|' read -r source target kind action <<< "$spec"
    if [[ ! -f "$ROOT/research/runs/$source/RUN_COMPLETE.json" ]]; then
      ready=0
    fi
  done
  if [[ "$ready" == 1 ]]; then
    break
  fi
  sleep 60
done

echo "[$(date -Is)] all 1M runs complete; launching independent 3M runs"
for spec in "${METHODS[@]}"; do
  IFS='|' read -r source target kind action <<< "$spec"
  out="$ROOT/research/runs/$target"
  if [[ -e "$out" ]]; then
    echo "[$(date -Is)] skip existing $out"
    continue
  fi
  if [[ "$kind" == ippo ]]; then
    nohup env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -m scripts.train_sb3_ippo \
      --out "$out" --seed 41 --timesteps 3000000 --clusters 3 --device cuda \
      --eval-count 36 --action-set "$action" \
      > "${out}_live.log" 2>&1 &
  else
    nohup env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u scripts/train_sb3_isac.py \
      --out "$out" --seed 41 --timesteps 3000000 --clusters 3 --device cuda \
      --eval-count 36 > "${out}_live.log" 2>&1 &
  fi
  echo "[$(date -Is)] started $target"
done
echo "[$(date -Is)] 3M launch commands submitted"
