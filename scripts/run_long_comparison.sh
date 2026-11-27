#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
timesteps=${TIMESTEPS:-500000}
seeds=${SEEDS:-41,42,43}
IFS=, read -r -a seed_list <<< "$seeds"
for seed in "${seed_list[@]}"; do
  # Keep historical 50k/100k runs immutable.  The previous entry point
  # reused those directories, then failed its own target-step validation when
  # the default was 500k.  A target-specific directory makes each run
  # resumable without confusing an older result for the requested budget.
  out="research/paired_navigation_corrected/mappo_v2_long_${timesteps}_s${seed}"
  echo "Starting or resuming MAPPO-V2 seed=${seed} timesteps=${timesteps} output=${out}"
  PATH=/home/wj/miniconda3/envs/navrl/bin:$PATH bash scripts/rerun_paired_navigation.sh "$seed" "$out" "$timesteps"
done
