#!/bin/bash
# usage: run_selection.sh ARM PROTO
R="/home/wj/桌面/vascular_marl_local.tar."; ARM=$1; PROTO=$2
S="$R/research/runs/EXP_0037_LOW_LR_20261001a/$ARM/source_snapshot"; O="$R/research/validation/EXP0037_SELECTION_20261001"
until [ -f "$R/research/runs/EXP_0037_LOW_LR_20261001a/$ARM/seed_42/evaluation_500000.json" ] && [ -f "$R/research/runs/EXP_0037_LOW_LR_20261001a/$ARM/seed_43/evaluation_500000.json" ] && [ -f "$R/research/runs/EXP_0037_LOW_LR_20261001a/$ARM/seed_44/evaluation_500000.json" ]; do sleep 30; done
for s in 42 43 44; do for m in 100000 200000 300000 400000 500000; do
 echo "cd '$S' && PYTHONPATH=. OMP_NUM_THREADS=1 /home/wj/miniconda3/envs/v/bin/python '$R/scripts/select_mca_checkpoint.py' --protocol configs/experiments/$PROTO.json --checkpoint ../seed_$s/policy_$m.pt --out '$O/${ARM}_seed_${s}_$m.json'"
done; done | taskset -c 20-23 xargs -P 4 -I{} bash -c "{}"
echo "$ARM selection done $(date)"
