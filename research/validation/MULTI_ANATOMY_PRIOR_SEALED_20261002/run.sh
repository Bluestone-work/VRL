#!/bin/bash
# Zero-shot sealed test of the route+avoid prior (gain 6, chosen on MCA diagnostic) on the 13 non-MCA anatomies.
cd "/home/wj/桌面/vascular_marl_local.tar."
first=1
for a in $(grep -v '^mca_m1_lvo$' research/validation/MULTI_ANATOMY_PRIOR_PROBE_20261002/anatomies.txt); do
  extra=""; [ $first = 1 ] && extra="--declare-candidates 13"; first=0
  PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c 20-23 $HOME/miniconda3/envs/v/bin/python scripts/evaluate_mca_sealed_test.py \
    --protocol configs/experiments/EXP_0042_ROUTE_AVOID_RESIDUAL.json \
    --checkpoint research/initializations/ROUTE_AVOID_PRIOR_ONLY_ANATOMY_20261002.pt \
    --study PRIOR_ZERO_SHOT_ANATOMIES --label route_avoid_prior_gain6_$a --anatomy $a $extra --workers 4 || echo "FAILED $a"
done
echo DONE
