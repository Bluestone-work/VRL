#!/bin/bash
# Checkpoint choice is fixed in advance: seed 42 (selected at 1M on MCA validation). No anatomy-specific tuning.
cd "/home/wj/桌面/vascular_marl_local.tar."
first=1
for a in $(grep -v '^mca_m1_lvo$' research/validation/MULTI_ANATOMY_PRIOR_PROBE_20261002/anatomies.txt); do
  extra=""; [ $first = 1 ] && extra="--declare-candidates 13"; first=0
  PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c 20-23 $HOME/miniconda3/envs/v/bin/python scripts/evaluate_mca_sealed_test.py \
    --protocol configs/experiments/EXP_0045_SHIELD_EVAL.json \
    --checkpoint research/initializations/EXP0045_FROM_EXP0043_SELECTED_20261002/seed_42.pt \
    --study EXP43_SHIELD_ZERO_SHOT_ANATOMIES --label exp43_seed42_shield_$a --anatomy $a $extra --workers 4 || echo "FAILED $a"
done
echo DONE
