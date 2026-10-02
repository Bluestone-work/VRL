#!/bin/bash
cd "/home/wj/桌面/vascular_marl_local.tar."
extra="--declare-candidates 3"
for s in 42 43 44; do
  PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c 0-5,8-23 ~/miniconda3/envs/v/bin/python scripts/evaluate_mca_sealed_test.py --protocol configs/experiments/EXP_0045_SHIELD_EVAL.json \
    --checkpoint research/initializations/EXP0045_FROM_EXP0043_SELECTED_20261002/seed_$s.pt --study EXP_0045_SHIELD_EVAL --label exp43_selected_shield_seed_$s $extra --workers 20 || echo FAILED $s
  extra=""
done; echo DONE
