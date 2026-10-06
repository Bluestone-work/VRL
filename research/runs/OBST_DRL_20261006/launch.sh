#!/bin/bash
# v3.4 experiment: 3 rounds x 5 runs, 100 min each, 4 rollout workers per run. Touch $R/STOP to stop
# before the next round (no kill race). Variants:
#   tres = T-IRPPO (ours) | mlp, gru = backbone ablations | tvel = Transformer + obstacle velocity input
#   tdir = Transformer without rule prior (Turbo-style direct) | tnodr = ours without domain randomisation
cd /home/wj/桌面/vascular_marl_local.tar.
R=research/runs/OBST_DRL_20261006
run() { name=$1; cores=$2; dev=$3; shift 3
  PYTHONPATH=. taskset -c $cores ~/miniconda3/envs/v/bin/python scripts/train_obstacle_drl.py --out $R/$name --minutes 100 --workers 4 --device cuda:$dev "$@" > $R/$name.log 2>&1; }
round() { [ -f $R/STOP ] && exit 0
  run $1 0-3 0 ${A1[@]} & run $2 4-5,8-9 1 ${A2[@]} & run $3 10-13 0 ${A3[@]} & run $4 14-17 1 ${A4[@]} & run $5 18-21 0 ${A5[@]} & wait; }
A1=(--seed 0); A2=(--seed 1); A3=(--seed 2); A4=(--arch mlp --seed 0); A5=(--obs-vel --seed 0)
round tres_s0 tres_s1 tres_s2 mlp_s0 tvel_s0
A1=(--arch gru --seed 0); A2=(--arch gru --seed 1); A3=(--arch mlp --seed 1); A4=(--direct --seed 0); A5=(--no-dr --seed 0)
round gru_s0 gru_s1 mlp_s1 tdir_s0 tnodr_s0
A1=(--arch gru --seed 2); A2=(--arch mlp --seed 2); A3=(--obs-vel --seed 1); A4=(--direct --seed 1); A5=(--no-dr --seed 1)
round gru_s2 mlp_s2 tvel_s1 tdir_s1 tnodr_s1
echo done > $R/ALL_TRAINED
