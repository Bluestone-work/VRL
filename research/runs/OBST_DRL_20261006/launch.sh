#!/bin/bash
# round-based launcher: each job = name|cores|args ; 4 workers per run, 120 min
cd /home/wj/桌面/vascular_marl_local.tar.
R=research/runs/OBST_DRL_20261006
run() { name=$1; cores=$2; dev=$3; shift 3
  PYTHONPATH=. taskset -c $cores ~/miniconda3/envs/v/bin/python scripts/train_obstacle_drl.py --out $R/$name --minutes 110 --workers 4 --device cuda:$dev "$@" > $R/$name.log 2>&1; }
round() {
  run $1 0-3 0 ${A1[@]} & run $2 4-5,8-9 1 ${A2[@]} & run $3 10-13 0 ${A3[@]} & run $4 14-17 1 ${A4[@]} & run $5 18-21 0 ${A5[@]} & wait; }
A1=(--arch transformer --seed 0); A2=(--arch transformer --seed 1); A3=(--arch transformer --seed 2); A4=(--arch gru --seed 0); A5=(--arch mlp --seed 0)
round tres_s0 tres_s1 tres_s2 gru_s0 mlp_s0
A1=(--arch gru --seed 1); A2=(--arch mlp --seed 1); A3=(--arch transformer --direct --seed 0); A4=(--arch transformer --direct --seed 1); A5=(--arch transformer --no-dr --seed 0)
round gru_s1 mlp_s1 tdir_s0 tdir_s1 tnodr_s0
A1=(--arch transformer --no-dr --seed 1); A2=(--arch gru --seed 2); A3=(--arch mlp --seed 2); A4=(--arch transformer --direct --seed 2); A5=(--arch transformer --no-dr --seed 2)
round tnodr_s1 gru_s2 mlp_s2 tdir_s2 tnodr_s2
echo done > $R/ALL_TRAINED
