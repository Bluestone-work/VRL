#!/bin/bash
# Benchmark v2: union-of-tubes junction model. Deployable (ours), privileged references, Codex HRL (measured).
cd "/home/wj/桌面/vascular_marl_local.tar."
E=research/validation/BENCHMARK_V2_UNION_20261005; PY=$HOME/miniconda3/envs/v/bin/python
cat $E/jobs.txt | xargs -P 22 -L 1 bash -c '
  kind=$0; m=$1; n=$2; a=$3; s0=$4; s1=$5; tag=$6; ck=$7
  case $kind in
    dep)   PYTHONPATH=. OMP_NUM_THREADS=1 '$PY' scripts/benchmark_deployable.py --method $m --junction union --clusters $n --anatomy $a --seeds $s0:$s1 --tag $tag --out '$E'/${tag}_N${n}_$a.jsonl ;;
    priv)  PYTHONPATH=. OMP_NUM_THREADS=1 '$PY' scripts/benchmark_multicluster.py --method $m --junction union --clusters $n --anatomy $a --seeds $s0:$s1 --tag $tag --out '$E'/${tag}_N${n}_$a.jsonl ;;
    codex) c=""; [ "$ck" != none ] && c="--checkpoint $ck"; PYTHONPATH=. OMP_NUM_THREADS=1 '$PY' scripts/codex_hrl_benchmark.py --variant $m $c --junction union --clusters $n --anatomy $a --seeds $s0:$s1 --tag $tag --out '$E'/${tag}_N${n}_$a.jsonl ;;
  esac 2>>'$E'/errors.log'
echo DONE >> $E/status.txt
