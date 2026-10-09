set -x
PY="taskset -c 0-5,8-23 /home/wj/miniconda3/envs/v/bin/python -m scripts.benchmark_lysis --count 10 --workers 20"
$PY --method ours_classical --kw '{"fallback":"help"}' --tag ours_classical_help --out research/validation/V4_LYSIS_20261008/ours_classical_help.jsonl
$PY --method ours_classical --kw '{"tpg":false}' --tag ours_classical_notpg --clusters 2,3 --out research/validation/V4_LYSIS_20261008/ours_classical_notpg.jsonl
