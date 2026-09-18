#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
export PY=/home/wj/miniconda3/envs/v/bin/python

study_root="${1:-experiments/world_model_study_20260905_134953}"
artifact_root="$study_root/artifacts"
mkdir -p "$artifact_root/policy_videos" "$artifact_root/policy_gifs" \
  "$artifact_root/all_scenarios"

select_policy() {
  local arm="$1"
  "$PY" -c \
    "import glob,json,pathlib; ps=glob.glob('$study_root/comparisons/seed_*/$arm/summary.json'); assert ps, 'no $arm summaries'; best=max(ps,key=lambda p:(lambda m:(m['success'],m['removal_rate']))(json.load(open(p))['final_evaluation']['macro'])); print(pathlib.Path(best).parent/'final_policy.pt')"
}

failed=0
for arm in pure mve; do
  policy="$(select_policy "$arm")" || exit 2
  for extension in mp4 gif; do
    out_dir="$artifact_root/policy_videos"
    if [[ "$extension" == "gif" ]]; then
      out_dir="$artifact_root/policy_gifs"
    fi
    env -u DISPLAY PYTHONPATH=. "$PY" make_gif.py \
      --policy "$policy" --architecture gat \
      --scenario mca_m1_lvo --robots 5 --clots 3 --horizon 300 \
      --robot-radius 0.0011 --reward-mode milestone \
      --views triple --wall-opacity 0.10 --spin 0.2 \
      --max-frames 180 --stride 1 --fps 12 --width 1280 --height 720 \
      --out "$out_dir/best_${arm}_mca_m1_lvo.${extension}" || failed=1
  done
done

while read -r scenario radius; do
  env -u DISPLAY PYTHONPATH=. "$PY" make_gif.py \
    --scenario "$scenario" --robots 5 --clots 3 --horizon 300 \
    --robot-radius "$radius" --reward-mode milestone \
    --views triple --wall-opacity 0.10 \
    --max-frames 40 --stride 2 --fps 12 --width 1280 --height 720 \
    --out "$artifact_root/all_scenarios/${scenario}.mp4" || failed=1
done < <("$PY" -c \
  "from environments.vessel_geometry import ALL_SCENARIOS,ANATOMICAL_SCENARIOS; print(''.join(f'{s} {0.0011 if s in ANATOMICAL_SCENARIOS else 0.0045}\\n' for s in ALL_SCENARIOS),end='')")

exit "$failed"
