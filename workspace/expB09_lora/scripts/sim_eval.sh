#!/usr/bin/env bash
# sim_eval.sh <tag> [lora.pt]: run predict_cond on the 6 sim mp4s (no rectify) and score vs expA00 colmap_gt.
set -u
TAG=$1; LORA=${2:-}
R=/data4/src/shunsuke/MICCAI2026/CLiMB; E=$R/workspace/expB09_lora; OUT=$E/sim_eval/$TAG; mkdir -p $OUT $E/sim_eval/logs
env DA3_PRECISION=fp16 DA3_STRIDE=6 DA3_CHUNK=12 DA3_OVERLAP=6 DA3_RECTIFY=0 NUM_RUNS=1 HF_HUB_OFFLINE=1 ${LORA:+DA3_LORA=$LORA} DA3_LORA_R=8 DA3_LORA_ALPHA=16 \
  /data4/src/shunsuke/MICCAI2026/iMED/.venv/bin/python $R/workspace/expB06_posecond/scripts/predict_cond.py \
  --input $R/workspace/expA00_baseline_eval/sim_input --output $OUT > $E/sim_eval/logs/$TAG.log 2>&1
python3 $R/reference/evaluation/slam_evaluation.py --colmap_path $R/workspace/expA00_baseline_eval/colmap_gt --slam_path $OUT --results_file $E/sim_eval/$TAG.json > $E/sim_eval/logs/${TAG}_eval.log 2>&1
python3 -c "
import json; d=json.load(open('$E/sim_eval/$TAG.json')); g=d['global_mean']; ps=d['per_sequence_mean']
print(f\"$TAG: sim ATE {g['mean_ates']:.3f} rot {g['mean_rpe_deg_40frame']:.2f} TFR {g['mean_matched_frames_percent']:.1f} | \"+'  '.join(f\"{k[-1]} {v['mean_ate']:.2f}\" for k,v in sorted(ps.items())))" > $E/sim_eval/$TAG.summary
cat $E/sim_eval/$TAG.summary
