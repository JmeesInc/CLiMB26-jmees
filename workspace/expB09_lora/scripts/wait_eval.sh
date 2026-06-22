#!/usr/bin/env bash
# wait_eval.sh <run> <step> <gpu> [rank alpha]: wait until the run has saved
# step <step>, freeze that checkpoint, evaluate it on the real 4-clip CV.
set -u
RUN=$1; STEP=$2; GPU=$3; R=${4:-8}; A=${5:-16}
E=/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expB09_lora
CV=/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expB04_realcv
until [ -f "$E/results/$RUN/training_log.json" ] && python3 -c "import json,sys; sys.exit(0 if len(json.load(open('$E/results/$RUN/training_log.json')))>=$STEP else 1)"; do sleep 20; done
sleep 5; cp "$E/results/$RUN/lora_last.pt" "$E/results/$RUN/lora_step$(printf %04d $STEP).pt"
TAG="lora_${RUN}_s${STEP}"
export DA3_LORA_R=$R DA3_LORA_ALPHA=$A
PRED=/data4/src/shunsuke/MICCAI2026/CLiMB/workspace/expB06_posecond/scripts/predict_cond.py CONFIG=seq STRIDE=6 CHUNK=12 OVERLAP=6 LORA="$E/results/$RUN/lora_step$(printf %04d $STEP).pt" TAG=$TAG GPU=$GPU bash $CV/run.sh > $CV/logs/$TAG.log 2>&1
cd $CV && python3 -c "
import json
for t in ['c12o6_s6','$TAG']:
    d=json.load(open(f'results/{t}.json')); g=d['global_mean']; ps=d['per_sequence_mean']
    print(f\"{t:24s} ATE {g['mean_ates']:.3f} rot {g['mean_rpe_deg_40frame']:.2f} | \"+'  '.join(f\"{k[-5:]} {v['mean_ate']:.2f}\" for k,v in sorted(ps.items())))" > $CV/logs/$TAG.summary 2>&1
cat $CV/logs/$TAG.summary
