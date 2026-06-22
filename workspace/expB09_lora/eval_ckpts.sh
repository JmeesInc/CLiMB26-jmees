#!/usr/bin/env bash
# Score every sim-LoRA checkpoint on the real CV clips, at the SHIPPED settings.
#
# Not the exploration harness: this reproduces what the v013 image bakes in
# (DA3_STRIDE=6, per-sequence calibration, no motion gate, res448, rotavg on,
# MAP_FRAMES=420), because the last two times a checkpoint was judged on
# harness defaults the comparison turned out to be against a different
# configuration entirely.
#
# Every checkpoint is scored, not just the last: the self-supervised LoRA peaked
# at 200 steps and was worse at 300 (4.249 / 4.075 / 4.486), so reading only the
# final weights would have picked the worst of the three.
set -euo pipefail
RUN="${1:?usage: eval_ckpts.sh <results/sim_rot_dir>}"
REPO=/data4/src/shunsuke/MICCAI2026/CLiMB
PY=/data4/src/shunsuke/MICCAI2026/iMED/.venv/bin/python
SP=/tmp/claude-1005/-data4-src-shunsuke-MICCAI2026-CLiMB/b2146ada-b3c4-48e7-a982-a404fb30051d/scratchpad
G="${G:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)}"
echo "== GPU $G, checkpoints in $RUN =="
echo "   reference (v013 container): ATE 2.72  rot 4.27"
for ck in "$RUN"/lora_*.pt; do
  tag="$(basename "$ck" .pt)"
  out="$SP/ck_$tag"
  CUDA_VISIBLE_DEVICES=$G env \
    DA3_PRECISION=fp16 DA3_STRIDE=6 NUM_RUNS=1 HF_HUB_OFFLINE=1 \
    DA3_RECTIFY=1 DA3_RECT_FSCALE=1.8 DA3_CHUNK=12 DA3_OVERLAP=6 DA3_RES=448 \
    DA3_MAP_FRAMES=420 DA3_ROTAVG=1 DA3_ROT_STEPS=1,2,6,7,8 DA3_ROT_STRIDE=6 \
    DA3_ROT_KP=512 DA3_ROT_SCALE=0.25 DA3_ROT_DTHR=0.0 DA3_ROT_POOL=4 \
    DA3_LORA="$ck" \
    "$PY" "$REPO/submit/v010_submaps/predict.py" \
    --input "$SP/real_input" --output "$out" > "$RUN/eval_$tag.log" 2>&1
  grep -q "LoRA loaded" "$RUN/eval_$tag.log" || { echo "   $tag: LoRA NOT LOADED -- skipping"; continue; }
  "$PY" "$REPO/reference/evaluation/slam_evaluation.py" \
    --colmap_path "$REPO/workspace/expB04_realcv/colmap_gt" --slam_path "$out" \
    --results_file "$RUN/eval_$tag.json" >/dev/null 2>&1
  "$PY" - "$RUN/eval_$tag.json" "$tag" <<'PYEOF'
import json, sys
g = json.load(open(sys.argv[1]))["global_mean"]
print(f"   {sys.argv[2]:12s} ATE {g['mean_ates']:6.3f}  rot {g['mean_rpe_deg_40frame']:6.3f}  "
      f"TFR {g['mean_matched_frames_percent']:5.1f}  {g['success_count']}/{g['num_sequences']}")
PYEOF
  rm -rf "$out"
done
echo EVAL_DONE
