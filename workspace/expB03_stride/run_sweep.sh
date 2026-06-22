#!/usr/bin/env bash
# expB03: keyframe-stride sweep for v003. Runs the DEPLOYED predict.py directly on
# the sim mp4s (same code path as the container) and scores with the official
# evaluator. Pinhole sim -> DA3_RECTIFY=0.
#
# Read the curve with the motion caveat: real sequences move ~3x faster per frame
# than sim (23 px vs 8 px), so "real stride k" ~ "sim stride 3k" in disparity.
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${EXP}/../.." && pwd)"
PY=/data4/src/shunsuke/MICCAI2026/iMED/.venv/bin/python
PRED="${REPO}/submit/v003_da3_stride/predict.py"
INPUT="${REPO}/workspace/expA00_baseline_eval/sim_input"
GT="${REPO}/workspace/expA00_baseline_eval/colmap_gt"
GPU="${GPU:-3}"
STRIDES="${STRIDES:-2 4 6 8 12}"

for s in ${STRIDES}; do
  out="${EXP}/out_s${s}"
  echo "===== stride ${s} ====="
  CUDA_VISIBLE_DEVICES=${GPU} DA3_PRECISION=fp16 DA3_RECTIFY=0 DA3_STRIDE=${s} NUM_RUNS=1 \
    HF_HUB_OFFLINE=1 "${PY}" "${PRED}" --input "${INPUT}" --output "${out}" \
    > "${EXP}/logs/stride_${s}.log" 2>&1
  grep -E "s/frame" "${EXP}/logs/stride_${s}.log" | tail -6
  python3 "${REPO}/reference/evaluation/slam_evaluation.py" \
    --colmap_path "${GT}" --slam_path "${out}" \
    --results_file "${EXP}/results/s${s}.json" 2>/dev/null | grep -E "^Mean"
done
