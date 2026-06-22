#!/usr/bin/env bash
# expB02: fisheye-domain verification for the v001 LB failure (ATE 28.3 vs CV 4.13).
#
# 2x2 evidence on sim (config A = existing expB01 results_s1: pinhole, no rectify):
#   B: synthetic kb4 fisheye  -> pipeline as-is        (expect blow-up if hypothesis true)
#   C: synthetic kb4 fisheye  -> rectify -> pipeline   (expect recovery toward A)
#
# STAGE=prep|slam|eval|all ; SEQS="Seq_0 Seq_5" (subset default for speed) ; GPU=0
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${EXP}/../.." && pwd)"
PY_IMED="/data4/src/shunsuke/MICCAI2026/iMED/.venv/bin/python"
SRC="${REPO}/workspace/expC00_vggtslam/sim_input"     # pinhole frames, 1-based ids
CHUNK_SLAM="${REPO}/workspace/expB01_da3_submap/scripts/chunk_slam.py"
GT="${REPO}/workspace/expA00_baseline_eval/colmap_gt"

SEQS="${SEQS:-Seq_0 Seq_5}"
GPU="${GPU:-0}"
STAGE="${STAGE:-all}"
CHUNK="${CHUNK:-8}"; OV="${OV:-3}"

if [[ "${STAGE}" == "prep" || "${STAGE}" == "all" ]]; then
  echo "== [prep] synth fisheye + rectify =="
  "${PY_IMED}" "${EXP}/scripts/fisheye_tools.py" synth \
      --src_root "${SRC}" --dst_root "${EXP}/frames_fisheye" --seqs ${SEQS}
  "${PY_IMED}" "${EXP}/scripts/fisheye_tools.py" rectify \
      --src_root "${EXP}/frames_fisheye" --dst_root "${EXP}/frames_rectified" --seqs ${SEQS}
fi

if [[ "${STAGE}" == "slam" || "${STAGE}" == "all" ]]; then
  for cfg in fisheye rectified; do
    echo "== [slam] ${cfg} =="
    CUDA_VISIBLE_DEVICES=${GPU} "${PY_IMED}" "${CHUNK_SLAM}" \
        --frames_root "${EXP}/frames_${cfg}" --out "${EXP}/output_${cfg}" \
        --seqs ${SEQS} --chunk ${CHUNK} --overlap ${OV} --s_lo 1 --s_hi 1 \
        2>&1 | tee "${EXP}/logs/slam_${cfg}.log"
  done
fi

if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
  for cfg in fisheye rectified; do
    echo "== [eval] ${cfg} =="
    python3 "${REPO}/reference/evaluation/slam_evaluation.py" \
        --colmap_path "${GT}" --slam_path "${EXP}/output_${cfg}" \
        --results_file "${EXP}/results_${cfg}.json" --verbose \
        2>/dev/null | grep -A12 "GLOBAL MEAN" || true
  done
fi
