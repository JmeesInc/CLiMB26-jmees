#!/usr/bin/env bash
# expB01 DA3 submap-chunked SLAM on sim sequences, scored vs expA00 colmap_gt.
# STAGE=slam|eval|all ; MODEL, CHUNK, OVERLAP, RES, GPU, SEQS, OUTPUT, RESULTS overridable.
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP}/../.." && pwd)"
PY="/data4/src/shunsuke/MICCAI2026/iMED/.venv/bin/python"   # DA3 + torch cu126
FRAMES="${REPO_ROOT}/workspace/expC00_vggtslam/sim_input"   # 1-based {id:06d}.png
GT="${REPO_ROOT}/workspace/expA00_baseline_eval/colmap_gt"

MODEL="${MODEL:-depth-anything/DA3NESTED-GIANT-LARGE}"
CHUNK="${CHUNK:-16}"
OVERLAP="${OVERLAP:-6}"
RES="${RES:-504}"
GPU="${GPU:-0}"
RUNS="${RUNS:-5}"
OUTPUT="${OUTPUT:-${EXP}/output}"
RESULTS="${RESULTS:-${EXP}/results.json}"
SEQS="${SEQS:-}"
STAGE="${STAGE:-all}"
mkdir -p "${EXP}/logs"

if [[ "${STAGE}" == "slam" || "${STAGE}" == "all" ]]; then
  CUDA_VISIBLE_DEVICES=${GPU} HF_HUB_OFFLINE=1 "${PY}" "${EXP}/scripts/chunk_slam.py" \
    --frames_root "${FRAMES}" --out "${OUTPUT}" \
    --model "${MODEL}" --chunk "${CHUNK}" --overlap "${OVERLAP}" \
    --process_res "${RES}" --runs "${RUNS}" \
    --s_lo "${S_LO:-0.25}" --s_hi "${S_HI:-4.0}" \
    ${SEQS:+--seqs ${SEQS}} \
    2>&1 | tee "${EXP}/logs/slam_$(date +%Y%m%d_%H%M%S).log"
fi

if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
  python3 "${REPO_ROOT}/reference/evaluation/slam_evaluation.py" \
    --colmap_path "${GT}" --slam_path "${OUTPUT}" \
    --results_file "${RESULTS}" --verbose \
    2>/dev/null | grep -A30 "GLOBAL MEAN" || true
  echo "Results: ${RESULTS}"
fi
