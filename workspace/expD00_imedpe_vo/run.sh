#!/usr/bin/env bash
# expD00 iMED-PE (tri3d) port: DA3 pair depth + ALIKED/LightGlue + IRLS Umeyama VO.
# Also writes the DA3 pair-extrinsic chain as a free ablation tree.
# STAGE=slam|eval|all ; MODEL, FIT, GPU, SEQS, OUT_LG, OUT_DA3 overridable.
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP}/../.." && pwd)"
PY="/data4/src/shunsuke/MICCAI2026/iMED/.venv/bin/python"   # DA3 + lightglue
FRAMES="${REPO_ROOT}/workspace/expC00_vggtslam/sim_input"
GT="${REPO_ROOT}/workspace/expA00_baseline_eval/colmap_gt"

MODEL="${MODEL:-depth-anything/DA3NESTED-GIANT-LARGE}"
FIT="${FIT:-se3}"
RES="${RES:-504}"
GPU="${GPU:-0}"
RUNS="${RUNS:-5}"
OUT_LG="${OUT_LG:-${EXP}/output_lg}"
OUT_DA3="${OUT_DA3:-${EXP}/output_da3pair}"
RESULTS_LG="${RESULTS_LG:-${EXP}/results_lg.json}"
RESULTS_DA3="${RESULTS_DA3:-${EXP}/results_da3pair.json}"
SEQS="${SEQS:-}"
STAGE="${STAGE:-all}"
mkdir -p "${EXP}/logs"

if [[ "${STAGE}" == "slam" || "${STAGE}" == "all" ]]; then
  CUDA_VISIBLE_DEVICES=${GPU} HF_HUB_OFFLINE=1 "${PY}" "${EXP}/scripts/vo_pair.py" \
    --frames_root "${FRAMES}" --out_lg "${OUT_LG}" --out_da3 "${OUT_DA3}" \
    --model "${MODEL}" --fit "${FIT}" --process_res "${RES}" --runs "${RUNS}" \
    ${SEQS:+--seqs ${SEQS}} \
    2>&1 | tee "${EXP}/logs/slam_$(date +%Y%m%d_%H%M%S).log"
fi

if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
  for pair in "${OUT_LG}:${RESULTS_LG}" "${OUT_DA3}:${RESULTS_DA3}"; do
    outdir="${pair%%:*}"; res="${pair##*:}"
    [[ -d "${outdir}" ]] || continue
    echo "== eval ${outdir} =="
    python3 "${REPO_ROOT}/reference/evaluation/slam_evaluation.py" \
      --colmap_path "${GT}" --slam_path "${outdir}" \
      --results_file "${res}" --verbose \
      2>/dev/null | grep -A30 "GLOBAL MEAN" || true
    echo "Results: ${res}"
  done
fi
