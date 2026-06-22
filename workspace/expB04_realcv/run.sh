#!/usr/bin/env bash
# expB04: REAL-domain local CV, finally possible — official_data/Examples ships 4
# real colonoscopy clips WITH COLMAP GT (Seq_001_a/c, Seq_003_a/b, all from
# trainval sequences, so legitimate for validation).
#
# Primary question: official cameras.txt shows the GT was reconstructed with a
# SINGLE fixed calibration (Endoscope_07, exact match) for BOTH Seq_001 and
# Seq_003 — even though info.json assigns them endoscope 1 and 10. So v002's
# per-sequence calibration lookup is picking the WRONG model. CONFIG=e07 tests
# using the GT's own calibration for everything.
#
# CONFIG=seq|e07|norect ; STRIDE=1 ; GPU=1
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${EXP}/../.." && pwd)"
PY=/data4/src/shunsuke/MICCAI2026/iMED/.venv/bin/python
PRED="${PRED:-${REPO}/submit/v003_da3_stride/predict.py}"
GPU="${GPU:-1}"
STRIDE="${STRIDE:-1}"
CONFIG="${CONFIG:-e07}"
TAG="${TAG:-${CONFIG}_s${STRIDE}}"

declare -a ENV=(DA3_PRECISION=fp16 DA3_STRIDE=${STRIDE} NUM_RUNS=1 HF_HUB_OFFLINE=1)
[[ -n "${CHUNK:-}"   ]] && ENV+=(DA3_CHUNK=${CHUNK})
[[ -n "${OVERLAP:-}" ]] && ENV+=(DA3_OVERLAP=${OVERLAP})
[[ -n "${FSCALE:-}"  ]] && ENV+=(DA3_RECT_FSCALE=${FSCALE})
[[ -n "${DA3_ROTAVG:-}" ]] && ENV+=(DA3_ROTAVG=${DA3_ROTAVG})
[[ -n "${LORA:-}" ]] && ENV+=(DA3_LORA=${LORA})
[[ -n "${SOLVER:-}" ]] && ENV+=(DA3_ROT_SOLVER=${SOLVER})
[[ -n "${GATE:-}"   ]] && ENV+=(DA3_ROT_GATE=${GATE})
[[ -n "${ROTFOV:-}" ]] && ENV+=(DA3_ROT_FSCALE=${ROTFOV})
# The container sets these; running predict.py directly falls back to rotavg.py's
# own defaults (STEPS=1,4,7 and SCALE=0.5), which is NOT what ships. Match the
# image so CV numbers mean the same thing as container numbers.
ENV+=(DA3_ROT_STEPS=${ROTSTEPS:-1,4,7,10,13,16} DA3_ROT_SCALE=${ROTSCALE:-0.25} DA3_ROT_KP=${ROTKP:-512} DA3_ROT_STRIDE=${DA3_ROT_STRIDE:-6})
[[ -n "${RES:-}" ]] && ENV+=(DA3_RES=${RES})
[[ -n "${MODEL:-}" ]] && ENV+=(DA3_MODEL=${MODEL})
[[ -n "${BACKEND:-}" ]] && ENV+=(DA3_BACKEND=${BACKEND})
[[ -n "${PI3MODEL:-}" ]] && ENV+=(PI3_MODEL=${PI3MODEL})
[[ -n "${PXLIM:-}" ]] && ENV+=(PI3_PIXEL_LIMIT=${PXLIM})
[[ -n "${BA:-}" ]] && ENV+=(DA3_ROT_BA=${BA})
[[ -n "${RES:-}"     ]] && ENV+=(DA3_RES=${RES})
[[ -n "${COND:-}"    ]] && ENV+=(DA3_COND=${COND})
[[ -n "${PHASE:-}"   ]] && ENV+=(DA3_PHASE=${PHASE})
[[ -n "${DUMP:-}"    ]] && ENV+=(DA3_DUMP=${DUMP})
[[ -n "${LORA:-}"    ]] && ENV+=(DA3_LORA=${LORA})
[[ -n "${ANCH:-}"    ]] && ENV+=(DA3_ANCHORS=${ANCH})
case "${CONFIG}" in
  seq)    ENV+=(DA3_RECTIFY=1) ;;                        # v002 behaviour: SEQ2ENDO lookup
  e07)    ENV+=(DA3_RECTIFY=1 DA3_FORCE_ENDO=7) ;;       # GT's own calibration
  norect) ENV+=(DA3_RECTIFY=0) ;;                        # no undistortion (v001 behaviour)
  # --- inter-window scale ablation (diagnosis: s=1 is FALSE on real data) ---
  s_raw)  ENV+=(DA3_RECTIFY=1 DA3_SCALE=overlap) ;;
  s_d25)  ENV+=(DA3_RECTIFY=1 DA3_SCALE=damped DA3_SCALE_ALPHA=0.25 DA3_SCALE_CLAMP=1.10) ;;
  s_d50)  ENV+=(DA3_RECTIFY=1 DA3_SCALE=damped DA3_SCALE_ALPHA=0.5  DA3_SCALE_CLAMP=1.15) ;;
  s_d100) ENV+=(DA3_RECTIFY=1 DA3_SCALE=damped DA3_SCALE_ALPHA=1.0  DA3_SCALE_CLAMP=1.15) ;;
  *) echo "bad CONFIG"; exit 1 ;;
esac

out="${EXP}/out_${TAG}"
echo "== [${TAG}] ${ENV[*]} =="
CUDA_VISIBLE_DEVICES=${GPU} env "${ENV[@]}" "${PY}" "${PRED}" \
    --input "${EXP}/input" --output "${out}" > "${EXP}/logs/${TAG}.log" 2>&1
grep -E "rectify|s/frame" "${EXP}/logs/${TAG}.log" || true

python3 "${REPO}/reference/evaluation/slam_evaluation.py" \
    --colmap_path "${EXP}/colmap_gt" --slam_path "${out}" \
    --results_file "${EXP}/results/${TAG}.json" 2>/dev/null | grep -A10 "GLOBAL MEAN"
