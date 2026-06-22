#!/usr/bin/env bash
# expC00 VGGT-SLAM evaluation on EndoMapper sim sequences, scored against the
# expA00 sim->COLMAP GT with the official evaluator.
#
# STAGE=slam|adapt|eval|all ; POSE_CONV=c2w|w2c ; SEQS="Seq_0 Seq_1 ..."
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP}/../.." && pwd)"
VENV="${EXP}/.venv/bin/python"
SLAMREPO="${EXP}/repo"
SIM_INPUT="${EXP}/sim_input"
OUTPUT="${OUTPUT:-${EXP}/output}"
RAWDIR="${RAWDIR:-${EXP}/raw}"
RESULTS="${RESULTS:-${EXP}/results.json}"
GT="${REPO_ROOT}/workspace/expA00_baseline_eval/colmap_gt"
SCRIPTS="${EXP}/scripts"

SUBMAP="${SUBMAP:-16}"
MAXLOOPS="${MAXLOOPS:-0}"
MINDISP="${MINDISP:-10}"
POSE_CONV="${POSE_CONV:-c2w}"
RUNS="${RUNS:-5}"
GPU="${GPU:-1}"
STAGE="${STAGE:-all}"
SEQS="${SEQS:-$(ls "${SIM_INPUT}")}"

mkdir -p "${OUTPUT}" "${EXP}/logs" "${RAWDIR}"

for seq in ${SEQS}; do
  raw="${RAWDIR}/${seq}"
  mkdir -p "${raw}"
  if [[ "${STAGE}" == "slam" || "${STAGE}" == "all" ]]; then
    echo "== [slam] ${seq} =="
    t0=$(date +%s)
    ( cd "${SLAMREPO}" && CUDA_VISIBLE_DEVICES=${GPU} HF_HUB_OFFLINE=1 \
      "${VENV}" main.py --image_folder "${SIM_INPUT}/${seq}" \
        --submap_size ${SUBMAP} --max_loops ${MAXLOOPS} --min_disparity ${MINDISP} \
        --log_results --log_path "${raw}/poses.txt" ) \
      > "${EXP}/logs/slam_${seq}.log" 2>&1
    echo $(( $(date +%s) - t0 )) > "${raw}/proc_seconds.txt"
    echo "  done in $(cat ${raw}/proc_seconds.txt)s"
  fi
  if [[ "${STAGE}" == "adapt" || "${STAGE}" == "all" ]]; then
    psec=$(cat "${raw}/proc_seconds.txt" 2>/dev/null || echo 0)
    pcd="${raw}/poses_points.pcd"
    for run in $(seq 1 ${RUNS}); do
      "${VENV}" "${SCRIPTS}/poses_to_climb.py" \
        --poses "${raw}/poses.txt" \
        $( [[ -f "${pcd}" ]] && echo --pcd "${pcd}" ) \
        --out_run_dir "${OUTPUT}/${seq}/${run}" \
        --pose_conv "${POSE_CONV}" --proc_seconds "${psec}"
    done
  fi
done

if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
  echo "== [eval] =="
  python3 "${REPO_ROOT}/reference/evaluation/slam_evaluation.py" \
    --colmap_path "${GT}" --slam_path "${OUTPUT}" \
    --results_file "${RESULTS}" --verbose \
    2>/dev/null | grep -A30 "GLOBAL MEAN" || true
  echo "Results: ${RESULTS}"
fi
