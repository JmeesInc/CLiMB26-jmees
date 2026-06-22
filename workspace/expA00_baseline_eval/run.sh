#!/usr/bin/env bash
# expA00 baseline evaluation: ORB-SLAM3 on EndoMapper Simulated_Sequences,
# scored locally against sim GT converted to COLMAP reference format.
#
# Stages (control with STAGE=mp4|gt|slam|eval|all):
#   mp4  : rgb PNG sequences -> sim_input/<seq>.mp4
#   gt   : trajectory.csv    -> colmap_gt/ (COLMAP tree + scales/traj/frames CSVs)
#   slam : run ORB-SLAM3 docker (EXECUTIONS runs/seq) -> output/<seq>/<run>/
#   eval : reference/evaluation/slam_evaluation.py -> results.json
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${EXP_DIR}/../.." && pwd)"
DATA_SIM="${REPO}/data/Simulated_Sequences"
SIM_INPUT="${EXP_DIR}/sim_input"
COLMAP_GT="${EXP_DIR}/colmap_gt"
OUTPUT="${EXP_DIR}/output"
SCRIPTS="${EXP_DIR}/scripts"
IMAGE="${IMAGE:-endovis-orbslam-submission:ci}"
EXECUTIONS="${EXECUTIONS:-5}"
FPS="${FPS:-30}"
STAGE="${STAGE:-all}"

mkdir -p "${SIM_INPUT}" "${COLMAP_GT}" "${OUTPUT}"

if [[ "${STAGE}" == "mp4" || "${STAGE}" == "all" ]]; then
  echo "== [mp4] rgb PNG -> mp4 =="
  for d in "${DATA_SIM}"/Seq_*; do
    [[ -d "${d}/rgb" ]] || continue
    seq="$(basename "${d}")"
    out="${SIM_INPUT}/${seq}.mp4"
    # Lossless H.264 from the contiguous image_%04d.png sequence (frame order preserved).
    ffmpeg -y -loglevel error -framerate "${FPS}" -start_number 0 \
      -i "${d}/rgb/image_%04d.png" \
      -c:v libx264 -qp 0 -pix_fmt yuv420p "${out}"
    n=$(ffprobe -v error -count_frames -select_streams v:0 \
        -show_entries stream=nb_read_frames -of csv=p=0 "${out}")
    echo "  ${seq}.mp4: ${n} frames"
  done
fi

if [[ "${STAGE}" == "gt" || "${STAGE}" == "all" ]]; then
  echo "== [gt] sim GT -> COLMAP reference =="
  python3 "${SCRIPTS}/sim_gt_to_colmap.py" --sim_root "${DATA_SIM}" --out "${COLMAP_GT}"
fi

if [[ "${STAGE}" == "slam" || "${STAGE}" == "all" ]]; then
  echo "== [slam] ORB-SLAM3 (EXECUTIONS=${EXECUTIONS}) =="
  # ORB-SLAM3 monocular is CPU-only; this host's docker default-runtime is a
  # broken 'nvidia' (no nvidia-container-runtime binary), so force runc.
  RUNTIME_ARGS=(--runtime=runc)
  docker run --rm "${RUNTIME_ARGS[@]}" --network=none --memory=64g \
    --user "$(id -u)":"$(id -g)" \
    -e EXECUTIONS="${EXECUTIONS}" -e SETTINGS=/opt/sim.yaml \
    -v "${SIM_INPUT}:/input:ro" \
    -v "${OUTPUT}:/output" \
    -v "${SCRIPTS}/Sim_Pinhole.yaml:/opt/sim.yaml:ro" \
    -v "${SCRIPTS}/sim_entrypoint.sh:/opt/sim_entrypoint.sh:ro" \
    --entrypoint /opt/sim_entrypoint.sh \
    "${IMAGE}" /input /output
fi

if [[ "${STAGE}" == "eval" || "${STAGE}" == "all" ]]; then
  echo "== [eval] slam_evaluation.py =="
  python3 "${REPO}/reference/evaluation/slam_evaluation.py" \
    --colmap_path "${COLMAP_GT}" \
    --slam_path   "${OUTPUT}" \
    --results_file "${EXP_DIR}/results.json" \
    --verbose | tee "${EXP_DIR}/logs/eval_$(date +%Y%m%d_%H%M%S).log"
  echo "Results: ${EXP_DIR}/results.json"
fi
