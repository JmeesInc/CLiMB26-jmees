#!/usr/bin/env bash
# Local regression test: run the image on the sim sequences exactly as the
# organizers will, verify the output tree, then score it with the official
# evaluator against the sim COLMAP reference.
#
# Expected (workspace/expB01_da3_submap, s=1, PNG input): Seq_0 ATE 3.11 mm,
# 6-seq mean 5.25 mm / TFR 96.3. This test feeds mp4 instead of PNG, so small
# differences from H.264 colour conversion are normal; a large gap is a bug.
#
# GPU on this host: docker's default-runtime is a broken 'nvidia' (the
# nvidia-container-runtime binary is missing), so --gpus all cannot be used.
# We pass the devices through with --runtime=runc and bind-mount the driver
# libraries instead. The organizers' host uses plain `--gpus all`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
IMAGE="${IMAGE:-climb-da3-stride:v003}"
INPUT_HOST="${INPUT_HOST:-${REPO}/workspace/expA00_baseline_eval/sim_input}"
OUTPUT_HOST="${OUTPUT_HOST:-${HERE}/output}"
COLMAP_GT="${COLMAP_GT:-${REPO}/workspace/expA00_baseline_eval/colmap_gt}"
SEQS="${SEQS:-}"          # e.g. SEQS="Seq_0" to test a single sequence
# Default to whichever GPU has the most free memory: this is a shared host and
# the pipeline needs ~20 GB.
GPU_INDEX="${GPU_INDEX:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')}"

# A single-sequence test needs an /input holding just that mp4. Copy rather than
# symlink: a link into the host filesystem does not resolve inside the container.
if [[ -n "${SEQS}" ]]; then
  STAGE="${HERE}/.test_input"
  rm -rf "${STAGE}"; mkdir -p "${STAGE}"
  for s in ${SEQS}; do cp "${INPUT_HOST}/${s}.mp4" "${STAGE}/${s}.mp4"; done
  INPUT_HOST="${STAGE}"
fi

mkdir -p "${OUTPUT_HOST}"

GPU_ARGS=(--runtime=runc)
# Pass every device node and select the GPU by UUID. /dev/nvidiaN is a minor
# number, which does NOT track the nvidia-smi index -- passing only /dev/nvidia1
# once handed the container a completely different (and busy) GPU.
for d in /dev/nvidia[0-9] /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
  [[ -e "${d}" ]] && GPU_ARGS+=(--device "${d}:${d}")
done
GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${GPU_INDEX}" | tr -d ' ')"
DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
for lib in libcuda libnvidia-ml libnvidia-ptxjitcompiler; do
  src="/usr/lib/x86_64-linux-gnu/${lib}.so.${DRV}"
  [[ -e "${src}" ]] && GPU_ARGS+=(-v "${src}:/usr/lib/x86_64-linux-gnu/${lib}.so.1:ro")
done

echo "== run container (driver ${DRV}, GPU ${GPU_INDEX} = ${GPU_UUID}) =="
# DA3_RECTIFY=0 for the pinhole sim regression (must reproduce v001's numbers);
# leave unset/1 when feeding fisheye or real clips. DA3_PRECISION=fp16 avoids
# this Turing host's 4.2x-slow emulated bf16 (eval host is Blackwell -> auto).
docker run --rm "${GPU_ARGS[@]}" \
  --network=none \
  --memory=64g \
  --user "$(id -u)":"$(id -g)" \
  -e CUDA_VISIBLE_DEVICES="${GPU_UUID}" \
  -e DA3_RECTIFY="${DA3_RECTIFY:-1}" \
  -e DA3_PRECISION="${DA3_PRECISION:-auto}" \
  -e DA3_STRIDE="${DA3_STRIDE:-6}" \
  -e DA3_CHUNK="${DA3_CHUNK:-12}" \
  -e DA3_OVERLAP="${DA3_OVERLAP:-6}" \
  -v "${INPUT_HOST}:/input:ro" \
  -v "${OUTPUT_HOST}:/output" \
  "${IMAGE}"

echo "== validate submission tree =="
shopt -s nullglob
fail=0
# A silent pass on an empty /input has bitten us once (symlinked mp4s do not
# resolve inside the container), so make "no inputs" a hard failure.
n_in=0; for v in "${INPUT_HOST}"/*.mp4; do n_in=$((n_in+1)); done
[[ ${n_in} -gt 0 ]] || { echo "   NO INPUT mp4 under ${INPUT_HOST}"; exit 1; }
for v in "${INPUT_HOST}"/*.mp4; do
  seq="$(basename "${v}" .mp4)"
  for run in 1 2 3 4 5; do
    for f in "camera_trajectory/cam_traj_map_000.txt" "3D_maps/000/points3D.txt" "runtime.txt"; do
      p="${OUTPUT_HOST}/${seq}/${run}/${f}"
      if [[ ! -s "${p}" ]]; then echo "   MISSING ${p}"; fail=1; fi
    done
  done
  t="${OUTPUT_HOST}/${seq}/1/camera_trajectory/cam_traj_map_000.txt"
  if [[ -s "${t}" ]]; then
    first_id=$(awk -F, '!/^#/{print $2; exit}' "${t}")
    [[ "${first_id}" == "000001.png" ]] || { echo "   BAD first frame id: ${first_id} (want 000001.png)"; fail=1; }
    grep -q "processing_seconds=" "${OUTPUT_HOST}/${seq}/1/runtime.txt" || { echo "   BAD runtime.txt"; fail=1; }
  fi
done
[[ ${fail} -eq 0 ]] && echo "   tree OK" || { echo "   TREE VALIDATION FAILED"; exit 1; }

echo "== score with the official evaluator =="
python3 "${REPO}/reference/evaluation/slam_evaluation.py" \
  --colmap_path "${COLMAP_GT}" \
  --slam_path   "${OUTPUT_HOST}" \
  --results_file "${HERE}/results_docker.json" \
  --verbose 2>/dev/null | grep -A20 "GLOBAL MEAN" || true
echo "Results: ${HERE}/results_docker.json"
