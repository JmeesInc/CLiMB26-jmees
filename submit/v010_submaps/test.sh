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
IMAGE="${IMAGE:-climb-submaps:v010}"
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

# A bind-mounted symlink does not resolve inside the container: the target path
# does not exist there. The CV input dir is all symlinks, so mounting it gives
# the container an empty /input and the run exits with "no .mp4 files". Stage
# real copies whenever any input is a link.
if compgen -G "${INPUT_HOST}/*.mp4" >/dev/null && \
   [[ -n "$(find "${INPUT_HOST}" -maxdepth 1 -name '*.mp4' -type l -print -quit)" ]]; then
  STAGE="${STAGE_DIR:-${HERE}/.test_input_$(basename "${OUTPUT_HOST}")}"
  mkdir -p "${STAGE}"
  for v in "${INPUT_HOST}"/*.mp4; do
    dst="${STAGE}/$(basename "${v}")"
    [[ -s "${dst}" ]] || cp -L "${v}" "${dst}"
  done
  echo "   staged $(ls "${STAGE}"/*.mp4 | wc -l) symlinked input(s) -> ${STAGE}"
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
# Two defaults here have each cost a day of chasing a phantom regression:
#
#   DA3_RECTIFY defaulted to 1, which applied the kb4 undistortion to the
#   ALREADY-PINHOLE sim clips. That is what produced the "bf16 breaks the LoRA"
#   result (sim 5.25 -> 8.87) later withdrawn on 8/30. The sim regression needs 0.
#
#   DA3_PRECISION defaulted to "auto", which OVERRODE the fp16 the image pins --
#   and on a Turing host auto means bf16. The organizers pass no -e at all, so
#   anything forced here tests something the submission will never run. It is
#   now passed only when the caller explicitly sets it.
#
# Set DA3_RECTIFY=1 explicitly when feeding fisheye or real clips.
# Only CUDA_VISIBLE_DEVICES and DA3_RECTIFY are forced; everything else is
# passed ONLY when the caller sets it explicitly. Hardcoded defaults here shadow
# the image's own ENV, which is the whole point of a build-arg'd Dockerfile --
# a `-e DA3_OVERLAP=6` default silently turned an OVERLAP=8 image back into a 6.
RUN_ENV=(-e CUDA_VISIBLE_DEVICES="${GPU_UUID}"
         -e DA3_RECTIFY="${DA3_RECTIFY:-0}")
for v in DA3_PRECISION DA3_STRIDE DA3_CHUNK DA3_OVERLAP DA3_ROT_DESPEC DA3_DUMP \
         DA3_HALF DA3_RES DA3_MAP_FRAMES DA3_MAP_MIN_SPAN DA3_ROTAVG DA3_ROT_PROF; do
  [[ -n "${!v:-}" ]] && RUN_ENV+=(-e "${v}=${!v}") || true
done
docker run --rm "${GPU_ARGS[@]}" \
  --network=none \
  --memory=64g \
  --user "$(id -u)":"$(id -g)" \
  "${RUN_ENV[@]}" \
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
    # The official validator only requires 1-based IDs (validate_submission.py:
    # "frame IDs must be 1-based"); map 000 need not start at frame 1, e.g. when
    # DA3_MAP_DROP discards the first sub-map. Reject only a 0-based id.
    [[ "${first_id}" != "000000.png" ]] || { echo "   BAD first frame id: ${first_id} (0-based)"; fail=1; }
    grep -q "processing_seconds=" "${OUTPUT_HOST}/${seq}/1/runtime.txt" || { echo "   BAD runtime.txt"; fail=1; }
  fi
done
[[ ${fail} -eq 0 ]] && echo "   tree OK" || { echo "   TREE VALIDATION FAILED"; exit 1; }

echo "== score with the official evaluator =="
python3 "${REPO}/reference/evaluation/slam_evaluation.py" \
  --colmap_path "${COLMAP_GT}" \
  --slam_path   "${OUTPUT_HOST}" \
  --results_file "${RESULTS_FILE:-${HERE}/results_$(basename "${OUTPUT_HOST}").json}" \
  --verbose 2>/dev/null | grep -A20 "GLOBAL MEAN" || true
echo "Results: ${HERE}/results_docker.json"
