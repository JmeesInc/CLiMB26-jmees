#!/usr/bin/env bash
# Build one submission variant and run the real-clip container regression.
#
# Everything that distinguishes a variant is a build ARG, so the code in the
# image is byte-identical across them and only the baked ENV differs. That
# matters because we ship several candidates per day and the only way to read
# the leaderboard is to know exactly one thing changed.
#
#   TAG=v012 MAP_FRAMES=350 ROTAVG=1 ROT_STEPS=1,2,6,7,8 RES=448 bash make_variant.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
TAG="${TAG:?set TAG, e.g. v012}"
IMAGE="climb-submaps:${TAG}"
REMOTE="docker.synapse.org/syn77166596/climb-da3:${TAG}"

IMAGE="${IMAGE}" bash "${HERE}/build.sh"

echo "== baked ENV =="
docker run --rm --runtime=runc --entrypoint env "${IMAGE}" | grep -E "^DA3_" | sort | sed 's/^/   /'

# Real clips, not the pinhole sim: rotavg needs the rectified K and is skipped
# without it, so a sim-only regression would not exercise the thing we changed.
echo "== real-clip regression (v011-500 reference: ATE 2.757 / rot 6.285 / TFR 100 / 4-4) =="
IMAGE="${IMAGE}" \
  INPUT_HOST="${REPO}/workspace/expB04_realcv/input" \
  COLMAP_GT="${REPO}/workspace/expB04_realcv/colmap_gt" \
  OUTPUT_HOST="${HERE}/out_${TAG}" \
  DA3_RECTIFY=1 DA3_STRIDE=1 \
  bash "${HERE}/test.sh"

docker tag "${IMAGE}" "${REMOTE}"
echo
echo "Tagged ${REMOTE}"
echo "Push manually:  docker push ${REMOTE}"
