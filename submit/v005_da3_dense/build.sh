#!/usr/bin/env bash
# Stage the build context (DA3 source + baked weights) and build the image.
#
# Both staged trees are git-ignored: the weights are 6.3 GB and the DA3 source
# is a vendored upstream checkout.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-climb-da3-dense:v005}"

DA3_SRC="${DA3_SRC:-/data4/src/shunsuke/Depth-Anything-3}"
DA3_COMMIT="f64bffe"          # ByteDance-Seed/Depth-Anything-3, the validated rev
MODEL_ID_PATH="models--depth-anything--DA3NESTED-GIANT-LARGE"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface/hub}"

echo "== stage DA3 source (${DA3_COMMIT}) =="
if [[ ! -d "${DA3_SRC}/src/depth_anything_3" ]]; then
  echo "ERROR: DA3 checkout not found at ${DA3_SRC}." >&2
  echo "       git clone https://github.com/ByteDance-Seed/Depth-Anything-3 ${DA3_SRC}" >&2
  exit 1
fi
mkdir -p "${HERE}/vendor/depth_anything_3_src"
rsync -a --delete "${DA3_SRC}/src" "${DA3_SRC}/pyproject.toml" "${DA3_SRC}/README.md" \
      "${HERE}/vendor/depth_anything_3_src/"

echo "== stage weights (${MODEL_ID_PATH}) =="
if [[ ! -d "${HF_CACHE}/${MODEL_ID_PATH}" ]]; then
  echo "ERROR: weights not in ${HF_CACHE}. Fetch them once with network access:" >&2
  echo "       huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE" >&2
  exit 1
fi
mkdir -p "${HERE}/model/hf"
# -a keeps the relative blobs/<-snapshots symlinks, which resolve inside the image.
rsync -a --delete "${HF_CACHE}/${MODEL_ID_PATH}" "${HERE}/model/hf/"
du -sh "${HERE}/model/hf" | sed 's/^/   /'

echo "== docker build ${IMAGE} =="
docker build -t "${IMAGE}" "${HERE}"
docker images --format '   {{.Repository}}:{{.Tag}}  {{.Size}}' | grep "${IMAGE%%:*}" || true
echo "Built ${IMAGE}"
