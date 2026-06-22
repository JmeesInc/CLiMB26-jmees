#!/usr/bin/env bash
# Stage the build context (DA3 source + baked weights) and build the image.
#
# Both staged trees are git-ignored: the weights are 6.3 GB and the DA3 source
# is a vendored upstream checkout.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-climb-pi3x:v009}"

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

echo "== stage lightglue + ALIKED/LightGlue weights =="
mkdir -p "${HERE}/vendor" "${HERE}/torchhub/hub/checkpoints"
rsync -a /data4/src/shunsuke/MICCAI2026/iMED/.venv/lib/python3.12/site-packages/lightglue/ "${HERE}/vendor/lightglue/"
cp -f ~/.cache/torch/hub/checkpoints/aliked-n16.pth ~/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth "${HERE}/torchhub/hub/checkpoints/"

echo "== stage Pi3 source + weights =="
PI3_SRC="${PI3_SRC:-/data4/src/shunsuke/Pi3}"
[[ -d "${PI3_SRC}/pi3" ]] || { echo "ERROR: Pi3 checkout not at ${PI3_SRC}" >&2; exit 1; }
mkdir -p "${HERE}/vendor/pi3"
rsync -a --delete --exclude examples --exclude assets --exclude .git \
      "${PI3_SRC}/pi3" "${PI3_SRC}/requirements.txt" "${HERE}/vendor/pi3/"
PI3_ID="models--yyfz233--Pi3X"
[[ -d "${HF_CACHE}/${PI3_ID}" ]] || { echo "ERROR: Pi3X weights not in ${HF_CACHE}" >&2; exit 1; }
rsync -a --delete "${HF_CACHE}/${PI3_ID}" "${HERE}/model/hf/"
du -sh "${HERE}/model/hf" "${HERE}/vendor/pi3" | sed 's/^/   /'

echo "== docker build ${IMAGE} =="
docker build -t "${IMAGE}" "${HERE}"
docker images --format '   {{.Repository}}:{{.Tag}}  {{.Size}}' | grep "${IMAGE%%:*}" || true
echo "Built ${IMAGE}"
