#!/usr/bin/env bash
# Stage the build context (DA3 source + baked weights) and build the image.
#
# Both staged trees are git-ignored: the weights are 6.3 GB and the DA3 source
# is a vendored upstream checkout.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-climb-submaps:v010}"

DA3_SRC="${DA3_SRC:-/data4/src/shunsuke/Depth-Anything-3}"
DA3_COMMIT="f64bffe"          # ByteDance-Seed/Depth-Anything-3, the validated rev
MODEL="${MODEL:-depth-anything/DA3NESTED-GIANT-LARGE}"
MODEL_ID_PATH="models--${MODEL//\//--}"
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
# LIGHTGLUE_SRC: an installed `lightglue` package dir (pip install git+https://github.com/cvg/LightGlue).
# TORCH_CKPT: torch hub cache holding aliked-n16.pth and aliked_lightglue_v0-1_arxiv.pth
# (the package downloads both on first use).
LIGHTGLUE_SRC="${LIGHTGLUE_SRC:-$(python3 -c 'import lightglue,os;print(os.path.dirname(lightglue.__file__))' 2>/dev/null || true)}"
TORCH_CKPT="${TORCH_CKPT:-${HOME}/.cache/torch/hub/checkpoints}"
if [[ ! -f "${LIGHTGLUE_SRC:-/nonexistent}/lightglue.py" ]]; then
  echo "ERROR: lightglue package not found. Set LIGHTGLUE_SRC=<site-packages>/lightglue" >&2; exit 1
fi
for f in aliked-n16.pth aliked_lightglue_v0-1_arxiv.pth; do
  [[ -f "${TORCH_CKPT}/${f}" ]] || { echo "ERROR: ${TORCH_CKPT}/${f} missing (run lightglue once, or set TORCH_CKPT)" >&2; exit 1; }
done
mkdir -p "${HERE}/vendor" "${HERE}/torchhub/hub/checkpoints"
rsync -a "${LIGHTGLUE_SRC}/" "${HERE}/vendor/lightglue/"
cp -f "${TORCH_CKPT}/aliked-n16.pth" "${TORCH_CKPT}/aliked_lightglue_v0-1_arxiv.pth" "${HERE}/torchhub/hub/checkpoints/"

echo "== docker build ${IMAGE} =="
BA=()
for v in MAP_FRAMES MAP_MIN_SPAN MAP_ADAPT MAP_FRAMES_LONG MAP_SPREAD_LO MAP_SPREAD_HI MAP_LONG_BANDS MAP_MIN_SPAN_LONG MAP_END_W MAP_START_W MAP_DROP MAP_DROP_MIN MAP_DROP_RATIO MAP_DROP_K MAP_DROP_K2_MIN MAP_DROP_GATE_MAX MODEL LORA BACKEND FSCALE PI3_PIXEL_LIMIT ROT_PI3 ROT_PI3_STEP ROT_PI3_CHUNK ROT_PI3_W ROT_WDA3 PI3_MODEL ROTAVG ROT_STEPS RES PRECISION HALF CHUNK OVERLAP STRIDE PHOTO LORA_FILE; do
  [[ -n "${!v:-}" ]] && BA+=(--build-arg "${v}=${!v}")
done
docker build "${BA[@]}" -t "${IMAGE}" "${HERE}"
docker images --format '   {{.Repository}}:{{.Tag}}  {{.Size}}' | grep "${IMAGE%%:*}" || true
echo "Built ${IMAGE}"
