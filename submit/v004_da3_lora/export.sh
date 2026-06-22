#!/usr/bin/env bash
# Prepare the image for submission.
#
# CLiMB submits by PUSHING to your own Synapse project's Docker registry
# (reference/submission_instructions/README.md §1) -- the tarball is only for
# transfer/backup. The push itself is left to the user (credentials + the
# Synapse submission UI are manual steps).
#
#   docker tag climb-da3-lora:v004 docker.synapse.org/syn<YOUR_PROJECT_ID>/climb-da3:v004
#   docker login docker.synapse.org        # password = Synapse Personal Access Token
#   docker push docker.synapse.org/syn<YOUR_PROJECT_ID>/climb-da3:v004
#
# Then submit that Docker entity to "CLiMB Validation" first, and to
# "CLiMB Scoring" once it comes back VALIDATED.
#
# TAR=1 also writes a gzipped tarball (slow: ~18 GB image).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-climb-da3-lora:v004}"

docker image inspect "${IMAGE}" >/dev/null || { echo "build it first: bash build.sh" >&2; exit 1; }
echo "== ${IMAGE} =="
docker images --format '   {{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedSince}}' \
  | grep "^   ${IMAGE%%:*}" || true

if [[ -n "${SYNAPSE_PROJECT_ID:-}" ]]; then
  REMOTE="docker.synapse.org/${SYNAPSE_PROJECT_ID}/climb-da3:v004"
  echo "== tag -> ${REMOTE} =="
  docker tag "${IMAGE}" "${REMOTE}"
  echo "   tagged. Now run (manually):"
  echo "     docker login docker.synapse.org && docker push ${REMOTE}"
else
  echo "   set SYNAPSE_PROJECT_ID=syn######## to also tag it for the registry."
fi

if [[ "${TAR:-0}" == "1" ]]; then
  OUT="${OUT:-${HERE}/output/$(echo "${IMAGE}" | tr ':/' '__').tar.gz}"
  mkdir -p "$(dirname "${OUT}")"
  echo "== docker save -> ${OUT} =="
  docker save "${IMAGE}" | gzip -1 > "${OUT}"
  ls -lh "${OUT}"
fi
