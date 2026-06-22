#!/usr/bin/env bash
# Save the built image to a tarball for submission/transfer.
# (The Synapse push instructions are still TBD in the official docs; when the
# queue opens, tag and push instead of shipping this tar.)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-climb-da3-submap:v001}"
OUT="${OUT:-${HERE}/output/$(echo "${IMAGE}" | tr ':/' '__').tar.gz}"

mkdir -p "$(dirname "${OUT}")"
echo "== docker save ${IMAGE} -> ${OUT} =="
docker save "${IMAGE}" | gzip -1 > "${OUT}"
ls -lh "${OUT}"
