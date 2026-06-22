#!/usr/bin/env bash
# Custom ORB-SLAM3 entrypoint for LOCAL simulated-sequence evaluation.
# Identical to the shipped entrypoint.sh but uses a mounted PinHole settings
# file (/opt/sim.yaml) instead of the hardcoded HCULB Kannala-Brandt yaml.
set -euo pipefail
cd /opt/ORB_SLAM3

input_dir="${1:-/input}"
pathResults="${2:-/output}"
executions="${EXECUTIONS:-5}"
settings="${SETTINGS:-/opt/sim.yaml}"

mkdir -p "${pathResults}"
shopt -s nullglob nocaseglob
Sequences=("${input_dir}"/*.mp4)
shopt -u nocaseglob
if [[ "${#Sequences[@]}" -eq 0 ]]; then
    echo "No .mp4 files found in ${input_dir}" 1>&2; exit 1
fi

for pathVideo in "${Sequences[@]}"; do
    name_seq="$(basename "${pathVideo%.*}")"
    pathResultSeq="${pathResults}/${name_seq}"
    mkdir -p "${pathResultSeq}"
    for i in $(seq 1 "${executions}"); do
        echo "Running ${name_seq} --> ${i}/${executions} (settings=${settings})"
        if ! (
            set -e
            rm -rf output
            ./Examples/Monocular/mono_endo_hculb \
                ./Vocabulary/ORBvoc.txt "${settings}" "${pathVideo}"
            pathResultExp="${pathResultSeq}/${i}"
            rm -rf "${pathResultExp}"; mkdir -p "${pathResultExp}"
            shopt -s nullglob; mv output/* "${pathResultExp}"; shopt -u nullglob
        ); then
            echo "[WARN] ${name_seq} execution ${i} failed; continuing" >&2
        fi
    done
done
echo "Done."
