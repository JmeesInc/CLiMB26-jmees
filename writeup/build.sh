#!/usr/bin/env bash
# Build the CLiMB 2026 write-up PDF using the local texlive container
# (no LaTeX on this host). Runs as the calling user so the outputs are not root-owned.
# References come from reference.bib via bibtex, so the sequence is
# pdflatex -> bibtex -> pdflatex -> pdflatex.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX="${1:-climb2026_jmees26.tex}"
BASE="${TEX%.tex}"
run() {
  docker run --rm --runtime=runc --user "$(id -u)":"$(id -g)" -e HOME=/tmp \
    -v "${HERE}":/w -w /w texlive/texlive:latest-small "$@" >/dev/null
}
run pdflatex -interaction=nonstopmode -halt-on-error "${TEX}"
run bibtex "${BASE}"
run pdflatex -interaction=nonstopmode -halt-on-error "${TEX}"
run pdflatex -interaction=nonstopmode -halt-on-error "${TEX}"
pdfinfo "${HERE}/${BASE}.pdf" 2>/dev/null | grep -i pages || true
ls -la "${HERE}/${BASE}.pdf"
