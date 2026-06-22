# CLiMB 2026 final method write-up (team Jmees26)

- `climb2026_jmees26.tex` — write-up source. Body occupies exactly 3 pages; the references
  sit alone on page 4 (the limit is 3 pages *excluding* references).
- `reference.bib` — bibliography, pulled in with `bibtex` (`unsrt`, so entries are numbered
  in citation order). Every `\cite` key must exist here.
- `build.sh` — builds the PDF with the local `texlive/texlive:latest-small` docker image
  (this host has no LaTeX). Runs pdflatex, bibtex, pdflatex, pdflatex as the calling user.
  Notes: `texlive-small` has no `titlesec` and no Courier (`pcrr7t`), so the preamble
  redefines section spacing manually and sets `\ttdefault` to `cmtt`.

Submission requirement (`reference/submission_instructions/README.md`, §Final submission):
PDF (max 3 pages excluding references) by **2026-09-12 23:59 AoE** (= 09-13 20:59 JST) to
`endocartoscope@unizar.es`, subject `[CLiMB 2026] Write-up Jmees26`, stating the
Synapse submission ID **9780121** (v099, `climb_score` 4.078, rank 2).
Publishing the source repository is encouraged, not required; ours is at
https://github.com/JmeesInc/CLiMB26-jmees.

No open placeholders remain.
