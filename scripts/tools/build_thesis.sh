#!/usr/bin/env bash
# Build the thesis PDF inside TeX Live Docker (no host LaTeX required).
# latexmk reads thesis/latexmkrc, so all artifacts land under thesis/build/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THESIS="${ROOT}/thesis"
IMAGE="${THESIS_TEX_IMAGE:-texlive/texlive:latest}"

docker run --rm \
  -v "${THESIS}:/work" \
  -w /work \
  "${IMAGE}" \
  latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex main.tex

mkdir -p "${THESIS}/exports"
cp -f "${THESIS}/build/main.pdf" "${THESIS}/exports/thesis_full.pdf"
echo "Built ${THESIS}/build/main.pdf -> ${THESIS}/exports/thesis_full.pdf"
