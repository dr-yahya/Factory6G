#!/usr/bin/env bash
# Build thesis/main.pdf inside TeX Live Docker (no host LaTeX required).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${THESIS_TEX_IMAGE:-texlive/texlive:latest}"

docker run --rm \
  -v "${ROOT}/thesis:/work" \
  -w /work \
  "${IMAGE}" \
  latexmk -synctex=1 -interaction=nonstopmode -file-line-error -xelatex main.tex

echo "Built ${ROOT}/thesis/main.pdf"
