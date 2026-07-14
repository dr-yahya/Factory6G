#!/usr/bin/env bash
# Count body-chapter words via texcount in TeX Live Docker (see thesis/CONTEXT.md Word count tracking).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THESIS="${ROOT}/thesis"
IMAGE="${THESIS_TEX_IMAGE:-texlive/texlive:latest}"
BUDGET="${THESIS}/word_budget.csv"

CHAPTERS=(
  chapters/ch01_introduction.tex
  chapters/ch02_literature_review.tex
  chapters/ch03_methodology.tex
  chapters/ch04_results.tex
  chapters/ch05_discussion.tex
  chapters/ch06_conclusions.tex
)

echo "=== Factory6G thesis body word count (texcount) ==="
echo "Scope: Ch1--Ch6 prose (see thesis/CONTEXT.md Word budget)"
echo

docker run --rm \
  -v "${THESIS}:/work" \
  -w /work \
  "${IMAGE}" \
  texcount -inc "${CHAPTERS[@]}"

echo
TOTAL="$(docker run --rm \
  -v "${THESIS}:/work" \
  -w /work \
  "${IMAGE}" \
  texcount -1 -sum -inc "${CHAPTERS[@]}")"

echo "Body total (texcount): ${TOTAL} words"
echo "Planning target: 90000 | Band: 80000--100000 | Pause new prose at ~95000"
echo

if [[ -f "${BUDGET}" ]]; then
  echo "Per-chapter targets (thesis/word_budget.csv):"
  awk -F, 'NR > 1 && $1 != "body_total" && $1 != "non_body" {
    printf "  %-4s %-32s target %6s\n", $1, $2, $3
  }' "${BUDGET}"
fi
