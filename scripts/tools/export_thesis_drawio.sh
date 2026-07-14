#!/usr/bin/env bash
# Export thesis/figures/*.drawio to matching PNGs via draw.io desktop CLI.
#
# Setup (once on macOS):
#   brew install --cask drawio
#   drawio --version
#
# Usage:
#   ./scripts/tools/export_thesis_drawio.sh
#   ./scripts/tools/export_thesis_drawio.sh thesis/figures/fig_phy_stack.drawio

# After editing in draw.io desktop or via drawio-skill:
#   Connectors → Arrange → To back (boxes above lines).
#   Methodology flows only: step badges near top-left corner (~6px outside, no overlap);
#   badges above connectors. See thesis/CONTEXT.md → Step-number badges.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
FIGDIR="${ROOT}/thesis/figures"
SCALE="${DRAWIO_EXPORT_SCALE:-2}"
BORDER="${DRAWIO_EXPORT_BORDER:-8}"

if ! command -v drawio >/dev/null 2>&1; then
  if [[ -x "/Applications/draw.io.app/Contents/MacOS/draw.io" ]]; then
    drawio() { "/Applications/draw.io.app/Contents/MacOS/draw.io" "$@"; }
  else
    echo "drawio CLI not found. Install: brew install --cask drawio" >&2
    exit 1
  fi
fi

export_one() {
  local src="$1"
  local dst="${src%.drawio}.png"
  echo "Exporting ${src} -> ${dst}"
  drawio \
    --export \
    --format png \
    --scale "${SCALE}" \
    --border "${BORDER}" \
    --output "${dst}" \
    "${src}"
}

if [[ $# -gt 0 ]]; then
  for f in "$@"; do
    [[ -f "$f" ]] || { echo "Not found: $f" >&2; exit 1; }
    export_one "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
  done
else
  shopt -s nullglob
  files=("${FIGDIR}"/*.drawio)
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No .drawio files in ${FIGDIR}" >&2
    exit 1
  fi
  for f in "${files[@]}"; do
    export_one "$f"
  done
fi

echo "Done."
echo "Pre-export checklist:"
echo "  • Connectors behind boxes (Arrange → To back in draw.io desktop)"
echo "  • Step badges (methodology flows only): near top-left corner (~6px outside), no box overlap, above connectors"
echo "  See thesis/CONTEXT.md → Step-number badges, Manual draw.io z-order"
