#!/usr/bin/env bash
# Export all Ch4 + Appendix B figures from thesis/figure_manifest.json (Docker).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker compose -f "${ROOT}/docker-compose.yml" run --rm \
  -v "${ROOT}:/app" \
  -w /app \
  --entrypoint python \
  simulation \
  scripts/tools/export_thesis_all_figures.py "$@"
