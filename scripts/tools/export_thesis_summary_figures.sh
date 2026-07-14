#!/usr/bin/env bash
# Export Ch4 summary figures from canonical runs (Docker).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker compose -f "${ROOT}/docker-compose.yml" run --rm \
  -v "${ROOT}:/app" \
  -w /app \
  --entrypoint python \
  simulation \
  scripts/tools/export_thesis_summary_figures.py "$@"
