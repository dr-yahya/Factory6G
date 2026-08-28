#!/usr/bin/env bash
# Export matplotlib engagement figures to thesis/figures/ (Docker).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker compose -f "${ROOT}/docker-compose.yml" run --rm \
  -v "${ROOT}:/app" \
  -w /app \
  --entrypoint bash \
  simulation \
  -lc 'pip install -e . -q && exec python scripts/tools/export_thesis_engagement_figures.py "$@"' bash "$@"
