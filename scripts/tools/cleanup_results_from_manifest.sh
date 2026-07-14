#!/usr/bin/env bash
# Delete incomplete/superseded results/ dirs after successful export (Docker).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker compose -f "${ROOT}/docker-compose.yml" run --rm \
  -v "${ROOT}:/app" \
  -w /app \
  --entrypoint python \
  simulation \
  scripts/tools/cleanup_results_from_manifest.py "$@"
