from __future__ import annotations

from datetime import datetime
from pathlib import Path


def generate_run_id(now: datetime | None = None) -> str:
    ts = now or datetime.now()
    return ts.strftime("%Y%m%d_%H%M%S")


def build_run_dir(output_dir: str | Path, run_id: str) -> Path:
    return Path(output_dir) / f"{run_id}_simulation"


def create_run_context(
    output_dir: str | Path,
    run_id: str | None = None,
) -> tuple[str, Path]:
    resolved_run_id = run_id or generate_run_id()
    run_dir = build_run_dir(output_dir, resolved_run_id)
    return resolved_run_id, run_dir
