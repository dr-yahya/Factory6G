#!/usr/bin/env python3
"""Delete incomplete and superseded results/ directories listed in figure_manifest.json."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]


def _referenced_paths(manifest: dict, root: Path) -> set[Path]:
    refs: set[Path] = set()
    for entry in manifest.get("figures", []):
        for key in ("sources", "run_b_dir"):
            val = entry.get(key)
            if isinstance(val, list):
                for item in val:
                    refs.add((root / item).resolve())
            elif isinstance(val, str):
                refs.add((root / val).resolve())
        for panel in entry.get("panels", []):
            refs.add((root / panel["source"]).resolve())
    for spec in manifest.get("tables", {}).values():
        for row in spec.get("rows", []):
            refs.add((root / row["source"]).resolve())
    return refs


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleanup results/ per figure_manifest.json")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "thesis" / "figure_manifest.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print paths only")
    parser.add_argument("--force", action="store_true", help="Skip reference safety check")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cleanup = manifest.get("cleanup", {})
    referenced = _referenced_paths(manifest, project_root)

    targets: list[Path] = []
    for rel in cleanup.get("incomplete", []):
        targets.append((project_root / rel).resolve())
    for item in cleanup.get("redundant", []):
        targets.append((project_root / item["path"]).resolve())

    for target in targets:
        if not target.exists():
            print(f"SKIP (missing): {target}")
            continue
        if not args.force:
            for ref in referenced:
                try:
                    ref.relative_to(target)
                    print(f"REFUSE (still referenced): {target} <- {ref}")
                    sys.exit(1)
                except ValueError:
                    continue
        if args.dry_run:
            print(f"DRY-RUN delete: {target}")
        else:
            shutil.rmtree(target)
            print(f"Deleted: {target}")

    print("Cleanup complete.")


if __name__ == "__main__":
    main()
