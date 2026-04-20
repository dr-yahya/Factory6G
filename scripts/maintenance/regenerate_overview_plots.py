#!/usr/bin/env python3
"""Regenerate overview/ plots for all existing multi-combo run directories.

Reads stage_results_v2.json files from existing results and calls
write_overview_plots() to produce overview/{stage}/*.png at the run root.

Run from the project root (inside Docker):
    python3 scripts/regenerate_overview_plots.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")

from src.sim.output import write_overview_plots

results_dir = project_root / "results"
run_dirs = sorted(
    d for d in results_dir.iterdir()
    if d.is_dir() and d.name != "archieve"
)

if not run_dirs:
    print("No run directories found under results/")
    sys.exit(0)

processed = 0
skipped = 0

for run_dir in run_dirs:
    json_files = sorted(run_dir.rglob("stage_results_v2.json"))
    if not json_files:
        skipped += 1
        continue

    # Group by stage; detect combo label from path relative to run_dir
    # Structure: run_dir / [combo_parts...] / stage_name / stage_results_v2.json
    by_stage: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for json_path in json_files:
        rel_parts = json_path.relative_to(run_dir).parts
        # rel_parts[-1] = "stage_results_v2.json"
        # rel_parts[-2] = stage_name
        # rel_parts[:-2] = combo path components
        stage_name = rel_parts[-2]
        combo_label = "/".join(rel_parts[:-2])  # empty string for single-combo runs
        by_stage[stage_name].append((combo_label, json_path))

    # Skip single-combo runs (no overview needed)
    all_single = all(
        label == ""
        for entries in by_stage.values()
        for label, _ in entries
    )
    if all_single:
        skipped += 1
        continue

    # Build entries list for write_overview_plots
    entries = []
    for stage_name, stage_entries in by_stage.items():
        payloads = [(cl, json.loads(jp.read_text(encoding="utf-8"))) for cl, jp in stage_entries]
        # Only include stages where all combos share the same ebno_range
        ranges = [p["ebno_db_range"] for _, p in payloads]
        if len({tuple(r) for r in ranges}) > 1:
            print(f"  [skip] {run_dir.name}/{stage_name}: mismatched ebno_db_range across combos")
            continue
        for combo_label, payload in payloads:
            entries.append((
                combo_label,
                stage_name,
                payload["methods"],
                payload["ebno_db_range"],
                payload.get("runtime_totals_sec", {}),
            ))

    # Generic title prefix for retroactive generation
    title_prefix = "All Results"

    print(f"\n{run_dir.name}")
    write_overview_plots(run_dir, entries, title_prefix=title_prefix)
    processed += 1

print(f"\nDone. {processed} run(s) updated, {skipped} skipped (single-combo or empty).")
