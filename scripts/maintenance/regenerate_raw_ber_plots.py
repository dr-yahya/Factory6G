#!/usr/bin/env python3
"""Regenerate ber_raw_vs_ebno.png for all existing stage results.

Run from the project root:
    python3 scripts/regenerate_raw_ber_plots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.sim.output import _plot_ber_raw

results_dir = project_root / "results"
json_files = sorted(results_dir.rglob("stage_results_v2.json"))

if not json_files:
    print("No stage_results_v2.json files found under results/")
    sys.exit(0)

print(f"Found {len(json_files)} result file(s). Regenerating raw BER plots...\n")

for json_path in json_files:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    stage_name = payload.get("stage", "").replace("_", " ").title()
    out_path = json_path.parent / "ber_raw_vs_ebno.png"
    _plot_ber_raw(
        plt=plt,
        methods=payload["methods"],
        ebno_range=payload["ebno_db_range"],
        title=f"{stage_name}: Raw BER vs Eb/No",
        output_path=out_path,
    )
    print(f"  Updated: {out_path.relative_to(project_root)}")

print("\nDone.")
