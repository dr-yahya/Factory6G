#!/usr/bin/env python3
"""Export Ch4 summary figures from canonical locked runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")

from factory6g.visualization.thesis_summary_figures import plot_ch04_ber_heatmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Export thesis Ch4 summary figures")
    parser.add_argument(
        "--canonical",
        type=Path,
        default=project_root / "thesis" / "notes" / "planning" / "canonical_runs.json",
        help="Path to thesis/notes/planning/canonical_runs.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "thesis" / "figures",
        help="Output directory",
    )
    parser.add_argument("--ebno-db", type=float, default=0.0, help="Anchor Eb/N0 for heatmap")
    args = parser.parse_args()

    mapping = json.loads(args.canonical.read_text(encoding="utf-8"))
    run_b_dir = project_root / mapping["resource_managers_run"]["dir"]
    plot_ch04_ber_heatmap(
        run_b_dir=run_b_dir,
        ebno_db=args.ebno_db,
        output_path=args.out_dir / "fig_ch04_ber_heatmap.png",
    )


if __name__ == "__main__":
    main()
