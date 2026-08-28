#!/usr/bin/env python3
"""Export conceptual / motivation thesis figures (matplotlib).

Example (inside Docker):
    python scripts/tools/export_thesis_engagement_figures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")

from factory6g.visualization.thesis_engagement_figures import (
    plot_fading_scenarios,
    plot_urllc_tradeoff,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export thesis engagement figures")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root / "thesis" / "figures",
        help="Output directory (default: thesis/figures)",
    )
    args = parser.parse_args()

    out = args.out_dir
    plot_urllc_tradeoff(output_path=out / "fig_urllc_tradeoff.png")
    plot_fading_scenarios(output_path=out / "fig_fading_scenarios.png")
    # fig_lr_eval_coupling: draw.io source — export via export_thesis_drawio.sh
    print(f"Engagement figures exported to {out}")


if __name__ == "__main__":
    main()
