"""
Compare BER vs Eb/N0: Dr. Athirah JIDD-SCMA vs Our LS (Rayleigh).

Reads saved CSV results and generates a publication-quality semilogy plot.
Run inside Docker:
    docker compose run --rm -v ./scripts:/app/scripts \
        --entrypoint python simulation scripts/plot_comparison.py
"""

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_ber(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read BER vs Eb/N0 from a stage_results_v2.csv file."""
    ebno_list, ber_list = [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["metric"] == "ber" and row["ebno_db"]:
                ebno_list.append(float(row["ebno_db"]))
                ber_list.append(float(row["value"]))
    return np.array(ebno_list), np.array(ber_list)


def main() -> None:
    # Detect base directory: /app inside Docker, project root otherwise
    if Path("/app/results").is_dir():
        base = Path("/app")
    else:
        base = Path(__file__).resolve().parent.parent

    athirah_csv = base / "results/20260320_171006_jidd_scma/jidd_scma/stage_results_v2.csv"
    ls_csv = base / "results/20260318_200508_ls_rayleigh_rician_umi_qpsk/rayleigh/estimators/stage_results_v2.csv"

    ebno_a, ber_a = _read_ber(str(athirah_csv))
    ebno_ls, ber_ls = _read_ber(str(ls_csv))

    # Filter out BER=0 points (can't plot on log scale)
    mask_a = ber_a > 0
    mask_ls = ber_ls > 0

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.semilogy(
        ebno_a[mask_a], ber_a[mask_a],
        marker="o", linewidth=2, markersize=7,
        label="Dr. Athirah — JIDD-SCMA (Rayleigh + MMSE)",
    )
    ax.semilogy(
        ebno_ls[mask_ls], ber_ls[mask_ls],
        marker="s", linewidth=2, markersize=7,
        label="Our LS Estimator (Rayleigh)",
    )

    ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax.set_ylabel("BER", fontsize=12)
    ax.set_title("BER Comparison: JIDD-SCMA vs LS Channel Estimation", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()

    out_path = base / "results" / "comparison_jidd_vs_ls_rayleigh.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot] Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
