"""Thesis summary plots and tables sourced from canonical stage_results_v2.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from factory6g.visualization.thesis_plot_style import (
    FIG_CALLOUT_PT,
    FIG_TITLE_PT,
    THESIS_DPI,
    apply_thesis_rcparams,
)

SUNWAY_GREY = "#64748b"
POINT_STATUS_UPPER_BOUND_ONLY = "upper_bound_only"

CANONICAL_SCHEDULERS: tuple[str, ...] = (
    "static",
    "round_robin",
    "max_throughput",
    "pf",
    "wmmse",
    "queue_aware",
    "drl",
    "reliability_drl",
)

RM_LABELS: dict[str, str] = {
    "static": "Static",
    "round_robin": "Round-robin",
    "max_throughput": "Max-throughput",
    "pf": "PF",
    "wmmse": "WMMSE",
    "queue_aware": "Queue-aware",
    "drl": "DRL",
    "reliability_drl": "Reliability-DRL",
}

CHANNEL_LABELS: dict[str, str] = {
    "rayleigh": "Rayleigh",
    "rician": "Rician",
    "tr38901": "TR~38.901 UMi",
}


def load_stage_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ebno_index(payload: dict[str, Any], ebno_db: float) -> int:
    return payload["ebno_db_range"].index(float(ebno_db))


def _effective_ber(metric_map: dict[str, list[Any]], index: int) -> tuple[float, str]:
    ber = float(metric_map["ber"][index])
    status = str(metric_map["point_status"][index])
    if ber > 0:
        return ber, status
    upper = float(metric_map.get("ber_upper_confidence", [ber])[index])
    return upper, status


def plot_ch04_ber_heatmap(
    *,
    run_b_dir: Path,
    ebno_db: float,
    output_path: Path,
) -> None:
    """Scheduler × channel BER summary at one anchor operating point (Run~B)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    apply_thesis_rcparams(plt)
    channels = ["rayleigh", "rician", "tr38901"]
    schedulers = list(CANONICAL_SCHEDULERS)

    matrix = np.full((len(schedulers), len(channels)), np.nan)
    statuses = [["" for _ in channels] for _ in schedulers]

    for col, channel in enumerate(channels):
        payload = load_stage_json(run_b_dir / channel / "resource_managers/stage_results_v2.json")
        idx = _ebno_index(payload, ebno_db)
        for row, scheduler in enumerate(schedulers):
            metric_map = payload["methods"][scheduler]
            ber, status = _effective_ber(metric_map, idx)
            matrix[row, col] = max(ber, 1e-12)
            statuses[row][col] = status

    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    im = ax.imshow(matrix, cmap="YlOrRd", norm=LogNorm(vmin=1e-5, vmax=1e-2), aspect="auto")

    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels([CHANNEL_LABELS[ch] for ch in channels])
    ax.set_yticks(range(len(schedulers)))
    ax.set_yticklabels([RM_LABELS[s] for s in schedulers])
    ax.set_xlabel("Channel model (Run~B)")
    ax.set_ylabel("Resource manager")
    ax.set_title(
        rf"BER summary at $E_b/N_0 = {ebno_db:.0f}\,\mathrm{{dB}}$"
        + "\n"
        + "(adaptive-estimator feedback)",
        fontsize=FIG_TITLE_PT,
    )

    for row in range(len(schedulers)):
        for col in range(len(channels)):
            value = matrix[row, col]
            is_upper = statuses[row][col] == POINT_STATUS_UPPER_BOUND_ONLY
            if value >= 1e-3:
                value_fmt = f"{value:.2e}"
            else:
                value_fmt = f"{value:.1e}"
            if is_upper:
                text = rf"${value_fmt}^{{\dagger}}$"
            else:
                text = rf"${value_fmt}$"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=FIG_CALLOUT_PT,
                color="black" if value < 5e-4 else "white",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("BER (colour scale)")
    fig.subplots_adjust(bottom=0.26)
    fig.text(
        0.08,
        0.055,
        r"$^\dagger$Upper-bound-only (95\% confidence bound",
        ha="left",
        va="bottom",
        fontsize=FIG_CALLOUT_PT,
        color=SUNWAY_GREY,
    )
    fig.text(
        0.08,
        0.02,
        "when zero errors observed)",
        ha="left",
        va="bottom",
        fontsize=FIG_CALLOUT_PT,
        color=SUNWAY_GREY,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=THESIS_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {output_path}")
