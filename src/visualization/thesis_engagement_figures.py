"""Conceptual / motivation thesis figures (matplotlib, no simulation data)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from src.visualization.thesis_plot_style import (
    FIG_CALLOUT_PT,
    FIG_NOTE_MAX_CHARS,
    FIG_TITLE_PT,
    PALETTE,
    TABLE_FONT_PT,
    THESIS_DPI,
    THESIS_FIGSIZE,
    apply_thesis_rcparams,
    wrap_figure_text,
)

SUNWAY_NAVY = "#233369"
SUNWAY_GREY = "#64748b"


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=THESIS_DPI, bbox_inches="tight")
    print(f"Wrote {path}")


def plot_urllc_tradeoff(*, output_path: Path) -> None:
    """Reliability--latency design space with throughput as marker scale (motivation).

    Layout is spaced so callouts never overlap the markers or each other: the two
    top service classes are labelled above the axes, the two lower ones below their
    markers, the band label sits in clear space inside the band, and the
    marker-size note sits in the empty mid-field.
    """
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    # Illustrative service-class anchors (not measured results).
    services = {
        "Factory URLLC\n(closed-loop control)": {
            "latency_ms": 1.0,
            "reliability_pct": 99.999,
            "throughput": 80,
            "xytext": (-10, 24),
            "ha": "center",
            "va": "bottom",
        },
        "URLLC (general)": {
            "latency_ms": 5.0,
            "reliability_pct": 99.99,
            "throughput": 120,
            "xytext": (48, -30),
            "ha": "left",
            "va": "center",
        },
        "eMBB": {
            "latency_ms": 40.0,
            "reliability_pct": 99.0,
            "throughput": 400,
            "xytext": (0, -22),
            "ha": "center",
            "va": "top",
        },
        "mMTC": {
            "latency_ms": 500.0,
            "reliability_pct": 99.0,
            "throughput": 60,
            "xytext": (0, -22),
            "ha": "center",
            "va": "top",
        },
    }

    ax.axvspan(0.3, 10.0, ymin=0.55, ymax=1.0, color=SUNWAY_NAVY, alpha=0.08, zorder=0)
    ax.text(
        0.42,
        99.60,
        "Factory URLLC\ntarget band",
        fontsize=FIG_CALLOUT_PT - 2,
        color=SUNWAY_NAVY,
        ha="left",
        va="center",
        linespacing=1.2,
        zorder=2,
    )

    for index, (label, spec) in enumerate(services.items()):
        color = PALETTE[index % len(PALETTE)]
        size = 40 + spec["throughput"] * 0.35
        marker_radius_pt = float(np.sqrt(size / np.pi))
        ax.scatter(
            spec["latency_ms"],
            spec["reliability_pct"],
            s=size,
            c=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
            label=label.replace("\n", " "),
        )
        ax.annotate(
            label,
            xy=(spec["latency_ms"], spec["reliability_pct"]),
            xytext=spec["xytext"],
            textcoords="offset points",
            fontsize=FIG_CALLOUT_PT - 1,
            ha=spec.get("ha", "left"),
            va=spec.get("va", "center"),
            color=color,
            linespacing=1.2,
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=0.9,
                shrinkA=2,
                shrinkB=max(2.0, marker_radius_pt * 0.35),
                relpos=(0.5, 0.5),
            ),
            zorder=4,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.4, 900)
    ax.set_ylim(98.55, 100.05)
    ax.set_xlabel("End-to-end latency (ms)")
    ax.set_ylabel("Reliability (%)")
    ax.grid(True, which="both", alpha=0.35)
    ax.annotate(
        "Marker area $\\propto$ throughput demand",
        xy=(0.60, 0.60),
        xycoords="axes fraction",
        fontsize=FIG_CALLOUT_PT - 2,
        color=SUNWAY_GREY,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=SUNWAY_GREY, alpha=0.9),
        zorder=5,
    )

    fig.tight_layout()
    _save(fig, output_path)
    plt.close(fig)


def plot_fading_scenarios(*, output_path: Path) -> None:
    """Rayleigh vs Rician fading and TR~38.901 UMi path-loss intuition (P4)."""
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    fig, axes = plt.subplots(1, 3, figsize=(5.8, 2.6))
    envelope_ylabel = wrap_figure_text("Envelope PDF (probability density)")

    # (a) Rayleigh channel magnitude |h|
    ax = axes[0]
    scale = 1.0 / np.sqrt(2.0)
    x = np.linspace(0.0, 3.0, 400)
    ax.plot(x, stats.rayleigh.pdf(x, scale=scale), color=PALETTE[0], lw=2.0)
    ax.set_xlabel(r"Channel magnitude $|h|$")
    ax.set_ylabel(envelope_ylabel)
    ax.set_title("Rayleigh", fontsize=TABLE_FONT_PT)
    ax.text(0.97, 0.95, "No dominant LOS", transform=ax.transAxes, ha="right", va="top", fontsize=FIG_CALLOUT_PT, color=SUNWAY_GREY)

    # (b) Rician with K = 6 dB
    ax = axes[1]
    k_db = 6.0
    nu = np.sqrt(2.0 * 10 ** (k_db / 10.0))
    x = np.linspace(0.0, 3.5, 400)
    ax.plot(x, stats.rice.pdf(x, nu / scale, scale=scale), color=PALETTE[2], lw=2.0)
    ax.set_xlabel(r"Channel magnitude $|h|$")
    ax.set_ylabel(envelope_ylabel)
    ax.set_title(r"Rician ($K = 6$\,dB)", fontsize=TABLE_FONT_PT)
    ax.text(0.97, 0.95, "LOS + scatter", transform=ax.transAxes, ha="right", va="top", fontsize=FIG_CALLOUT_PT, color=SUNWAY_GREY)

    # (c) TR 38.901 UMi path loss at 3.5 GHz (conceptual, 2--D distance)
    ax = axes[2]
    f_ghz = 3.5
    d_m = np.linspace(10.0, 500.0, 400)
    pl_los = 32.4 + 21.0 * np.log10(d_m) + 20.0 * np.log10(f_ghz)
    pl_nlos = 35.3 + 22.4 * np.log10(d_m) + 21.3 * np.log10(f_ghz)
    pl_nlos = np.maximum(pl_los, pl_nlos)
    ax.plot(d_m, pl_los, color=PALETTE[1], lw=2.0, label="UMi LOS")
    ax.plot(d_m, pl_nlos, color=PALETTE[3], lw=2.0, linestyle="--", label="UMi NLOS")
    ax.set_xlabel("2--D distance $d$ (m)")
    ax.set_ylabel("Path loss (dB)")
    ax.set_title(r"TR~38.901 UMi @ 3.5\,GHz", fontsize=TABLE_FONT_PT)
    ax.legend(fontsize=FIG_CALLOUT_PT, loc="lower right")
    ax.text(0.03, 0.95, "Standards geometry", transform=ax.transAxes, ha="left", va="top", fontsize=FIG_CALLOUT_PT, color=SUNWAY_GREY)

    fig.tight_layout()
    _save(fig, output_path)
    plt.close(fig)


def _draw_mini_flow(
    ax: Any,
    *,
    x0: float,
    y0: float,
    width: float,
    labels: list[str],
    colors: list[str],
) -> None:
    """Draw a short left-to-right flow of rounded boxes with arrows."""
    import matplotlib.patches as mpatches

    n = len(labels)
    gap = 0.04
    box_w = (width - gap * (n - 1)) / n
    box_h = 0.11
    for index, (label, color) in enumerate(zip(labels, colors)):
        x = x0 + index * (box_w + gap)
        rect = mpatches.FancyBboxPatch(
            (x, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.0,
            edgecolor=color,
            facecolor=color,
            alpha=0.18,
            transform=ax.transAxes,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + box_w / 2,
            y0 + box_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=FIG_CALLOUT_PT,
            transform=ax.transAxes,
            zorder=3,
        )
        if index < n - 1:
            ax.annotate(
                "",
                xy=(x + box_w + gap * 0.2, y0 + box_h / 2),
                xytext=(x + box_w + gap * 0.8, y0 + box_h / 2),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color=SUNWAY_GREY, lw=1.2),
                zorder=1,
            )


def plot_lr_eval_coupling(*, output_path: Path) -> None:
    """Literature-facing CE/RM evaluation coupling patterns (Ch2 §2.7)."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    apply_thesis_rcparams(plt)
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "Literature evaluation coupling patterns",
        ha="center",
        va="top",
        fontsize=FIG_TITLE_PT,
        fontweight="bold",
        color=SUNWAY_NAVY,
    )
    ax.text(
        0.5,
        0.90,
        wrap_figure_text(
            "How prior channel-estimation and resource-management studies are typically benchmarked",
            max_chars=FIG_NOTE_MAX_CHARS,
        ),
        ha="center",
        va="top",
        fontsize=FIG_CALLOUT_PT,
        color=SUNWAY_GREY,
    )

    columns = [
        {
            "title": "PHY-only CE studies",
            "subtitle": "Neural / classical estimator surveys",
            "x": 0.04,
            "w": 0.28,
            "flow": ["Channel\npilots", "CE\nalgorithm", "BER /\nMSE"],
            "colors": [PALETTE[0], PALETTE[1], PALETTE[2]],
            "footer": "No MAC masks or scheduling feedback",
            "border": PALETTE[0],
        },
        {
            "title": "MAC-only RM studies",
            "subtitle": "Heuristic and DRL scheduling",
            "x": 0.36,
            "w": 0.28,
            "flow": ["Channel\nmodel", "Scheduler\n(perfect CSI)", "Throughput /\nfairness"],
            "colors": [PALETTE[3], PALETTE[4], PALETTE[5]],
            "footer": "No $\\hat{\\mathbf{H}}$ error propagation",
            "border": PALETTE[3],
        },
        {
            "title": "Integrated (sparse)",
            "subtitle": "Few factory-grounded joint reports",
            "x": 0.68,
            "w": 0.28,
            "flow": ["Shared\ncontext", "CE $\\rightarrow$ RM", "BER +\nthroughput"],
            "colors": [PALETTE[6], PALETTE[7], PALETTE[2]],
            "footer": "Shared realisations + estimated CSI",
            "border": PALETTE[6],
            "integrated": True,
        },
    ]

    for column in columns:
        integrated = column.get("integrated", False)
        frame = mpatches.FancyBboxPatch(
            (column["x"], 0.14),
            column["w"],
            0.68,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=column["border"],
            facecolor=column["border"],
            alpha=0.06,
            linestyle="--" if integrated else "-",
            transform=ax.transAxes,
            zorder=0,
        )
        ax.add_patch(frame)
        ax.text(
            column["x"] + column["w"] / 2,
            0.76,
            column["title"],
            ha="center",
            va="center",
            fontsize=TABLE_FONT_PT,
            fontweight="bold",
            color=SUNWAY_NAVY,
            transform=ax.transAxes,
        )
        ax.text(
            column["x"] + column["w"] / 2,
            0.70,
            wrap_figure_text(column["subtitle"]),
            ha="center",
            va="center",
            fontsize=FIG_CALLOUT_PT,
            color=SUNWAY_GREY,
            transform=ax.transAxes,
        )
        _draw_mini_flow(
            ax,
            x0=column["x"] + 0.02,
            y0=0.42,
            width=column["w"] - 0.04,
            labels=column["flow"],
            colors=column["colors"],
        )
        ax.text(
            column["x"] + column["w"] / 2,
            0.22,
            wrap_figure_text(column["footer"]),
            ha="center",
            va="center",
            fontsize=FIG_CALLOUT_PT,
            color=column["border"],
            transform=ax.transAxes,
        )

    for y in (0.48, 0.52):
        ax.plot([0.32, 0.36], [y, y], color="#be123c", lw=1.0, linestyle="--", transform=ax.transAxes, zorder=1)
    ax.text(0.34, 0.56, "broken\nlink", ha="center", va="bottom", fontsize=FIG_CALLOUT_PT, color="#be123c", transform=ax.transAxes)

    ax.annotate(
        "",
        xy=(0.66, 0.50),
        xytext=(0.64, 0.50),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", color=SUNWAY_GREY, lw=1.0, linestyle="dotted"),
    )
    ax.text(0.65, 0.54, "rare", ha="center", fontsize=FIG_CALLOUT_PT, color=SUNWAY_GREY, transform=ax.transAxes)

    ax.text(
        0.5,
        0.06,
        "Integrated end-to-end evaluation under shared factory geometry remains underrepresented (cf.\\ taxonomy gap)",
        ha="center",
        va="center",
        fontsize=FIG_CALLOUT_PT,
        color=SUNWAY_GREY,
        transform=ax.transAxes,
    )

    fig.tight_layout()
    _save(fig, output_path)
    plt.close(fig)
