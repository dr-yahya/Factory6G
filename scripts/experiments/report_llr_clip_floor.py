"""Turn an llr_clip_floor.py run into a thesis-ready evidence note and figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _format_ber(errors: int, bits: int) -> str:
    """Show a measured BER, or a one-sided bound when nothing was observed."""
    if errors == 0:
        from factory6g.sim.stages.common import zero_error_upper_bound

        return f"< {zero_error_upper_bound(bits, 0.95):.2e}"
    return f"{errors / max(bits, 1):.3e}"


def _floor_summary(ebno: list[float], ber: list[float]) -> dict:
    """Detect a high-SNR floor: where the waterfall stops falling."""
    arr = np.asarray(ber, dtype=float)
    valid = arr > 0
    # Slope over the top half of the sweep, in decades per dB.
    half = len(ebno) // 2
    top_ebno = np.asarray(ebno[half:], dtype=float)
    top_ber = arr[half:]
    usable = top_ber > 0
    if usable.sum() < 2:
        # No errors at all in the tail is the strongest possible evidence of no
        # floor -- the link is error-free to the limit of the evidence.
        return {
            "floored": False,
            "reason": "no errors observed in the tail (error-free to the evidence limit)",
        }
    if valid.sum() < 3:
        return {"floored": False, "reason": "not enough non-zero points to fit a slope"}
    slope = float(
        np.polyfit(top_ebno[usable], np.log10(top_ber[usable]), 1)[0]
    )
    return {
        "floored": bool(slope > -0.05),
        "tail_slope_decades_per_db": slope,
        "final_ber": float(arr[-1]),
    }


def build_report(payload: dict) -> str:
    ebno = payload["ebno_db"]
    results = payload["results"]
    labels = list(results)

    lines: list[str] = []
    lines.append("# LLR Clipping And The High-SNR BER Floor")
    lines.append("")
    lines.append(
        f"Channel `{payload['channel']}`, estimator `{payload['estimator']}`, "
        f"{payload['batches_per_point']} batches x {payload['batch_size']} per Eb/No point "
        f"({payload['batches_per_point'] * payload['batch_size'] * payload['config']['num_ut']:,} "
        f"codewords per point)."
    )
    lines.append("")
    lines.append("## Why this experiment exists")
    lines.append("")
    lines.append(
        "The receiver clipped demapper LLRs hard at +/-20, justified by a diagnostic "
        "noting that 27.5% of LLRs exceeded 50 in magnitude. At high Eb/No a large "
        "share of LLRs are legitimately large; saturating them discards the "
        "reliability information belief propagation needs to correct the remaining "
        "weak bits, precisely where the waterfall should be steepest."
    )
    lines.append("")
    lines.append(
        "Every clip setting below decodes the *same* channel realisations, noise "
        "draws and source bits (common random numbers), so the differences are not "
        "contaminated by Monte Carlo variance and the paired intervals are exact."
    )
    lines.append("")

    lines.append("## Measured BER")
    lines.append("")
    lines.append("| Eb/No (dB) | " + " | ".join(f"clip {l}" for l in labels) + " |")
    lines.append("|---" * (len(labels) + 1) + "|")
    for index, value in enumerate(ebno):
        row = [f"{value:.0f}"]
        for label in labels:
            data = results[label]
            row.append(_format_ber(data["bit_errors"][index], data["total_bits"][index]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Tail behaviour")
    lines.append("")
    lines.append("| Clip | Tail slope (decades/dB) | Floored? | Basis |")
    lines.append("|---|---|---|---|")
    summaries = {}
    for label in labels:
        summary = _floor_summary(ebno, results[label]["ber"])
        summaries[label] = summary
        slope = summary.get("tail_slope_decades_per_db")
        slope_text = f"{slope:+.3f}" if slope is not None else "—"
        lines.append(
            f"| {label} | {slope_text} | "
            f"{'**yes**' if summary.get('floored') else 'no'} | "
            f"{summary.get('reason', 'fitted over the upper half of the sweep')} |"
        )
    lines.append("")
    lines.append(
        "A working link's BER falls steadily with Eb/No, so a tail slope near zero "
        "means the curve has stopped responding to SNR -- an error floor."
    )
    lines.append("")

    lines.append("## Paired difference against the historical +/-20 clip")
    lines.append("")
    lines.append("| Eb/No (dB) | Clip | Mean BER delta | 95% CI | Significant |")
    lines.append("|---|---|---|---|---|")
    for label, rows in payload["paired_vs_clip20"].items():
        for row in rows:
            lines.append(
                f"| {row['ebno_db']:.0f} | {label} | {row['mean_ber_delta']:+.3e} | "
                f"[{row['ci_lower']:+.3e}, {row['ci_upper']:+.3e}] | "
                f"{'yes' if row['significant'] else 'no'} |"
            )
    lines.append("")
    lines.append(
        "A negative delta means the wider clip produced *fewer* errors on the same "
        "channel realisation. An interval excluding zero is a statistically "
        "significant difference."
    )
    lines.append("")

    return "\n".join(lines)


def build_figure(payload: dict, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from factory6g.visualization.thesis_plot_style import (
        THESIS_DPI,
        THESIS_FIGSIZE,
        apply_thesis_rcparams,
        style_ebno_axis,
    )

    apply_thesis_rcparams(plt)
    fig, ax = plt.subplots(figsize=THESIS_FIGSIZE)
    ebno = np.asarray(payload["ebno_db"], dtype=float)
    styles = {"20": ("o-", "#c0392b"), "200": ("s-", "#2980b9"), "none": ("^--", "#27ae60")}
    for label, data in payload["results"].items():
        ber = np.asarray(data["ber"], dtype=float)
        ber = np.where(ber > 0, ber, np.nan)
        marker, color = styles.get(label, ("o-", None))
        ax.semilogy(ebno, ber, marker, color=color, label=f"LLR clip {label}", linewidth=1.8)
    ax.legend()
    style_ebno_axis(
        ax,
        ylabel="BER",
        title=f"LLR clip vs BER ({payload['channel']}, {payload['estimator']})",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=THESIS_DPI)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", help="JSON produced by llr_clip_floor.py")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    payload = json.loads(Path(args.payload).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    channel = payload["channel"]
    report_path = output_dir / f"llr_clip_floor_{channel}.md"
    figure_path = output_dir / f"llr_clip_floor_{channel}.png"
    report_path.write_text(build_report(payload))
    build_figure(payload, figure_path)
    print(f"Wrote {report_path}")
    print(f"Wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
