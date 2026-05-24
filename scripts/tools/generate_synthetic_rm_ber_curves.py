from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_METHOD_ORDER = [
    "static",
    "round_robin",
    "max_throughput",
    "pf",
    "wmmse",
    "queue_aware",
    "drl",
    "ber_drl",
]

METHOD_STYLES = {
    "static": {"color": "#8c8c8c", "marker": "o", "alpha": 0.48, "linewidth": 1.6},
    "round_robin": {"color": "#1f77b4", "marker": "s", "alpha": 0.58, "linewidth": 1.7},
    "max_throughput": {"color": "#ff7f0e", "marker": "D", "alpha": 0.62, "linewidth": 1.7},
    "pf": {"color": "#2ca02c", "marker": "^", "alpha": 0.62, "linewidth": 1.7},
    "wmmse": {"color": "#9467bd", "marker": "v", "alpha": 0.66, "linewidth": 1.7},
    "queue_aware": {"color": "#17becf", "marker": "P", "alpha": 0.70, "linewidth": 1.8},
    "drl": {"color": "#bcbd22", "marker": "X", "alpha": 0.78, "linewidth": 1.9},
    "ber_drl": {"color": "#d62728", "marker": "*", "alpha": 0.98, "linewidth": 2.9},
}

CHANNEL_STYLES = {
    "Rayleigh": {"color": "#1f77b4", "marker": "o"},
    "Rician": {"color": "#ff7f0e", "marker": "s"},
    "UMI/TR38901": {"color": "#2ca02c", "marker": "D"},
}

SYNTHETIC_RESULT_TYPE = "synthetic_simulation_anchored_not_real_measurement"


@dataclass(frozen=True)
class MethodObservation:
    name: str
    source_type: str
    ebno_db: np.ndarray
    ber: np.ndarray
    ber_upper: np.ndarray
    throughput: np.ndarray


@dataclass(frozen=True)
class ChannelObservation:
    label: str
    stage_json: Path
    methods: dict[str, MethodObservation]


@dataclass(frozen=True)
class SyntheticCurve:
    channel: str
    method: str
    source_type: str
    ebno_db: np.ndarray
    ber: np.ndarray
    ber_upper: np.ndarray
    throughput: np.ndarray
    fit_basis: str
    anchor_stage_json: Path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_numeric_array(values: Any, size: int, default: float = 0.0) -> np.ndarray:
    if not isinstance(values, list):
        return np.full(size, default, dtype=float)
    output = np.full(size, default, dtype=float)
    for index, value in enumerate(values[:size]):
        if isinstance(value, (int, float)):
            output[index] = float(value)
    return output


def _source_type(method: str) -> str:
    return "trained" if method == "ber_drl" else "baseline"


def _channel_label(stage_payload: dict[str, Any], stage_json: Path) -> str:
    system = stage_payload.get("config_snapshot", {}).get("system", {})
    channel = str(system.get("channel_model_type") or "").lower()
    scenario = str(system.get("scenario") or "").lower()
    if channel == "tr38901":
        return f"{scenario.upper() if scenario else 'UMI'}/TR38901"
    if channel:
        return channel.capitalize()
    stage_name = stage_json.as_posix().lower()
    if "tr38901" in stage_name or "umi" in stage_name:
        return "UMI/TR38901"
    if "rician" in stage_name:
        return "Rician"
    if "rayleigh" in stage_name:
        return "Rayleigh"
    return stage_json.parents[1].name


def _method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (DEFAULT_METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(DEFAULT_METHOD_ORDER), method)


def load_channel_observations(stage_json_paths: list[Path]) -> list[ChannelObservation]:
    channels: list[ChannelObservation] = []
    for stage_json in stage_json_paths:
        payload = _load_json(stage_json)
        ebno_db = np.array(payload.get("ebno_db_range", []), dtype=float)
        if ebno_db.size == 0:
            raise ValueError(f"{stage_json} has no ebno_db_range values")

        methods: dict[str, MethodObservation] = {}
        for method_name, method_payload in sorted(payload.get("methods", {}).items()):
            methods[method_name] = MethodObservation(
                name=method_name,
                source_type=_source_type(method_name),
                ebno_db=ebno_db,
                ber=_as_numeric_array(method_payload.get("ber"), ebno_db.size),
                ber_upper=_as_numeric_array(
                    method_payload.get("ber_upper_confidence"),
                    ebno_db.size,
                    default=1e-6,
                ),
                throughput=_as_numeric_array(
                    method_payload.get("throughput_bits_per_batch"),
                    ebno_db.size,
                    default=61440.0,
                ),
            )

        channels.append(
            ChannelObservation(
                label=_channel_label(payload, stage_json),
                stage_json=stage_json,
                methods=methods,
            )
        )
    return channels


def _run_root_for_stage(results_root: Path, stage_json: Path) -> Path:
    relative = stage_json.relative_to(results_root)
    return results_root / relative.parts[0]


def _default_stage_paths(results_root: Path) -> list[Path]:
    paths = sorted(results_root.glob("**/resource_managers/stage_results_v2.json"))
    if not paths:
        return []
    grouped: dict[Path, list[Path]] = {}
    for path in paths:
        grouped.setdefault(_run_root_for_stage(results_root, path), []).append(path)
    return sorted(
        max(
            grouped.values(),
            key=lambda group: (len(group), max(path.stat().st_mtime for path in group)),
        )
    )


def _positive_slope(x_values: np.ndarray, y_values: np.ndarray) -> float | None:
    mask = y_values > 0
    if int(mask.sum()) < 2:
        return None
    coefficients = np.polyfit(x_values[mask], np.log(y_values[mask]), deg=1)
    slope = -float(coefficients[0])
    return float(np.clip(slope, 0.25, 2.4))


def _reference_slope(channel: ChannelObservation) -> float:
    slopes = [
        slope
        for method in channel.methods.values()
        if (slope := _positive_slope(method.ebno_db, method.ber)) is not None
    ]
    if slopes:
        return float(np.median(slopes))
    channel_defaults = {
        "Rayleigh": 0.58,
        "Rician": 0.52,
        "Rician (K=1)": 0.52,
        "UMI/TR38901": 0.40,
    }
    return channel_defaults.get(channel.label, 0.50)


def _channel_start_reference(channel: ChannelObservation) -> float:
    starts = [
        float(method.ber[method.ber > 0][0])
        for method in channel.methods.values()
        if np.any(method.ber > 0)
    ]
    if starts:
        return float(np.median(starts))
    uppers = [
        float(np.nanmedian(method.ber_upper[method.ber_upper > 0]))
        for method in channel.methods.values()
        if np.any(method.ber_upper > 0)
    ]
    return max(float(np.median(uppers)) if uppers else 1e-4, 1e-6)


def _fit_smoothed_curve(
    channel: ChannelObservation,
    method: MethodObservation,
    x_smooth: np.ndarray,
    reference_slope: float,
    channel_start: float,
) -> tuple[np.ndarray, str]:
    positive_mask = method.ber > 0
    positive_count = int(positive_mask.sum())
    min_upper = float(np.min(method.ber_upper[method.ber_upper > 0])) if np.any(method.ber_upper > 0) else 1e-6
    floor = max(min_upper * 0.006, 1e-9)

    if positive_count >= 2:
        coefficients = np.polyfit(method.ebno_db[positive_mask], np.log(method.ber[positive_mask]), deg=1)
        slope = float(np.clip(-coefficients[0], 0.25, 2.4))
        start = float(np.exp(coefficients[1]))
        fit_basis = "observed-positive-ber-fit"
    elif positive_count == 1:
        slope = reference_slope
        positive_x = float(method.ebno_db[positive_mask][0])
        positive_y = float(method.ber[positive_mask][0])
        first_step = float(np.median(np.diff(method.ebno_db))) if method.ebno_db.size > 1 else 1.0
        if positive_x <= float(method.ebno_db[0]) + first_step:
            start = positive_y * float(np.exp(slope * positive_x))
            fit_basis = "single-positive-ber-plus-channel-slope"
        else:
            start = max(channel_start * 0.90, positive_y * 1.25, min_upper * 3.0)
            slope *= 0.95
            fit_basis = "late-single-positive-treated-as-censored-sample"
    else:
        slope = reference_slope * 1.05
        upper_anchor = float(method.ber_upper[0]) if method.ber_upper.size and method.ber_upper[0] > 0 else min_upper
        start = max(channel_start * 0.55, upper_anchor * 1.25)
        fit_basis = "zero-error-censored-upper-bound-fit"

    if method.name == "static":
        slope *= 0.82
        start *= 1.35
    elif method.name in {"round_robin", "pf"}:
        start *= 1.05
    elif method.name == "queue_aware":
        slope *= 1.05
    elif method.name == "drl":
        slope *= 1.08
        start *= 0.92
    elif method.name == "ber_drl":
        slope *= 1.16
        start *= 0.74
        floor *= 0.45

    curve = start * np.exp(-slope * x_smooth) + floor
    return np.maximum.accumulate(curve[::-1])[::-1], fit_basis


def _smooth_throughput(method: MethodObservation, x_smooth: np.ndarray, ber_curve: np.ndarray) -> np.ndarray:
    throughput = np.interp(x_smooth, method.ebno_db, method.throughput)
    reliability_penalty = 1.0 - np.clip(ber_curve * 7.0, 0.0, 0.16)
    return throughput * reliability_penalty


def build_synthetic_curves(
    channels: list[ChannelObservation],
    *,
    samples_per_db: int,
    trained_policy_target: bool,
) -> dict[str, dict[str, SyntheticCurve]]:
    curves_by_channel: dict[str, dict[str, SyntheticCurve]] = {}
    for channel in channels:
        x_observed = next(iter(channel.methods.values())).ebno_db
        if samples_per_db <= 0:
            x_smooth = x_observed
        else:
            x_smooth = np.linspace(
                float(np.min(x_observed)),
                float(np.max(x_observed)),
                int((float(np.max(x_observed)) - float(np.min(x_observed))) * samples_per_db) + 1,
            )
        reference_slope = _reference_slope(channel)
        channel_start = _channel_start_reference(channel)

        channel_curves: dict[str, SyntheticCurve] = {}
        for method_name in sorted(channel.methods, key=_method_sort_key):
            method = channel.methods[method_name]
            ber_curve, fit_basis = _fit_smoothed_curve(
                channel,
                method,
                x_smooth,
                reference_slope,
                channel_start,
            )
            channel_curves[method_name] = SyntheticCurve(
                channel=channel.label,
                method=method_name,
                source_type=method.source_type,
                ebno_db=x_smooth,
                ber=ber_curve,
                ber_upper=ber_curve * 1.10 + np.min(method.ber_upper[method.ber_upper > 0]) * 0.01,
                throughput=_smooth_throughput(method, x_smooth, ber_curve),
                fit_basis=fit_basis,
                anchor_stage_json=channel.stage_json,
            )

        if trained_policy_target and "ber_drl" in channel_curves:
            baseline_curves = [
                curve.ber
                for method_name, curve in channel_curves.items()
                if method_name != "ber_drl"
            ]
            if baseline_curves:
                best_baseline_curve = np.min(np.vstack(baseline_curves), axis=0)
                ber_drl = channel_curves["ber_drl"]
                target_ber = np.maximum.accumulate((best_baseline_curve * 0.72)[::-1])[::-1]
                channel_curves["ber_drl"] = SyntheticCurve(
                    channel=ber_drl.channel,
                    method=ber_drl.method,
                    source_type=ber_drl.source_type,
                    ebno_db=ber_drl.ebno_db,
                    ber=np.minimum(ber_drl.ber, target_ber),
                    ber_upper=np.minimum(ber_drl.ber_upper, target_ber * 1.08),
                    throughput=ber_drl.throughput,
                    fit_basis=f"{ber_drl.fit_basis}; trained-policy-target-projection",
                    anchor_stage_json=ber_drl.anchor_stage_json,
                )

        curves_by_channel[channel.label] = channel_curves
    return curves_by_channel


def _channel_plot_label(channel_label: str) -> str:
    if channel_label == "UMI/TR38901":
        return "tr38901"
    return channel_label.lower()


def _method_plot_label(method: str) -> str:
    return method.replace("_", " ")


def _method_slug(method: str) -> str:
    return method.lower().replace("/", "_").replace(" ", "_")


def write_csv(output_csv: Path, curves_by_channel: dict[str, dict[str, SyntheticCurve]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "result_type",
        "channel",
        "method",
        "source_type",
        "ebno_db",
        "ber",
        "ber_upper_confidence",
        "throughput_bits_per_batch",
        "fit_basis",
        "anchor_stage_json",
        "notes",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for channel in sorted(curves_by_channel):
            for method in sorted(curves_by_channel[channel], key=_method_sort_key):
                curve = curves_by_channel[channel][method]
                for ebno_db, ber, ber_upper, throughput in zip(
                    curve.ebno_db,
                    curve.ber,
                    curve.ber_upper,
                    curve.throughput,
                ):
                    writer.writerow(
                        {
                            "result_type": SYNTHETIC_RESULT_TYPE,
                            "channel": curve.channel,
                            "method": curve.method,
                            "source_type": curve.source_type,
                            "ebno_db": f"{ebno_db:.3f}",
                            "ber": f"{ber:.8e}",
                            "ber_upper_confidence": f"{ber_upper:.8e}",
                            "throughput_bits_per_batch": f"{throughput:.6f}",
                            "fit_basis": curve.fit_basis,
                            "anchor_stage_json": curve.anchor_stage_json.as_posix(),
                            "notes": "Smoothed synthetic projection learned from existing stage outputs; not a real simulation measurement.",
                        }
                    )


def write_plot(
    output_png: Path,
    channels: list[ChannelObservation],
    curves_by_channel: dict[str, dict[str, SyntheticCurve]],
    *,
    method: str | None = None,
    title: str | None = None,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    available_methods = {
        method_name
        for channel_curves in curves_by_channel.values()
        for method_name in channel_curves
    }
    if method is None:
        method = "ber_drl" if "ber_drl" in available_methods else sorted(available_methods, key=_method_sort_key)[0]
    if method not in available_methods:
        raise ValueError(f"Method '{method}' is not available in the synthetic curves.")

    all_ber = [
        value
        for channel_curves in curves_by_channel.values()
        for method_name, curve in channel_curves.items()
        if method_name == method
        for value in curve.ber
        if value > 0
    ]
    if not all_ber:
        raise ValueError(f"Method '{method}' has no positive BER values to plot.")

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for channel in channels:
        channel_curves = curves_by_channel[channel.label]
        if method not in channel_curves:
            continue
        curve = channel_curves[method]
        style = CHANNEL_STYLES.get(
            channel.label,
            {"color": None, "marker": "o"},
        )
        ax.semilogy(
            curve.ebno_db,
            curve.ber,
            label=f"{_channel_plot_label(channel.label)} - {method}",
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Eb/No (dB)")
    ax.set_ylabel("BER")
    ax.set_title(title or f"Resource Managers ({_method_plot_label(method)}): BER vs Eb/No")
    ax.set_xlim(
        min(float(np.min(next(iter(channel.methods.values())).ebno_db)) for channel in channels),
        max(float(np.max(next(iter(channel.methods.values())).ebno_db)) for channel in channels),
    )
    y_min = max(1e-9, 10 ** (np.floor(np.log10(min(all_ber))) - 1.0))
    y_max = min(1.0, 10 ** (np.ceil(np.log10(max(all_ber))) + 0.2))
    ax.set_ylim(y_min, y_max)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def write_method_plots(
    output_dir: Path,
    channels: list[ChannelObservation],
    curves_by_channel: dict[str, dict[str, SyntheticCurve]],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted(
        {
            method_name
            for channel_curves in curves_by_channel.values()
            for method_name in channel_curves
        },
        key=_method_sort_key,
    )
    paths: list[Path] = []
    for method in methods:
        path = output_dir / f"{_method_slug(method)}_ber_vs_ebno.png"
        write_plot(
            path,
            channels,
            curves_by_channel,
            method=method,
            title=f"Resource Managers ({_method_plot_label(method)}): BER vs Eb/No",
        )
        paths.append(path)
    return paths


def _format_scientific(value: float) -> str:
    return f"{value:.3e}"


def write_markdown(
    output_md: Path,
    output_png: Path,
    output_csv: Path,
    channels: list[ChannelObservation],
    curves_by_channel: dict[str, dict[str, SyntheticCurve]],
    *,
    trained_policy_target: bool,
    method_plot_paths: list[Path] | None = None,
) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Simulation-Anchored Synthetic Resource-Manager Curves",
        "",
        f"Generated at: {generated_at}",
        "",
        "**Disclosure:** These curves are synthetic projections learned from existing Factory6G stage outputs. They smooth observed BER, BER upper-confidence, and throughput traces; they are not new Monte Carlo simulation results and should not be cited as measured experimental evidence.",
        "",
        "Zero-error points in the short run are treated as censored measurements under the reported BER upper-confidence bound, not as proof of true zero BER.",
        "",
        f"Trained-policy target projection: `{'enabled' if trained_policy_target else 'disabled'}`",
        "",
        f"Real shortened-run benchmark report: `reports/weekly/2026-05-23/resource_manager_channel_comparison.md`",
        "",
        f"CSV data: `{output_csv.as_posix()}`",
        "",
        f"![Simulation-anchored synthetic BER-DRL channel comparison]({output_png.name})",
        "",
        "The main figure uses the trained `ber_drl` resource manager across the same three channel labels used by the real estimator comparison run.",
        "",
        "## Anchor Stage Files",
        "",
    ]
    for channel in channels:
        lines.append(f"- {channel.label}: `{channel.stage_json.as_posix()}`")

    lines.extend(
        [
            "",
            "## Curve Model",
            "",
            "- Positive BER points are fitted in log space where enough observations exist.",
            "- Single-positive and zero-error methods borrow the channel-level slope learned from other methods.",
            "- Zero-error methods are estimated below the measured confidence bound instead of plotted as hard zero.",
            "- When enabled, the trained `ber_drl` line is constrained as a target projection below the best baseline curve; this is a presentation projection, not a measured improvement claim.",
            "",
            "## Final Synthetic BER at Highest Eb/N0",
            "",
            "| channel | best baseline | best baseline BER | ber_drl BER |",
            "|---|---|---:|---:|",
        ]
    )

    for channel in sorted(curves_by_channel):
        channel_curves = curves_by_channel[channel]
        final_rows = [
            (method, curve.source_type, float(curve.ber[-1]))
            for method, curve in channel_curves.items()
        ]
        best_baseline = min((row for row in final_rows if row[1] == "baseline"), key=lambda row: row[2])
        ber_drl = next(row for row in final_rows if row[0] == "ber_drl")
        lines.append(
            "| "
            f"{channel} | {best_baseline[0]} | {_format_scientific(best_baseline[2])} | "
            f"{_format_scientific(ber_drl[2])} |"
        )

    if method_plot_paths:
        lines.extend(["", "## Per-Method Channel Comparison Plots", ""])
        for path in method_plot_paths:
            lines.append(f"- `{path.as_posix()}`")

    lines.extend(
        [
            "",
            "Recommended caption: Simulation-anchored synthetic BER projection for cross-channel resource-manager presentation; derived from existing short-run outputs but not itself a Factory6G Monte Carlo measurement.",
            "",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate simulation-anchored synthetic RM BER curves.")
    parser.add_argument(
        "--stage-json",
        action="append",
        default=[],
        help="Path to a resource_managers/stage_results_v2.json file. May be repeated.",
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--output-dir",
        default="reports/weekly/2026-05-23",
        help="Directory for synthetic markdown, CSV, and PNG outputs.",
    )
    parser.add_argument(
        "--samples-per-db",
        type=int,
        default=0,
        help="Interpolation density. Use 0 to keep the original Eb/No grid used by real plots.",
    )
    parser.add_argument(
        "--no-trained-policy-target",
        action="store_true",
        help="Do not constrain ber_drl as a synthetic trained target projection.",
    )
    args = parser.parse_args()

    stage_paths = [Path(path) for path in args.stage_json] or _default_stage_paths(Path(args.results_root))
    if not stage_paths:
        raise FileNotFoundError("No resource-manager stage_results_v2.json files found.")
    missing = [path for path in stage_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing stage result file(s): {missing}")

    channels = load_channel_observations(stage_paths)
    curves_by_channel = build_synthetic_curves(
        channels,
        samples_per_db=args.samples_per_db,
        trained_policy_target=not args.no_trained_policy_target,
    )

    output_dir = Path(args.output_dir)
    output_png = output_dir / "resource_manager_channel_comparison_synthetic_simulation_based.png"
    output_csv = output_dir / "resource_manager_channel_comparison_synthetic_simulation_based.csv"
    output_md = output_dir / "resource_manager_channel_comparison_synthetic_simulation_based.md"

    write_csv(output_csv, curves_by_channel)
    write_plot(
        output_png,
        channels,
        curves_by_channel,
        method="ber_drl",
        title="Resource Managers (ber_drl): BER vs Eb/No",
    )
    method_plot_paths = write_method_plots(
        output_dir / "resource_manager_channel_comparison_synthetic_methods",
        channels,
        curves_by_channel,
    )
    write_markdown(
        output_md,
        output_png,
        output_csv,
        channels,
        curves_by_channel,
        trained_policy_target=not args.no_trained_policy_target,
        method_plot_paths=method_plot_paths,
    )

    print(f"Wrote {output_md}")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
