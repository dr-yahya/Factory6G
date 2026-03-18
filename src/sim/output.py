from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.sim.stages.common import (
    MIN_RESOLVED_BIT_ERRORS,
    POINT_STATUS_RESOLVED,
    POINT_STATUS_UPPER_BOUND_ONLY,
)


SCHEMA_VERSION = "2.0"


def write_stage_outputs(
    *,
    run_id: str,
    stage_name: str,
    stage_result: dict[str, Any],
    stage_dir: Path,
    config_snapshot: dict[str, Any],
    confidence_level: float,
    plot_results: bool,
) -> dict[str, str]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "stage": stage_name,
        "ebno_db_range": stage_result["ebno_db_range"],
        "confidence_level": confidence_level,
        "config_snapshot": config_snapshot,
        "methods": stage_result["methods"],
        "runtime_totals_sec": stage_result["runtime_totals_sec"],
    }

    json_path = stage_dir / "stage_results_v2.json"
    csv_path = stage_dir / "stage_results_v2.csv"
    _write_json(json_path, payload)
    _write_stage_csv(csv_path, payload)
    if plot_results:
        _write_stage_plots(stage_dir=stage_dir, payload=payload)
    return {
        "dir": str(stage_dir),
        "json": str(json_path),
        "csv": str(csv_path),
    }


def write_summary_outputs(
    *,
    run_id: str,
    run_dir: Path,
    stage_order: list[str],
    stage_paths: dict[str, dict[str, str]],
    stage_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    aggregate_means: dict[str, dict[str, dict[str, float]]] = {}
    runtime_totals_sec: dict[str, dict[str, float]] = {}
    for stage_name, payload in stage_payloads.items():
        stage_means: dict[str, dict[str, float]] = {}
        for method, metrics in payload["methods"].items():
            stage_means[method] = {
                metric: _safe_mean(values)
                for metric, values in metrics.items()
                if _is_numeric_series(values)
            }
        aggregate_means[stage_name] = stage_means
        runtime_totals_sec[stage_name] = {
            method: float(value)
            for method, value in payload.get("runtime_totals_sec", {}).items()
        }

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "stage_order": stage_order,
        "stage_paths": stage_paths,
        "aggregate_means": aggregate_means,
        "runtime_totals_sec": runtime_totals_sec,
    }
    summary_json_path = run_dir / "summary_v2.json"
    summary_csv_path = run_dir / "summary_v2.csv"
    _write_json(summary_json_path, summary_payload)
    _write_summary_csv(summary_csv_path, summary_payload)
    return {
        "summary_payload": summary_payload,
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
    }


def _write_stage_csv(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    ebno_values = payload["ebno_db_range"]
    stage = payload["stage"]
    methods = payload["methods"]
    for method_name, metric_map in methods.items():
        for metric_name, values in metric_map.items():
            if not isinstance(values, list):
                continue
            for index, value in enumerate(values):
                ebno = ebno_values[index] if index < len(ebno_values) else ""
                rows.append(
                    {
                        "stage": stage,
                        "method": method_name,
                        "metric": metric_name,
                        "ebno_db": ebno,
                        "value": value,
                    }
                )
        rows.append(
            {
                "stage": stage,
                "method": method_name,
                "metric": "runtime_total_sec",
                "ebno_db": "",
                "value": payload.get("runtime_totals_sec", {}).get(method_name, 0.0),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "method", "metric", "ebno_db", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: Path, payload: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for stage in payload["stage_order"]:
        stage_path = payload["stage_paths"][stage]
        rows.extend(
            [
                {"stage": stage, "method": "__stage__", "metric": "stage_dir", "value": stage_path["dir"]},
                {"stage": stage, "method": "__stage__", "metric": "stage_json", "value": stage_path["json"]},
                {"stage": stage, "method": "__stage__", "metric": "stage_csv", "value": stage_path["csv"]},
            ]
        )
        for method, metric_map in payload["aggregate_means"].get(stage, {}).items():
            for metric, value in metric_map.items():
                rows.append(
                    {
                        "stage": stage,
                        "method": method,
                        "metric": f"mean_{metric}",
                        "value": value,
                    }
                )
        for method, runtime in payload["runtime_totals_sec"].get(stage, {}).items():
            rows.append(
                {
                    "stage": stage,
                    "method": method,
                    "metric": "runtime_total_sec",
                    "value": runtime,
                }
            )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "method", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_combined_modulation_plot(
    run_dir: Path,
    entries: list[tuple[str, str, dict[str, Any], list[float]]],
) -> None:
    """Plot BER for all (method, modulation) combinations on one chart.

    entries: list of (mod_display_name, stage_name, methods_dict, ebno_db_range)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    markers = ["o", "s", "D", "^", "v", "x", "*", "+", "p", "h"]
    linestyles = ["-", "--", ":", "-."]

    plot_idx = 0
    for mod_display, stage_name, methods, ebno_range in entries:
        x = np.asarray(ebno_range, dtype=float)
        for method_name, metric_map in methods.items():
            if "ber" not in metric_map:
                continue
            ber = _coerce_float_array(metric_map["ber"])
            upper = _coerce_float_array(metric_map.get("ber_upper_confidence", metric_map["ber"]))
            statuses = _point_status_array(metric_map, fallback_len=ber.size)

            label = f"{method_name} – {mod_display}"
            marker = markers[plot_idx % len(markers)]
            ls = linestyles[plot_idx % len(linestyles)]
            upper_mask = statuses == POINT_STATUS_UPPER_BOUND_ONLY
            valid_mask = ber > 0

            color = None
            if np.any(valid_mask):
                handle = ax.semilogy(
                    x[valid_mask], ber[valid_mask],
                    marker=marker, linestyle=ls, linewidth=2,
                    label=label,
                )
                color = handle[0].get_color()

            upper_show = upper_mask & ~valid_mask
            if np.any(upper_show):
                handle = ax.semilogy(
                    x[upper_show], upper[upper_show],
                    marker=marker, linestyle="--", linewidth=1.5,
                    markerfacecolor="none",
                    color=color,
                    label=label if color is None else "_nolegend_",
                )
            plot_idx += 1

    ax.set_xlabel("Eb/No (dB)")
    ax.set_ylabel("BER")
    ax.set_title("BER vs Eb/No — All Modulations")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    ax.text(
        0.01, 0.02,
        "dashed/open = 95% BER upper bound (zero observed errors)",
        transform=ax.transAxes, fontsize=9, alpha=0.75,
    )
    fig.tight_layout()
    out_path = run_dir / "combined_ber.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[plot] Combined BER plot: {out_path}")


def _write_stage_plots(*, stage_dir: Path, payload: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage_name = payload["stage"].replace("_", " ").title()
    ebno_range = payload["ebno_db_range"]
    methods = payload["methods"]
    runtime_totals = payload.get("runtime_totals_sec", {})

    _plot_ber_publication(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        title=f"{stage_name}: BER vs Eb/No",
        output_path=stage_dir / "ber_vs_ebno.png",
    )
    _plot_ber_raw(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        title=f"{stage_name}: Raw BER vs Eb/No",
        output_path=stage_dir / "ber_raw_vs_ebno.png",
    )
    _plot_metric_vs_ebno(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        metric="latency_ms",
        ylabel="Latency (ms)",
        title=f"{stage_name}: Latency vs Eb/No",
        output_path=stage_dir / "latency_vs_ebno.png",
    )
    _plot_metric_vs_ebno(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        metric="throughput_bits_per_batch",
        ylabel="Throughput (bits/batch)",
        title=f"{stage_name}: Throughput vs Eb/No",
        output_path=stage_dir / "throughput_vs_ebno.png",
    )
    _plot_metric_vs_ebno(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        metric="avg_power_w",
        ylabel="Average Power (W)",
        title=f"{stage_name}: Power vs Eb/No",
        output_path=stage_dir / "power_vs_ebno.png",
    )

    fig_runtime, ax_runtime = plt.subplots(1, 1, figsize=(9, 6))
    names = list(runtime_totals.keys())
    values = [runtime_totals[name] for name in names]
    ax_runtime.bar(names, values)
    ax_runtime.set_ylabel("Runtime (sec)")
    ax_runtime.set_title(f"{stage_name}: Runtime by Method")
    ax_runtime.grid(True, axis="y", alpha=0.3)
    fig_runtime.tight_layout()
    fig_runtime.savefig(stage_dir / "runtime_by_method.png", dpi=300)
    plt.close(fig_runtime)


def _plot_ber_publication(
    *,
    plt,
    methods: dict[str, dict[str, list[Any]]],
    ebno_range: list[float],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    markers = ["o", "s", "D", "^", "v", "x", "*", "+"]
    x = np.asarray(ebno_range, dtype=float)
    for idx, (name, metric_map) in enumerate(methods.items()):
        if "ber" not in metric_map:
            continue

        ber = _coerce_float_array(metric_map["ber"])
        upper = _coerce_float_array(metric_map.get("ber_upper_confidence", metric_map["ber"]))
        statuses = _point_status_array(metric_map, fallback_len=ber.size)

        upper_mask = statuses == POINT_STATUS_UPPER_BOUND_ONLY
        valid_mask = ber > 0
        marker = markers[idx % len(markers)]
        color = None

        if np.any(valid_mask):
            handle = ax.semilogy(
                x[valid_mask],
                ber[valid_mask],
                marker=marker,
                linewidth=2,
                label=name,
            )
            color = handle[0].get_color()

        upper_show = upper_mask & ~valid_mask
        if np.any(upper_show):
            handle = ax.semilogy(
                x[upper_show],
                upper[upper_show],
                marker=marker,
                markerfacecolor="none",
                linestyle="--",
                linewidth=1.5,
                color=color,
                label=name if color is None else "_nolegend_",
            )
            if color is None:
                color = handle[0].get_color()

    ax.set_xlabel("Eb/No (dB)")
    ax.set_ylabel("BER")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.text(
        0.01,
        0.02,
        "dashed/open = 95% BER upper bound (zero observed errors)",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.75,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_ber_raw(
    *,
    plt,
    methods: dict[str, dict[str, list[Any]]],
    ebno_range: list[float],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    markers = ["o", "s", "D", "^", "v", "x", "*", "+"]
    x = np.asarray(ebno_range, dtype=float)

    for idx, (name, metric_map) in enumerate(methods.items()):
        if "ber" not in metric_map:
            continue
        values = _coerce_float_array(metric_map["ber"])
        ax.semilogy(
            x,
            values,
            marker=markers[idx % len(markers)],
            label=name,
            linewidth=2,
        )

    ax.set_xlabel("Eb/No (dB)")
    ax.set_ylabel("BER")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_metric_vs_ebno(
    *,
    plt,
    methods: dict[str, dict[str, list[Any]]],
    ebno_range: list[float],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    markers = ["o", "s", "D", "^", "v", "x", "*", "+"]
    for idx, (name, metric_map) in enumerate(methods.items()):
        if metric not in metric_map:
            continue
        values = _coerce_float_array(metric_map[metric])
        ax.plot(
            ebno_range,
            values,
            marker=markers[idx % len(markers)],
            label=name,
            linewidth=2,
        )
    ax.set_xlabel("Eb/No (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)



def _point_status_array(metric_map: dict[str, list[Any]], *, fallback_len: int) -> np.ndarray:
    statuses = metric_map.get("point_status", [])
    if isinstance(statuses, list) and len(statuses) == fallback_len:
        return np.asarray([str(value) for value in statuses], dtype=object)
    bit_errors = metric_map.get("bit_errors", [])
    if isinstance(bit_errors, list) and len(bit_errors) == fallback_len:
        return np.asarray(
            [
                POINT_STATUS_UPPER_BOUND_ONLY
                if float(value) < float(MIN_RESOLVED_BIT_ERRORS)
                else POINT_STATUS_RESOLVED
                for value in bit_errors
            ],
            dtype=object,
        )
    return np.asarray([POINT_STATUS_RESOLVED] * fallback_len, dtype=object)


def _coerce_float_array(values: list[Any]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _is_numeric_series(values: Any) -> bool:
    if not isinstance(values, list) or not values:
        return False
    try:
        arr = np.asarray(values)
    except Exception:
        return False
    return bool(np.issubdtype(arr.dtype, np.number))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _safe_mean(values: list[Any]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))
