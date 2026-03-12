from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


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
                if isinstance(values, list) and values
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


def _write_stage_plots(*, stage_dir: Path, payload: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage_name = payload["stage"].replace("_", " ").title()
    ebno_range = payload["ebno_db_range"]
    methods = payload["methods"]
    runtime_totals = payload.get("runtime_totals_sec", {})

    _plot_metric_vs_ebno(
        plt=plt,
        methods=methods,
        ebno_range=ebno_range,
        metric="ber",
        ylabel="BER",
        title=f"{stage_name}: BER vs Eb/No",
        output_path=stage_dir / "ber_vs_ebno.png",
        ylog=True,
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


def _plot_metric_vs_ebno(
    *,
    plt,
    methods: dict[str, dict[str, list[float]]],
    ebno_range: list[float],
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    ylog: bool = False,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    markers = ["o", "s", "D", "^", "v", "x", "*", "+"]
    for idx, (name, metric_map) in enumerate(methods.items()):
        if metric not in metric_map:
            continue
        raw_values = np.asarray(metric_map[metric], dtype=float)
        smooth_values = _smooth_metric_curve(
            metric=metric,
            raw_values=raw_values,
            metric_map=metric_map,
        )
        if ylog:
            raw_plot = np.clip(raw_values, 1e-12, np.inf)
            smooth_plot = np.clip(smooth_values, 1e-12, np.inf)
            raw_handle = ax.semilogy(
                ebno_range,
                raw_plot,
                linestyle="None",
                marker=markers[idx % len(markers)],
                markersize=6,
                alpha=0.25,
                label="_nolegend_",
            )
            color = raw_handle[0].get_color()
            ax.semilogy(
                ebno_range,
                smooth_plot,
                color=color,
                linewidth=2.2,
                label=name,
            )
        else:
            raw_handle = ax.plot(
                ebno_range,
                raw_values,
                linestyle="None",
                marker=markers[idx % len(markers)],
                markersize=6,
                alpha=0.25,
                label="_nolegend_",
            )
            color = raw_handle[0].get_color()
            ax.plot(
                ebno_range,
                smooth_values,
                color=color,
                linewidth=2.2,
                label=name,
            )
    ax.set_xlabel("Eb/No (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.text(
        0.01,
        0.02,
        "solid=smoothed, markers=raw",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.7,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _smooth_metric_curve(
    *,
    metric: str,
    raw_values: np.ndarray,
    metric_map: dict[str, list[float]],
) -> np.ndarray:
    if raw_values.size == 0:
        return raw_values
    if metric == "ber":
        return _smooth_ber_curve(metric_map=metric_map, raw_values=raw_values)
    if metric == "throughput_bits_per_batch":
        weights = _metric_weights(metric_map, fallback_len=raw_values.size)
        return _weighted_isotonic(raw_values, weights, increasing=True)
    if metric == "latency_ms":
        weights = _metric_weights(metric_map, fallback_len=raw_values.size)
        return _weighted_isotonic(raw_values, weights, increasing=False)
    if metric == "avg_power_w":
        return _ema_smooth(raw_values, alpha=0.35)
    return raw_values


def _smooth_ber_curve(
    *,
    metric_map: dict[str, list[float]],
    raw_values: np.ndarray,
) -> np.ndarray:
    n = _coerce_array(metric_map.get("total_bits", []), fallback_len=raw_values.size, default=1.0)
    k = _coerce_array(metric_map.get("bit_errors", []), fallback_len=raw_values.size, default=0.0)
    n = np.maximum(n, 0.0)
    k = np.maximum(k, 0.0)

    if n.size != raw_values.size or k.size != raw_values.size:
        p = np.clip(raw_values, 1e-12, 1.0)
        weights = np.maximum(_metric_weights(metric_map, fallback_len=raw_values.size), 1.0)
    else:
        p = (k + 0.5) / (n + 1.0)
        p = np.clip(p, 1e-12, 1.0)
        weights = np.maximum(n, 1.0)

    y = np.log10(p)
    y_smooth = _weighted_isotonic(y, weights, increasing=False)
    return np.clip(np.power(10.0, y_smooth), 1e-12, 1.0)


def _metric_weights(metric_map: dict[str, list[float]], fallback_len: int) -> np.ndarray:
    if "total_bits" in metric_map:
        weights = _coerce_array(metric_map["total_bits"], fallback_len=fallback_len, default=1.0)
    elif "num_batches" in metric_map:
        weights = _coerce_array(metric_map["num_batches"], fallback_len=fallback_len, default=1.0)
    else:
        weights = np.ones(fallback_len, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
    return weights


def _coerce_array(values: list[float], *, fallback_len: int, default: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == fallback_len:
        return arr
    if arr.size == 0:
        return np.full(fallback_len, default, dtype=float)
    if arr.size > fallback_len:
        return arr[:fallback_len]
    padded = np.full(fallback_len, default, dtype=float)
    padded[: arr.size] = arr
    return padded


def _ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    out = np.array(arr, copy=True)
    first_idx = None
    for idx, val in enumerate(out):
        if math.isfinite(float(val)):
            first_idx = idx
            break
    if first_idx is None:
        return out
    for idx in range(first_idx + 1, out.size):
        current = out[idx]
        prev = out[idx - 1]
        if not math.isfinite(float(current)):
            out[idx] = prev
        else:
            out[idx] = alpha * current + (1.0 - alpha) * prev
    return out


def _weighted_isotonic(values: np.ndarray, weights: np.ndarray, *, increasing: bool) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if y.size == 0:
        return y
    if y.size != w.size:
        raise ValueError(
            f"values and weights must have same length (got {y.size} vs {w.size})."
        )

    valid = np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        return y

    y_work = y[valid]
    if not increasing:
        y_work = -y_work
    w_work = w[valid]

    block_starts: list[int] = []
    block_ends: list[int] = []
    block_means: list[float] = []
    block_weights: list[float] = []

    for idx, (val, wt) in enumerate(zip(y_work, w_work)):
        block_starts.append(idx)
        block_ends.append(idx)
        block_means.append(float(val))
        block_weights.append(float(wt))

        while len(block_means) >= 2 and block_means[-2] > block_means[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_mean = (
                (block_means[-2] * block_weights[-2]) + (block_means[-1] * block_weights[-1])
            ) / merged_weight
            merged_start = block_starts[-2]
            merged_end = block_ends[-1]

            block_starts = block_starts[:-2] + [merged_start]
            block_ends = block_ends[:-2] + [merged_end]
            block_means = block_means[:-2] + [float(merged_mean)]
            block_weights = block_weights[:-2] + [float(merged_weight)]

    fitted_valid = np.empty_like(y_work, dtype=float)
    for start, end, mean in zip(block_starts, block_ends, block_means):
        fitted_valid[start : end + 1] = mean

    if not increasing:
        fitted_valid = -fitted_valid

    fitted = np.array(y, copy=True)
    fitted[valid] = fitted_valid
    return fitted


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))
