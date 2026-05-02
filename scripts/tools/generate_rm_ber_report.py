from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_STAGE_PATHS = [
    Path(
        "results/20260420_040402_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_rayleigh_qpsk_s/"
        "resource_managers/stage_results_v2.json"
    ),
    Path(
        "results/20260420_043640_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_umi_qpsk_s/"
        "resource_managers/stage_results_v2.json"
    ),
]

CSV_FIELDS = [
    "run_id",
    "channel_label",
    "method",
    "rank_by_ber",
    "source_type",
    "ber",
    "ber_upper_confidence",
    "throughput_bits_per_batch",
    "latency_ms",
    "runtime_total_sec",
    "avg_power_w",
    "num_batches",
    "bit_errors",
    "total_bits",
    "stage_json",
]

RESEARCH_ANCHORS = [
    (
        "WMMSE baseline",
        "Shi et al., 2011",
        "https://doi.org/10.1109/TSP.2011.2147784",
    ),
    (
        "DNN approximation of wireless optimization",
        "Sun et al., 2017/2018",
        "https://arxiv.org/abs/1705.09412",
    ),
    (
        "Deep learning for physical-layer reliability/BER framing",
        "O'Shea and Hoydis, 2017",
        "https://arxiv.org/abs/1702.00832",
    ),
    (
        "DRL scheduling with buffer/state features",
        "Bansbach et al., 2021",
        "https://arxiv.org/abs/2108.12198",
    ),
    (
        "URLLC reliability/error-probability resource allocation",
        "Sun et al., 2019",
        "https://doi.org/10.1109/TWC.2018.2880907",
    ),
]


@dataclass(frozen=True)
class ReportRow:
    run_id: str
    channel_label: str
    method: str
    rank_by_ber: int
    source_type: str
    ber: float
    ber_upper_confidence: float
    throughput_bits_per_batch: float
    latency_ms: float
    runtime_total_sec: float
    avg_power_w: float
    num_batches: float
    bit_errors: float
    total_bits: float
    stage_json: str

    def as_csv_row(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in CSV_FIELDS}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_path_for_stage(stage_json: Path) -> Path:
    return stage_json.parents[1] / "summary_v2.json"


def _channel_label(stage_payload: dict[str, Any], stage_json: Path) -> str:
    system = stage_payload.get("config_snapshot", {}).get("system", {})
    channel = str(system.get("channel_model_type") or "").lower()
    scenario = str(system.get("scenario") or "").lower()
    if channel == "tr38901":
        return f"{scenario.upper() if scenario else 'UMI'}/TR38901"
    if channel:
        return channel.capitalize()
    name = stage_json.as_posix().lower()
    if "umi" in name:
        return "UMI/TR38901"
    if "rayleigh" in name:
        return "Rayleigh"
    return stage_json.parents[1].name


def _metric_from_stage(stage_payload: dict[str, Any], method: str, metric: str) -> float:
    values = stage_payload.get("methods", {}).get(method, {}).get(metric, [])
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return 0.0
    return float(sum(numeric) / len(numeric))


def _aggregate_metrics(
    stage_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    method: str,
) -> dict[str, float]:
    aggregate = (
        summary_payload.get("aggregate_means", {})
        .get("resource_managers", {})
        .get(method, {})
    )
    runtime_total = (
        summary_payload.get("runtime_totals_sec", {})
        .get("resource_managers", {})
        .get(method)
    )

    def metric(name: str) -> float:
        if name in aggregate:
            return float(aggregate[name])
        return _metric_from_stage(stage_payload, method, name)

    return {
        "ber": metric("ber"),
        "ber_upper_confidence": metric("ber_upper_confidence"),
        "throughput_bits_per_batch": metric("throughput_bits_per_batch"),
        "latency_ms": metric("latency_ms"),
        "runtime_total_sec": float(runtime_total) if runtime_total is not None else metric("runtime_sec"),
        "avg_power_w": metric("avg_power_w"),
        "num_batches": metric("num_batches"),
        "bit_errors": metric("bit_errors"),
        "total_bits": metric("total_bits"),
    }


def load_report_rows(stage_json_paths: list[Path]) -> list[ReportRow]:
    rows: list[ReportRow] = []
    for stage_json in stage_json_paths:
        stage_payload = _load_json(stage_json)
        summary_path = _summary_path_for_stage(stage_json)
        summary_payload = _load_json(summary_path) if summary_path.exists() else {}
        run_id = str(stage_payload.get("run_id") or stage_json.parents[1].name)
        channel_label = _channel_label(stage_payload, stage_json)
        methods = sorted(stage_payload.get("methods", {}).keys())
        for method in methods:
            metrics = _aggregate_metrics(stage_payload, summary_payload, method)
            rows.append(
                ReportRow(
                    run_id=run_id,
                    channel_label=channel_label,
                    method=method,
                    rank_by_ber=0,
                    source_type="trained" if method == "ber_drl" else "baseline",
                    stage_json=str(stage_json),
                    **metrics,
                )
            )

    reranked: list[ReportRow] = []
    by_channel: dict[str, list[ReportRow]] = {}
    for row in rows:
        by_channel.setdefault(row.channel_label, []).append(row)
    for channel_rows in by_channel.values():
        ranked_rows = sorted(
            channel_rows,
            key=lambda row: (
                row.ber,
                row.ber_upper_confidence,
                -row.throughput_bits_per_batch,
                row.latency_ms,
                row.runtime_total_sec,
                row.avg_power_w,
            ),
        )
        for rank, row in enumerate(ranked_rows, start=1):
            reranked.append(replace(row, rank_by_ber=rank))
    return reranked


def _format_float(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3):
        return f"{value:.4e}"
    return f"{value:.6g}"


def _write_csv(rows: list[ReportRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.channel_label, item.rank_by_ber, item.method)):
            writer.writerow(row.as_csv_row())


def _acceptance_lines(rows: list[ReportRow]) -> list[str]:
    lines: list[str] = []
    by_channel: dict[str, list[ReportRow]] = {}
    for row in rows:
        by_channel.setdefault(row.channel_label, []).append(row)

    for channel_label, channel_rows in sorted(by_channel.items()):
        ber_drl = next((row for row in channel_rows if row.method == "ber_drl"), None)
        baselines = [row for row in channel_rows if row.method != "ber_drl"]
        if not baselines:
            continue
        best_baseline = min(baselines, key=lambda row: (row.ber, row.ber_upper_confidence))
        if ber_drl is None:
            lines.append(
                f"- {channel_label}: no `ber_drl` benchmark row found yet; no trained-model improvement claim is made."
            )
            continue
        if ber_drl.ber < best_baseline.ber:
            verdict = "beats"
        elif ber_drl.ber == best_baseline.ber and ber_drl.ber_upper_confidence <= best_baseline.ber_upper_confidence:
            verdict = "matches"
        else:
            verdict = "does not beat"
        lines.append(
            f"- {channel_label}: `ber_drl` {verdict} best baseline `{best_baseline.method}` "
            f"(ber_drl BER={_format_float(ber_drl.ber)}, baseline BER={_format_float(best_baseline.ber)})."
        )
    return lines


def _write_markdown(rows: list[ReportRow], output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Resource Manager BER Comparison",
        "",
        f"Generated at: {generated_at}",
        "",
        "This report ranks resource managers by mean BER. BER upper confidence is used as the second reliability key, followed by throughput, latency, runtime, and power for engineering interpretation.",
        "",
        "## Acceptance Check",
        "",
    ]
    lines.extend(_acceptance_lines(rows))
    lines.extend(["", "## Ranked Results", ""])

    def append_table(table_rows: list[ReportRow]) -> None:
        lines.extend(
            [
                "| rank | method | type | BER | BER upper | throughput/batch | latency ms | runtime sec | avg power W |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in table_rows:
            lines.append(
                "| "
                f"{row.rank_by_ber} | {row.method} | {row.source_type} | "
                f"{_format_float(row.ber)} | {_format_float(row.ber_upper_confidence)} | "
                f"{_format_float(row.throughput_bits_per_batch)} | {_format_float(row.latency_ms)} | "
                f"{_format_float(row.runtime_total_sec)} | {_format_float(row.avg_power_w)} |"
            )

    channels = sorted({row.channel_label for row in rows})
    for channel in channels:
        channel_rows = sorted(
            [row for row in rows if row.channel_label == channel],
            key=lambda row: (row.rank_by_ber, row.method),
        )
        lines.extend(
            [
                f"### {channel}",
                "",
                "#### Baseline Table",
                "",
            ]
        )
        append_table([row for row in channel_rows if row.source_type == "baseline"])
        lines.extend(["", "#### Trained-Model Table", ""])
        trained_rows = [row for row in channel_rows if row.source_type == "trained"]
        if trained_rows:
            append_table(trained_rows)
        else:
            lines.append("No trained-model benchmark row found for this channel.")
        plot_paths = sorted({Path(row.stage_json).parent / "ber_vs_ebno.png" for row in channel_rows})
        for plot_path in plot_paths:
            if plot_path.exists():
                lines.append(f"\nBER plot: `{plot_path.as_posix()}`")
        lines.append("")

    lines.extend(
        [
            "## Research Anchors",
            "",
            "The hybrid BER-first policy follows the wireless resource-management literature: use optimization or heuristic policies to create strong labels, then train a neural policy for low-latency inference and reliability-aware adaptation.",
            "",
        ]
    )
    for label, citation, url in RESEARCH_ANCHORS:
        lines.append(f"- {label}: [{citation}]({url})")
    lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")


def _default_stage_paths(results_root: Path) -> list[Path]:
    preferred = [path for path in DEFAULT_STAGE_PATHS if path.exists()]
    if preferred:
        return preferred
    return sorted(results_root.glob("**/resource_managers/stage_results_v2.json"))


def generate_report(stage_json_paths: list[Path], output_md: Path, output_csv: Path) -> list[ReportRow]:
    rows = load_report_rows(stage_json_paths)
    _write_csv(rows, output_csv)
    _write_markdown(rows, output_md)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate BER-first resource-manager comparison report.")
    parser.add_argument(
        "--stage-json",
        action="append",
        default=[],
        help="Path to a resource_managers/stage_results_v2.json file. May be repeated.",
    )
    parser.add_argument("--results-root", default="results", help="Results root used when --stage-json is omitted.")
    parser.add_argument("--output-md", default="reports/resource_manager_ber_comparison.md")
    parser.add_argument("--output-csv", default="reports/resource_manager_ber_comparison.csv")
    args = parser.parse_args()

    stage_paths = [Path(path) for path in args.stage_json] or _default_stage_paths(Path(args.results_root))
    if not stage_paths:
        raise FileNotFoundError("No resource manager stage_results_v2.json files found.")
    missing = [path for path in stage_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing stage result file(s): {missing}")
    generate_report(stage_paths, Path(args.output_md), Path(args.output_csv))
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
