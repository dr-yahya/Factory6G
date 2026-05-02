#!/usr/bin/env python3
"""Generate fresh-vs-archived RM benchmark comparison artifacts.

Usage (run in Docker):
  python scripts/maintenance/generate_rm_benchmark_refresh.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
ARCHIVED_STAGE_JSON = (
    RESULTS_DIR / "archieve" / "20260314_131034_simulation" / "resource_managers" / "stage_results_v2.json"
)
OVERLAP_EBNO = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


@dataclass(frozen=True)
class FreshProfile:
    profile: str
    run_dir: Path
    summary_json: Path
    stage_json: Path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _find_latest_profile_run(channel: str) -> FreshProfile:
    # The run suffix maps tr38901 -> scenario name ("umi") in main._build_run_suffix().
    suffix_channel = "umi" if channel == "tr38901" else channel
    pattern = f"*_static_round_robin_max_throughput_pf_wmmse_queue_aware_drl_{suffix_channel}_qpsk_s"
    matches = sorted(
        [p for p in RESULTS_DIR.glob(pattern) if p.is_dir()],
        key=lambda p: p.name,
    )
    if not matches:
        raise FileNotFoundError(f"No fresh run directory found for pattern: {pattern}")
    run_dir = matches[-1]
    summary_json = run_dir / "summary_v2.json"
    stage_json = run_dir / "resource_managers" / "stage_results_v2.json"
    if not summary_json.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_json}")
    if not stage_json.exists():
        raise FileNotFoundError(f"Missing stage file: {stage_json}")
    return FreshProfile(
        profile=f"{channel},s,low",
        run_dir=run_dir,
        summary_json=summary_json,
        stage_json=stage_json,
    )


def _series_at_ebno(stage_payload: dict[str, Any], method: str, metric: str, ebno_points: list[float]) -> list[float]:
    ebno_all = [float(v) for v in stage_payload["ebno_db_range"]]
    values = stage_payload["methods"][method][metric]
    by_ebno: dict[float, float] = {float(e): float(values[i]) for i, e in enumerate(ebno_all)}
    out: list[float] = []
    for point in ebno_points:
        if point in by_ebno:
            out.append(by_ebno[point])
    return out


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / float(len(values))


def _format_float(value: float) -> str:
    if value != value:  # NaN
        return "nan"
    if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:.6e}"
    return f"{value:.6f}"


def _format_pct(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value * 100.0:.2f}%"


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    archived_stage = _load_json(ARCHIVED_STAGE_JSON)
    rayleigh = _find_latest_profile_run("rayleigh")
    tr38901 = _find_latest_profile_run("tr38901")
    profiles = [rayleigh, tr38901]

    archived_methods = archived_stage["methods"]
    method_order = [
        "static",
        "round_robin",
        "max_throughput",
        "pf",
        "wmmse",
        "queue_aware",
        "drl",
    ]
    for method in method_order:
        if method not in archived_methods:
            raise KeyError(f"Archived baseline missing method '{method}'.")

    rows: list[dict[str, Any]] = []
    run_notes: list[str] = []

    for profile in profiles:
        summary = _load_json(profile.summary_json)
        stage = _load_json(profile.stage_json)
        run_notes.append(f"- `{profile.profile}` -> `{profile.run_dir}`")

        agg = summary["aggregate_means"]["resource_managers"]
        runtime_totals = summary["runtime_totals_sec"]["resource_managers"]

        for method in method_order:
            if method not in stage["methods"]:
                raise KeyError(f"Fresh run '{profile.run_dir.name}' missing method '{method}'.")
            mean_ber = float(agg[method]["ber"])
            mean_ber_u = float(agg[method]["ber_upper_confidence"])
            mean_latency = float(agg[method]["latency_ms"])
            mean_tp = float(agg[method]["throughput_bits_per_batch"])
            runtime_total = float(runtime_totals[method])

            fresh_overlap = _series_at_ebno(stage, method, "ber", OVERLAP_EBNO)
            arch_overlap = _series_at_ebno(archived_stage, method, "ber", OVERLAP_EBNO)
            fresh_overlap_mean = _mean(fresh_overlap)
            arch_overlap_mean = _mean(arch_overlap)
            if arch_overlap_mean == 0.0:
                delta = float("nan")
            else:
                delta = (fresh_overlap_mean - arch_overlap_mean) / arch_overlap_mean

            rows.append(
                {
                    "method": method,
                    "profile": profile.profile,
                    "mean_ber": mean_ber,
                    "mean_ber_upper_confidence": mean_ber_u,
                    "mean_latency_ms": mean_latency,
                    "mean_throughput_bits_per_batch": mean_tp,
                    "runtime_total_sec": runtime_total,
                    "delta_vs_20260314": delta,
                }
            )

    date_tag = datetime.now().strftime("%Y%m%d")
    csv_path = REPORTS_DIR / f"resource_manager_benchmark_refresh_{date_tag}.csv"
    md_path = REPORTS_DIR / f"resource_manager_benchmark_refresh_{date_tag}.md"

    fieldnames = [
        "method",
        "profile",
        "mean_ber",
        "mean_ber_upper_confidence",
        "mean_latency_ms",
        "mean_throughput_bits_per_batch",
        "runtime_total_sec",
        "delta_vs_20260314",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines: list[str] = []
    md_lines.append("# Resource Manager Benchmark Refresh")
    md_lines.append("")
    md_lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    md_lines.append(f"- Archived baseline: `{ARCHIVED_STAGE_JSON}`")
    md_lines.append("- Overlap Eb/N0 points used for delta: `0, 2, 4, 6, 8, 10`")
    md_lines.append("- Fresh run directories:")
    md_lines.extend(run_notes)
    md_lines.append("")
    md_lines.append("| method | profile | mean_ber | mean_ber_upper_confidence | mean_latency_ms | mean_throughput_bits_per_batch | runtime_total_sec | delta_vs_20260314 |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["profile"]),
                    _format_float(float(row["mean_ber"])),
                    _format_float(float(row["mean_ber_upper_confidence"])),
                    _format_float(float(row["mean_latency_ms"])),
                    _format_float(float(row["mean_throughput_bits_per_batch"])),
                    _format_float(float(row["runtime_total_sec"])),
                    _format_pct(float(row["delta_vs_20260314"])),
                ]
            )
            + " |"
        )
    md_lines.append("")
    md_lines.append("`delta_vs_20260314` is relative change of overlap-mean BER versus archived run 20260314_131034.")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
