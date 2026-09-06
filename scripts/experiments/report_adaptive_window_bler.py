#!/usr/bin/env python3
"""Turn an estimator sweep into the adaptive-window reliability table.

Reads a run's ``stage_results_v2.json`` and reports, per Eb/No point, each
estimator's BLER against a reference alongside the paired bootstrap interval.

The paired interval is the number worth quoting. Every method sees the identical
channel, noise draw and source bits at a given batch, so differencing per batch
before averaging removes the shared variance -- which is what turns "the curve
is lower" into "improves BLER by X, 95% CI [a, b]". A comparison of the two
marginal curves cannot support the second statement.

    python scripts/experiments/report_adaptive_window_bler.py <run_dir_or_json>
        [--method adaptive_window] [--reference dft]

If the run was produced before per-batch samples were retained, and the named
reference is not the one the run itself paired against, the interval columns are
reported as unavailable rather than silently reusing the wrong reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from factory6g.sim.stages.common import compare_methods_paired


def load_stage(target: Path) -> dict[str, Any]:
    if target.is_dir():
        matches = sorted(target.rglob("stage_results_v2.json"))
        if not matches:
            raise SystemExit(f"no stage_results_v2.json under {target}")
        target = matches[-1]
    return json.loads(target.read_text())


def repair(stage: dict[str, Any], reference: str, confidence_level: float) -> dict[str, Any]:
    """Recompute the paired comparison against `reference` from stored samples."""
    samples = stage.get("paired_samples")
    if not samples or reference not in samples:
        return {}
    points = {
        method: [
            {
                "batch_block_errors": entry.get("batch_block_errors", []),
                "batch_blocks": entry.get("batch_blocks", []),
            }
            for entry in entries
        ]
        for method, entries in samples.items()
    }
    return compare_methods_paired(
        points, reference=reference, confidence_level=confidence_level
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="run directory or stage_results_v2.json")
    parser.add_argument("--method", default="adaptive_window")
    parser.add_argument("--reference", default="dft")
    args = parser.parse_args()

    stage = load_stage(args.target)
    methods = stage["methods"]
    ebno = stage["ebno_db_range"]
    for name in (args.method, args.reference):
        if name not in methods:
            raise SystemExit(f"'{name}' is not in this run ({', '.join(methods)})")

    confidence = float(stage.get("confidence_level", 0.95))
    if stage.get("paired_reference") == args.reference:
        comparisons = stage.get("paired_comparisons", {})
    else:
        comparisons = repair(stage, args.reference, confidence)
    rows = comparisons.get(args.method, [])

    method = methods[args.method]
    baseline = methods[args.reference]
    print(f"{args.method} vs {args.reference}   ({int(confidence * 100)}% paired bootstrap CI)")
    header = ("Eb/No", f"BLER {args.reference}", f"BLER {args.method}", "delta", "95% CI", "sig")
    print("{:>6} {:>14} {:>16} {:>11} {:>24} {:>5}".format(*header))

    for index, ebno_db in enumerate(ebno):
        ref_bler = baseline["bler"][index]
        new_bler = method["bler"][index]
        row = rows[index] if index < len(rows) else {}
        if row.get("num_paired_batches"):
            interval = f"[{row['ci_lower']:+.2e}, {row['ci_upper']:+.2e}]"
            delta = f"{row['mean_bler_delta']:+.2e}"
            significant = "yes" if row.get("significant") else "no"
        else:
            interval, delta, significant = "unavailable", f"{new_bler - ref_bler:+.2e}", "-"
        print(
            f"{ebno_db:6.1f} {ref_bler:14.4e} {new_bler:16.4e} "
            f"{delta:>11} {interval:>24} {significant:>5}"
        )

    print()
    print("NMSE (dB) and error-variance calibration, for context:")
    print("{:>6} {:>10} {:>10} {:>10} {:>10}".format(
        "Eb/No", f"nmse {args.reference[:6]}", f"nmse {args.method[:6]}", "cal ref", "cal new"))
    for index, ebno_db in enumerate(ebno):
        def fmt(values: list[Any]) -> str:
            value = values[index] if index < len(values) else None
            return "-" if value is None else f"{value:.2f}"
        print("{:6.1f} {:>10} {:>10} {:>10} {:>10}".format(
            ebno_db, fmt(baseline["nmse_db"]), fmt(method["nmse_db"]),
            fmt(baseline["err_var_calibration"]), fmt(method["err_var_calibration"])))


if __name__ == "__main__":
    main()
