"""Calibrate the adaptive estimator's branch policy against BLER.

Why
---
The adaptive hybrid blends a DFT branch and an LMMSE branch, weighted by two
observable statistics: per-user LS SNR and delay-domain leakage past the cyclic
prefix. Those thresholds were set from MSE reasoning — the assumption that LMMSE
is the better estimator at low SNR.

On TR 38.901 UMi that assumption is false at every SNR: LMMSE has the best NMSE
and a BER roughly ninety times worse than DFT
(`reports/evidence/estimator-floor-tr38901/`). The policy is therefore optimising
against the wrong objective, and where it happens to choose correctly it does so
by luck.

Method
------
1. Measure DFT-only and LMMSE-only BLER across the Eb/No sweep, on shared batch
   contexts so the comparison is paired.
2. Derive the *oracle* branch per point — whichever actually decoded better — and
   the oracle's BLER, which is the best a hard-switching policy could achieve.
3. Measure the blend weight the current policy would produce at each point,
   without decoding, so the fit is cheap.
4. Fit `quality_low` / `quality_high` so the policy's weight crosses over where
   the oracle does, and report the residual.

The output is a calibrated set of thresholds plus the evidence for them, which is
what makes the branch rule defensible rather than hand-tuned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _bler(bits: np.ndarray, bits_hat: np.ndarray) -> tuple[int, int]:
    diff = np.not_equal(bits, bits_hat)
    block_error = np.any(diff, axis=-1)
    return int(block_error.sum()), int(block_error.size)


def measure(args) -> dict:
    import tensorflow as tf

    from factory6g.components.estimators.adaptive_estimator import (
        delay_spread_ratio,
        per_user_quality,
    )
    from factory6g.models.model import Model
    from factory6g.sim.config import load_config
    from factory6g.sim.stages.common import (
        bler_upper_confidence_bound,
        derive_seed,
        seed_global_rngs,
    )

    config = load_config(args.config)
    runtime = config.system_runtime_config
    ebno_values = [float(v) for v in config.monte_carlo.ebno_db_range]

    context_model = Model(config=runtime, estimator_type="ls", perfect_csi=False)
    branches = {
        name: Model(config=runtime, estimator_type=name, perfect_csi=False)
        for name in ("dft", "lmmse")
    }

    rows = []
    for ebno_db in ebno_values:
        tally = {n: [0, 0] for n in branches}
        quality, leakage = [], []

        for batch_index in range(args.batches):
            seed_global_rngs(derive_seed(args.seed, "branch-cal", ebno_db, batch_index))
            ctx = context_model.prepare_batch_context(
                batch_size=args.batch_size, ebno_db=ebno_db, include_feedback=False
            )
            # The selection statistics, measured once per batch from the LS estimate.
            x_rg, _, _ = context_model.get_transmitter().call(
                ctx.batch_size, bits=ctx.source_bits
            )
            y = context_model.get_channel().apply_frequency_response(x_rg, ctx.h_freq)
            y = y + ctx.data_noise
            h_ls, _ = context_model.get_receiver().estimate_channel(y, ctx.noise_variance)
            quality.append(float(tf.reduce_mean(per_user_quality(h_ls, ctx.noise_variance))))
            leakage.append(
                float(tf.reduce_mean(delay_spread_ratio(h_ls, int(runtime["cyclic_prefix_length"]))))
            )

            for name, model in branches.items():
                res = model.run_batch(ctx, include_details=False)
                errs, total = _bler(res["bits"], res["bits_hat"])
                tally[name][0] += errs
                tally[name][1] += total

        entry = {"ebno_db": ebno_db,
                 "quality": float(np.mean(quality)),
                 "leakage": float(np.mean(leakage))}
        for name in branches:
            errs, total = tally[name]
            entry[f"{name}_bler"] = errs / max(total, 1)
            entry[f"{name}_bler_ub"] = bler_upper_confidence_bound(errs, total, 0.95)
        # Oracle: the branch that actually decoded better here.
        entry["oracle"] = "lmmse" if entry["lmmse_bler"] < entry["dft_bler"] else "dft"
        entry["oracle_bler"] = min(entry["lmmse_bler"], entry["dft_bler"])
        entry["decisive"] = abs(entry["lmmse_bler"] - entry["dft_bler"]) > max(
            entry["lmmse_bler_ub"] - entry["lmmse_bler"],
            entry["dft_bler_ub"] - entry["dft_bler"],
        )
        rows.append(entry)
        print(
            f"  Eb/No {ebno_db:5.1f} dB | DFT {entry['dft_bler']:.3e} | "
            f"LMMSE {entry['lmmse_bler']:.3e} | oracle {entry['oracle']:<5} | "
            f"quality {entry['quality']:8.2f} leakage {entry['leakage']:.3f}",
            flush=True,
        )
    return {"rows": rows, "config": args.config}


def fit(rows: list[dict]) -> dict:
    """Choose thresholds whose weight crosses over where the oracle does.

    The policy ramps the LMMSE weight from 1 below `quality_low` to 0 above
    `quality_high`. A crossover exists only if the oracle actually prefers LMMSE
    somewhere; if DFT wins everywhere the honest fit is to disable the LMMSE
    branch, which is a result about the channel rather than a tuning failure.
    """
    decisive = [r for r in rows if r["decisive"]] or rows
    lmmse_q = [r["quality"] for r in decisive if r["oracle"] == "lmmse"]
    dft_q = [r["quality"] for r in decisive if r["oracle"] == "dft"]

    if not lmmse_q:
        return {"verdict": "dft_only",
                "note": "The oracle never prefers LMMSE. Disable the LMMSE branch "
                        "(quality_low = quality_high = 0) rather than tuning a "
                        "crossover that does not exist.",
                "quality_low": 0.0, "quality_high": 0.0}
    if not dft_q:
        return {"verdict": "lmmse_only",
                "note": "The oracle never prefers DFT on this channel.",
                "quality_low": float("inf"), "quality_high": float("inf")}

    # Crossover sits between the strongest LMMSE-preferring point and the
    # weakest DFT-preferring one.
    lo, hi = max(lmmse_q), min(dft_q)
    if lo > hi:            # interleaved — fall back to the midpoint of the overlap
        lo, hi = min(lo, hi), max(lo, hi)
    span = max(hi - lo, 1e-6)
    return {"verdict": "crossover",
            "quality_low": float(lo),
            "quality_high": float(hi),
            "crossover_quality": float((lo + hi) / 2),
            "note": f"LMMSE preferred below quality {lo:.2f}, DFT above {hi:.2f} "
                    f"(transition span {span:.2f}).",
            "leakage_reference": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/thesis/estimators_inf_s.json")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Measuring branch performance on {args.config}...")
    payload = measure(args)
    payload["fit"] = fit(payload["rows"])

    print("\nCalibration:")
    for key, value in payload["fit"].items():
        print(f"  {key}: {value}")

    oracle = [r["oracle_bler"] for r in payload["rows"]]
    dft = [r["dft_bler"] for r in payload["rows"]]
    print(f"\nOracle vs DFT-only, mean BLER: {np.mean(oracle):.4e} vs {np.mean(dft):.4e}")
    print("A negligible gap means a hard DFT branch is already near-optimal here.")

    out = Path(args.output or "adaptive_branch_calibration.json")
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
