"""Quantify the high-SNR BER floor caused by LLR clipping.

Background
----------
The receiver used to clip demapper LLRs hard at +/-20, justified by a diagnostic
noting that "27.5% of LLRs have |LLR| > 50". At high Eb/No a large share of LLRs
are legitimately large, and saturating them discards exactly the reliability
information belief propagation needs to correct the few weak bits -- producing an
error floor precisely where the waterfall should be steepest. That is the
suspected origin of the "TR 38.901 BER floor" recorded in the March and May 2026
weekly reports.

This experiment measures the effect directly.

Method
------
For every (Eb/No point, Monte Carlo batch) one channel realisation, noise draw
and source-bit set is generated and then decoded by receivers that differ *only*
in their LLR clip. Common random numbers make this a paired comparison, so the
difference between clip settings is not contaminated by Monte Carlo variance,
and a paired bootstrap gives a confidence interval on the difference itself.

Usage
-----
    python scripts/experiments/llr_clip_floor.py --channel rayleigh --batches 80
    python scripts/experiments/llr_clip_floor.py --channel tr38901 --batches 80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


def _clip_label(clip: float | None) -> str:
    return "none" if clip is None else f"{clip:g}"


def run_experiment(args: argparse.Namespace) -> dict:
    from factory6g.models.model import Model
    from factory6g.sim.stages.common import (
        bler_upper_confidence_bound,
        derive_seed,
        paired_bootstrap_ci,
        seed_global_rngs,
        zero_error_upper_bound,
    )

    ebno_values = [
        float(v)
        for v in np.arange(args.ebno_min, args.ebno_max + 1e-9, args.ebno_step)
    ]
    clips: list[float | None] = [20.0, 200.0, None]

    base_config = {
        "num_ut": args.num_ut,
        "num_bs_ant": args.num_bs_ant,
        "num_ut_ant": 1,
        "fft_size": args.fft_size,
        "num_ofdm_symbols": 14,
        "subcarrier_spacing": 30e3,
        "cyclic_prefix_length": 20,
        "pilot_ofdm_symbol_indices": [2, 11],
        "num_bits_per_symbol": 2,
        "coderate": 0.5,
        "num_decoding_iter": args.decoder_iterations,
        "channel_model_type": args.channel,
        "scenario": "umi",
        "direction": "uplink",
        "o2i_model": "low",
        "enable_pathloss": False,
        "enable_shadow_fading": False,
        "min_ut_velocity": 0.0,
        "max_ut_velocity": 0.0,
        "carrier_frequency": 3.5e9,
        "tx_pattern": "tr38901",
        "tx_polarization": "cross",
        "rx_pattern": "iso",
        "rx_polarization": "V",
        "antenna_spacing": 0.5,
        "graph_mode": args.graph_mode,
    }

    # One model per clip setting. They share everything else, so a shared batch
    # context isolates the clip as the only difference.
    # `perfect` is not an estimator but an idealisation: it isolates whether a
    # floor comes from channel-estimation error rather than from the demapper.
    perfect_csi = args.estimator.lower() == "perfect"
    models = {
        _clip_label(clip): Model(
            config={**base_config, "llr_clip": clip},
            estimator_type=args.estimator,
            perfect_csi=perfect_csi,
        )
        for clip in clips
    }
    context_model = models[_clip_label(clips[0])]

    results: dict[str, dict] = {
        label: {"bit_errors": [], "total_bits": [], "block_errors": [], "total_blocks": []}
        for label in models
    }
    per_batch_ber: dict[str, list[list[float]]] = {label: [] for label in models}

    start = time.perf_counter()
    for point_index, ebno_db in enumerate(ebno_values):
        totals = {
            label: {"bit_errors": 0, "total_bits": 0, "block_errors": 0, "total_blocks": 0}
            for label in models
        }
        batch_ber = {label: [] for label in models}

        for batch_index in range(args.batches):
            seed_global_rngs(derive_seed(args.seed, "llr_clip", ebno_db, batch_index))
            context = context_model.prepare_batch_context(
                batch_size=args.batch_size, ebno_db=ebno_db, include_feedback=False
            )
            for label, model in models.items():
                result = model.run_batch(context, include_details=False)
                diff = np.not_equal(result["bits"], result["bits_hat"])
                errors = int(diff.sum())
                totals[label]["bit_errors"] += errors
                totals[label]["total_bits"] += int(diff.size)
                totals[label]["block_errors"] += int(np.any(diff, axis=-1).sum())
                totals[label]["total_blocks"] += int(np.any(diff, axis=-1).size)
                batch_ber[label].append(errors / max(diff.size, 1))

        for label in models:
            for key, value in totals[label].items():
                results[label][key].append(value)
            per_batch_ber[label].append(batch_ber[label])

        line = f"Eb/No {ebno_db:5.1f} dB |"
        for label in models:
            bits = totals[label]["total_bits"]
            errors = totals[label]["bit_errors"]
            ber = errors / max(bits, 1)
            shown = f"{ber:.3e}" if errors else f"<{zero_error_upper_bound(bits, 0.95):.1e}"
            line += f"  clip {label:>4}: BER {shown}"
        print(f"{line}   [{time.perf_counter() - start:6.1f}s]", flush=True)

    # Paired comparison of each clip setting against the historical +/-20.
    comparisons = {}
    for label in models:
        if label == "20":
            continue
        rows = []
        for point_index, ebno_db in enumerate(ebno_values):
            differences = [
                a - b
                for a, b in zip(per_batch_ber[label][point_index], per_batch_ber["20"][point_index])
            ]
            mean, lower, upper = paired_bootstrap_ci(
                differences, seed=derive_seed("clip", label, point_index)
            )
            rows.append(
                {
                    "ebno_db": ebno_db,
                    "mean_ber_delta": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "significant": bool(np.isfinite(lower) and (lower > 0.0 or upper < 0.0)),
                }
            )
        comparisons[label] = rows

    payload = {
        "channel": args.channel,
        "estimator": args.estimator,
        "ebno_db": ebno_values,
        "batches_per_point": args.batches,
        "batch_size": args.batch_size,
        "config": base_config,
        "results": {
            label: {
                **data,
                "ber": [
                    e / max(b, 1)
                    for e, b in zip(data["bit_errors"], data["total_bits"])
                ],
                "bler": [
                    e / max(b, 1)
                    for e, b in zip(data["block_errors"], data["total_blocks"])
                ],
                "bler_upper_confidence": [
                    bler_upper_confidence_bound(e, b, 0.95)
                    for e, b in zip(data["block_errors"], data["total_blocks"])
                ],
            }
            for label, data in results.items()
        },
        "paired_vs_clip20": comparisons,
        "elapsed_sec": time.perf_counter() - start,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="rayleigh", choices=["rayleigh", "tr38901", "rician"])
    parser.add_argument("--estimator", default="ls")
    parser.add_argument("--ebno-min", type=float, default=0.0)
    parser.add_argument("--ebno-max", type=float, default=20.0)
    parser.add_argument("--ebno-step", type=float, default=2.0)
    parser.add_argument("--batches", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-ut", type=int, default=4)
    parser.add_argument("--num-bs-ant", type=int, default=8)
    parser.add_argument("--fft-size", type=int, default=128)
    parser.add_argument("--decoder-iterations", type=int, default=20)
    parser.add_argument("--graph-mode", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Where to write the JSON payload")
    args = parser.parse_args()

    payload = run_experiment(args)

    output = Path(args.output or f"llr_clip_floor_{args.channel}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
