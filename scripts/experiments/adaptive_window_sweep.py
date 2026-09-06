#!/usr/bin/env python3
"""Adaptive truncation window against fixed DFT and an exhaustive oracle.

Reports, per Eb/No point: the best NMSE any single window length could reach
(found by exhaustive search over 1..CP taps), what the fixed CP-length window
reaches, what the adaptive rule reaches, the mean window it chose, and both
estimators' error-variance calibration.

The oracle column is the point of this script. "Better than the fixed window"
is a weak claim on its own; "within 0.1 dB of the best window that exists" is
the one worth making, and only an exhaustive search can support it.

    python scripts/experiments/adaptive_window_sweep.py config/thesis/estimators_inf_s.json

Environment note: this project's CLAUDE.md requires scripts to run inside the
Docker container. Run it as

    docker compose run --rm --entrypoint bash -v "$PWD:/app" simulation -lc \
      "pip install -e . -q && python scripts/experiments/adaptive_window_sweep.py <config>"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sionna.phy.ofdm import RemoveNulledSubcarriers

from factory6g.components.estimators import (
    AdaptiveWindowChannelEstimator,
    DFTChannelEstimator,
)
from factory6g.models.model import Model


def nmse_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    return 10.0 * np.log10(
        np.mean(np.abs(reference - estimate) ** 2) / np.mean(np.abs(reference) ** 2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ebno-db", type=float, nargs="+", default=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    )
    args = parser.parse_args()

    system = dict(json.loads(args.config.read_text())["system"])
    # Eager, so `last_mean_taps` is readable and the oracle search stays simple.
    system["graph_mode"] = False
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    model = Model(system, estimator_type="ls")
    grid = model._rg
    ls_estimator = model._receiver._channel_estimator
    adaptive = AdaptiveWindowChannelEstimator(grid, config=system)
    fixed = DFTChannelEstimator(grid, config=system)
    remove_nulled = RemoveNulledSubcarriers(grid)

    fft_size = int(grid.fft_size)
    cp_length = int(grid.cyclic_prefix_length)
    directives = model.default_directives()

    print(
        f"{args.config.name}  fft={fft_size} cp={cp_length} "
        f"pilot_decimation={adaptive.pilot_decimation} "
        f"min_relative_gain={adaptive.min_relative_gain}"
    )
    header = ("Eb/No", "oracle", "fixedDFT", "adaptive", "gain", "taps", "cal_adapt", "cal_dft")
    print("{:>6} {:>8} {:>9} {:>9} {:>7} {:>6} {:>10} {:>9}".format(*header))

    for ebno_db in args.ebno_db:
        oracle, fixed_nmse, adaptive_nmse, taps = [], [], [], []
        cal_adaptive, cal_fixed = [], []
        for _ in range(args.repeats):
            context = model.prepare_batch_context(
                args.batch_size, ebno_db, include_feedback=False
            )
            x, _, _ = model._transmitter.call(args.batch_size, directives=directives)
            y = model._channel.apply_frequency_response(x, context.h_freq)
            y = y + context.data_noise
            h_ls, err_var_ls = ls_estimator(y, context.noise_variance)

            truth = remove_nulled(context.h_freq).numpy()
            delay = np.fft.ifft(h_ls.numpy(), axis=-1)
            oracle.append(
                min(
                    nmse_db(truth, np.fft.fft(delay * (np.arange(fft_size) < length), axis=-1))
                    for length in range(1, cp_length + 1)
                )
            )

            h_adaptive, declared_adaptive = adaptive.estimate_from_ls(h_ls, err_var_ls)
            h_fixed, declared_fixed = fixed.estimate_from_ls(h_ls, err_var_ls)
            adaptive_nmse.append(nmse_db(truth, h_adaptive.numpy()))
            fixed_nmse.append(nmse_db(truth, h_fixed.numpy()))
            taps.append(adaptive.last_mean_taps)
            cal_adaptive.append(
                float(tf.reduce_mean(declared_adaptive))
                / np.mean(np.abs(truth - h_adaptive.numpy()) ** 2)
            )
            cal_fixed.append(
                float(tf.reduce_mean(declared_fixed))
                / np.mean(np.abs(truth - h_fixed.numpy()) ** 2)
            )

        print(
            "{:6.1f} {:8.2f} {:9.2f} {:9.2f} {:+7.2f} {:6.1f} {:10.3f} {:9.3f}".format(
                ebno_db,
                float(np.mean(oracle)),
                float(np.mean(fixed_nmse)),
                float(np.mean(adaptive_nmse)),
                float(np.mean(fixed_nmse) - np.mean(adaptive_nmse)),
                float(np.mean(taps)),
                float(np.mean(cal_adaptive)),
                float(np.mean(cal_fixed)),
            )
        )


if __name__ == "__main__":
    main()
