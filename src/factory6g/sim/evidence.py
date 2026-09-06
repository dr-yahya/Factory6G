"""Monte Carlo evidence budget and tail extrapolation.

The reliability regime this project targets -- URLLC in factory automation, so
residual error around 1e-5 to 1e-6 -- is far below what the configured Monte
Carlo budget can observe. With the shipped configuration a point sees at most a
few million bits and a couple of thousand codewords, which puts the smallest
resolvable BLER around 1e-3. No amount of confidence-bound plotting fixes that:
the evidence is simply not there.

This module makes the limit explicit rather than leaving it implicit in
`upper_bound_only` labels:

* `evidence_ceiling` reports what a given configuration can actually resolve.
* `check_reliability_target` says whether a stated target is reachable, and what
  budget would be needed.
* `extrapolate_bler` fits the waterfall's log-linear tail so a deep-tail claim
  can be stated as an extrapolation with an interval, rather than silently
  presented as a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# A point needs at least this many observed errors before its rate estimate is
# treated as a measurement rather than an upper bound.
MIN_RESOLVED_EVENTS = 30


@dataclass(frozen=True)
class EvidenceCeiling:
    """What a Monte Carlo configuration can and cannot resolve."""

    max_bits_per_point: int
    max_blocks_per_point: int
    min_resolvable_ber: float
    min_resolvable_bler: float
    zero_error_ber_bound: float
    zero_error_bler_bound: float

    def describe(self) -> str:
        return (
            f"Evidence ceiling per Eb/No point: {self.max_bits_per_point:,} bits / "
            f"{self.max_blocks_per_point:,} codewords. "
            f"Smallest resolvable BER ~{self.min_resolvable_ber:.2e}, "
            f"BLER ~{self.min_resolvable_bler:.2e}. "
            f"With zero observed errors the 95% bounds are "
            f"BER <= {self.zero_error_ber_bound:.2e}, BLER <= {self.zero_error_bler_bound:.2e}."
        )


def info_bits_per_codeword(
    *,
    fft_size: int,
    num_ofdm_symbols: int,
    num_pilot_symbols: int,
    num_bits_per_symbol: int,
    coderate: float,
) -> int:
    """Information bits carried by one codeword, from the numerology."""
    data_symbols = max(num_ofdm_symbols - num_pilot_symbols, 1)
    coded_bits = fft_size * data_symbols * num_bits_per_symbol
    return max(int(coded_bits * coderate), 1)


def evidence_ceiling(
    *,
    batch_size: int,
    max_batches: int,
    num_ut: int,
    num_streams_per_ut: int = 1,
    fft_size: int = 128,
    num_ofdm_symbols: int = 14,
    num_pilot_symbols: int = 2,
    num_bits_per_symbol: int = 2,
    coderate: float = 0.5,
    confidence_level: float = 0.95,
) -> EvidenceCeiling:
    """Best-case evidence a single Eb/No point can accumulate."""
    bits_per_codeword = info_bits_per_codeword(
        fft_size=fft_size,
        num_ofdm_symbols=num_ofdm_symbols,
        num_pilot_symbols=num_pilot_symbols,
        num_bits_per_symbol=num_bits_per_symbol,
        coderate=coderate,
    )
    blocks_per_batch = max(batch_size * num_ut * num_streams_per_ut, 1)
    max_blocks = blocks_per_batch * max(max_batches, 1)
    max_bits = max_blocks * bits_per_codeword

    alpha = max(1e-12, 1.0 - confidence_level)
    return EvidenceCeiling(
        max_bits_per_point=int(max_bits),
        max_blocks_per_point=int(max_blocks),
        min_resolvable_ber=MIN_RESOLVED_EVENTS / max(max_bits, 1),
        min_resolvable_bler=MIN_RESOLVED_EVENTS / max(max_blocks, 1),
        # Rule of three: the one-sided bound when nothing is observed.
        zero_error_ber_bound=-math.log(alpha) / max(max_bits, 1),
        zero_error_bler_bound=-math.log(alpha) / max(max_blocks, 1),
    )


def batches_needed_for(
    target_rate: float,
    *,
    blocks_per_batch: int,
    min_events: int = MIN_RESOLVED_EVENTS,
) -> int:
    """Batches required to observe `min_events` errors at a given error rate."""
    if target_rate <= 0.0:
        return -1
    return int(math.ceil(min_events / (target_rate * max(blocks_per_batch, 1))))


def check_reliability_target(
    target_bler: float,
    ceiling: EvidenceCeiling,
    *,
    blocks_per_batch: int,
) -> dict[str, object]:
    """Is `target_bler` reachable under this budget, and if not, what is needed?"""
    reachable = target_bler >= ceiling.min_resolvable_bler
    needed = batches_needed_for(target_bler, blocks_per_batch=blocks_per_batch)
    message = (
        f"Target BLER {target_bler:.2e} is resolvable under this budget."
        if reachable
        else (
            f"Target BLER {target_bler:.2e} is BELOW the evidence ceiling "
            f"({ceiling.min_resolvable_bler:.2e}). Observing {MIN_RESOLVED_EVENTS} block "
            f"errors at that rate needs about {needed:,} batches per Eb/No point. "
            f"Either raise monte_carlo.max_batches, enable system.graph_mode to afford "
            f"the runtime, or report the tail as an extrapolation with its interval "
            f"(see extrapolate_bler) rather than as a measurement."
        )
    )
    return {
        "target_bler": float(target_bler),
        "reachable": bool(reachable),
        "batches_needed": int(needed),
        "ceiling_bler": float(ceiling.min_resolvable_bler),
        "message": message,
    }


def extrapolate_bler(
    ebno_db: np.ndarray,
    bler: np.ndarray,
    *,
    target_ebno_db: float | np.ndarray | None = None,
    target_bler: float | None = None,
    min_points: int = 3,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Log-linear fit of the BLER waterfall, with a prediction interval.

    Above the waterfall knee log10(BLER) is close to linear in Eb/No, so a
    least-squares fit there extrapolates the deep tail. This is an
    *extrapolation*, and the returned interval is what makes it honest -- it must
    be reported as such, never merged into a measured curve.

    Provide `target_ebno_db` to predict the BLER at an operating point, or
    `target_bler` to predict the Eb/No needed to reach a reliability target.
    """
    ebno = np.asarray(ebno_db, dtype=np.float64)
    rates = np.asarray(bler, dtype=np.float64)
    usable = np.isfinite(ebno) & np.isfinite(rates) & (rates > 0.0) & (rates < 1.0)
    if usable.sum() < min_points:
        return {
            "ok": False,
            "reason": (
                f"Need at least {min_points} points with 0 < BLER < 1 to fit a tail; "
                f"got {int(usable.sum())}."
            ),
        }

    x = ebno[usable]
    y = np.log10(rates[usable])
    slope, intercept = np.polyfit(x, y, 1)

    residuals = y - (slope * x + intercept)
    dof = max(x.size - 2, 1)
    residual_std = float(np.sqrt(np.sum(residuals**2) / dof))
    x_mean = float(np.mean(x))
    sum_sq = float(np.sum((x - x_mean) ** 2)) or 1e-12

    # Normal quantile; with few points this is mildly optimistic, which the
    # caller should note alongside the interval.
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence_level) / 2.0)

    result: dict[str, object] = {
        "ok": True,
        "slope_decades_per_db": float(slope),
        "intercept": float(intercept),
        "residual_std_decades": residual_std,
        "num_points": int(x.size),
        "fit_range_db": (float(x.min()), float(x.max())),
        "method": "log-linear least squares on the waterfall tail",
    }

    if target_ebno_db is not None:
        target = np.asarray(target_ebno_db, dtype=np.float64)
        predicted_log = slope * target + intercept
        margin = z * residual_std * np.sqrt(
            1.0 + 1.0 / x.size + (target - x_mean) ** 2 / sum_sq
        )
        result["predicted_bler"] = np.power(10.0, predicted_log)
        result["predicted_bler_lower"] = np.power(10.0, predicted_log - margin)
        result["predicted_bler_upper"] = np.power(10.0, predicted_log + margin)
        result["extrapolated_beyond_data"] = bool(
            np.any(target > x.max()) or np.any(target < x.min())
        )

    if target_bler is not None and abs(slope) > 1e-12:
        required = (math.log10(target_bler) - intercept) / slope
        result["required_ebno_db"] = float(required)
        result["required_ebno_extrapolated"] = bool(required > x.max())

    return result
