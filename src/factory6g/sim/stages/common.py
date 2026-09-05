from __future__ import annotations

import hashlib
import math
import time
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
from scipy.special import betaincinv

from factory6g.models.resource_manager import ResourceDirectives

if TYPE_CHECKING:
    from factory6g.models.model import Model


MIN_RESOLVED_BIT_ERRORS = 30


def fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
POINT_STATUS_RESOLVED = "resolved"
POINT_STATUS_UPPER_BOUND_ONLY = "upper_bound_only"


METRIC_KEYS = (
    # Reliability. BLER is the headline for URLLC and is the metric whose
    # confidence interval is statistically defensible (codewords are close to
    # independent; bits within a codeword are not).
    "bler",
    "bler_upper_confidence",
    "worst_user_bler",
    "ber",
    "ber_upper_confidence",
    # Latency, as a distribution rather than a constant.
    "latency_ms",
    "latency_p99_ms",
    "latency_p999_ms",
    "harq_rounds_mean",
    # Throughput / energy / fairness.
    "throughput_bits_per_batch",
    "energy_joules_per_batch",
    "avg_power_w",
    "radiated_power_w",
    "jains_index",
    "num_scheduled_users",
    # Estimator accuracy.
    "nmse_db",
    # Declared error variance divided by the error actually made. 1.0 is honest;
    # below 1.0 means the estimator understates its own error, which inflates its
    # LLRs through the equalizer and flatters its BER relative to more honest
    # estimators. Reported so that confound can never hide again.
    "err_var_calibration",
    # Evidence bookkeeping.
    "runtime_sec",
    "bit_errors",
    "total_bits",
    "block_errors",
    "total_blocks",
    "num_batches",
    "stop_reason",
    "point_status",
)


class PointAccumulator:
    """Monte Carlo accumulator for one (method, Eb/No) point.

    Shared by the estimator and resource-manager stages so both emit the same
    metric set, and so per-batch samples are retained for the paired analysis
    that the common-random-numbers design makes possible.
    """

    def __init__(self, num_ut: int = 0) -> None:
        self.errors = 0
        self.bits = 0
        self.block_errors = 0
        self.blocks = 0
        self.throughput = 0.0
        self.latency_sec_sum = 0.0
        self.energy_joules = 0.0
        self.radiated_power_sum = 0.0
        self.scheduled_users_sum = 0.0
        self.runtime_sec = 0.0
        self.num_batches = 0
        self.done = False
        self.stop_reason: str | None = None
        # Delivery-round histogram: index r-1 counts codewords delivered in round
        # r; the final entry counts codewords never delivered. Compact, and gives
        # exact latency percentiles without storing every sample.
        self.round_histogram: dict[int, int] = {}
        self.failed_blocks = 0
        # NMSE accumulators (error power / signal power).
        self.nmse_error_power = 0.0
        self.nmse_signal_power = 0.0
        self.nmse_elements = 0
        # Declared-versus-actual estimation error, for the calibration metric.
        self.declared_err_var_sum = 0.0
        self.declared_err_var_batches = 0
        # Per-user reliability, for worst-user BLER and fairness.
        self.user_block_errors = np.zeros(max(num_ut, 1), dtype=np.int64)
        self.user_blocks = np.zeros(max(num_ut, 1), dtype=np.int64)
        self.user_delivered_bits = np.zeros(max(num_ut, 1), dtype=np.float64)
        # Per-batch samples, retained for paired bootstrap comparisons.
        self.batch_block_errors: list[int] = []
        self.batch_blocks: list[int] = []

    # -- serialisation, so a resumed run continues rather than restarting cold --
    def to_dict(self) -> dict[str, Any]:
        payload = {
            key: value
            for key, value in self.__dict__.items()
            if not isinstance(value, np.ndarray)
        }
        payload["round_histogram"] = {str(k): v for k, v in self.round_histogram.items()}
        payload["user_block_errors"] = self.user_block_errors.tolist()
        payload["user_blocks"] = self.user_blocks.tolist()
        payload["user_delivered_bits"] = self.user_delivered_bits.tolist()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PointAccumulator":
        acc = cls()
        for key, value in payload.items():
            if key in {"user_block_errors", "user_blocks", "user_delivered_bits"}:
                setattr(acc, key, np.asarray(value))
            elif key == "round_histogram":
                acc.round_histogram = {int(k): int(v) for k, v in (value or {}).items()}
            else:
                setattr(acc, key, value)
        return acc

    def _ensure_user_arrays(self, num_ut: int) -> None:
        if self.user_blocks.shape[0] >= num_ut:
            return
        pad = num_ut - self.user_blocks.shape[0]
        self.user_block_errors = np.pad(self.user_block_errors, (0, pad))
        self.user_blocks = np.pad(self.user_blocks, (0, pad))
        self.user_delivered_bits = np.pad(self.user_delivered_bits, (0, pad))

    def add_batch(
        self,
        result: dict[str, Any],
        *,
        ut_mask: Sequence[int] | None,
        elapsed_sec: float,
        num_ut: int,
    ) -> dict[str, int]:
        """Fold one batch result into the accumulator.

        Returns the batch's error statistics so callers can apply the stop rule.
        """
        self._ensure_user_arrays(num_ut)
        bits = result["bits"]
        bits_hat = result["bits_hat"]

        stats = extract_error_stats(bits, bits_hat, ut_mask=ut_mask)
        self.errors += stats["bit_errors"]
        self.bits += stats["total_bits"]
        self.block_errors += stats["block_errors"]
        self.blocks += stats["total_blocks"]
        self.throughput += max(0.0, stats["total_bits"] - stats["bit_errors"])
        self.runtime_sec += float(elapsed_sec)
        self.num_batches += 1
        self.batch_block_errors.append(int(stats["block_errors"]))
        self.batch_blocks.append(int(stats["total_blocks"]))

        # Per-user reliability over the transmitter axis.
        diff = np.not_equal(bits, bits_hat)
        per_block_error = np.any(diff, axis=-1)  # [batch, num_tx, num_streams]
        active = (
            np.ones(per_block_error.shape[1], dtype=bool)
            if ut_mask is None
            else np.asarray(list(ut_mask), dtype=bool)
        )
        bits_per_block = diff.shape[-1]
        for user in range(min(per_block_error.shape[1], num_ut)):
            if not active[user]:
                continue
            user_errors = int(per_block_error[:, user, ...].sum())
            user_total = int(per_block_error[:, user, ...].size)
            self.user_block_errors[user] += user_errors
            self.user_blocks[user] += user_total
            self.user_delivered_bits[user] += float((user_total - user_errors) * bits_per_block)

        # Latency distribution, from the HARQ delivery rounds.
        delivery = result.get("delivery_round")
        scheduled = result.get("scheduled_block_mask")
        if delivery is not None:
            selected = delivery if scheduled is None else delivery[scheduled]
            for round_index in np.asarray(selected).ravel():
                index = int(round_index)
                if index <= 0:
                    self.failed_blocks += 1
                else:
                    self.round_histogram[index] = self.round_histogram.get(index, 0) + 1

        self.latency_sec_sum += float(result.get("latency_sec", 0.0))
        self.energy_joules += float(result.get("energy_joules", 0.0))
        self.radiated_power_sum += float(result.get("radiated_power_w", 0.0))
        self.scheduled_users_sum += float(np.sum(active[:num_ut]))

        # Channel estimation accuracy.
        h_true = result.get("channel")
        h_hat = result.get("channel_hat")
        if h_true is not None and h_hat is not None:
            true_arr = np.asarray(h_true)
            hat_arr = np.asarray(h_hat)
            if true_arr.shape == hat_arr.shape:
                self.nmse_error_power += float(np.sum(np.abs(true_arr - hat_arr) ** 2))
                self.nmse_signal_power += float(np.sum(np.abs(true_arr) ** 2))
                self.nmse_elements += int(true_arr.size)

        declared = result.get("declared_err_var")
        if declared is not None and np.isfinite(declared):
            self.declared_err_var_sum += float(declared)
            self.declared_err_var_batches += 1

        return stats

    def _latency_percentile(self, quantile: float, slot_duration_sec: float, max_rounds: int) -> float:
        """Exact percentile of per-codeword latency from the round histogram."""
        total = sum(self.round_histogram.values()) + self.failed_blocks
        if total == 0:
            return 0.0
        target = quantile * total
        cumulative = 0
        for round_index in sorted(self.round_histogram):
            cumulative += self.round_histogram[round_index]
            if cumulative >= target:
                return round_index * slot_duration_sec * 1000.0
        # Remaining mass is codewords that never decoded: charge the full budget.
        return max_rounds * slot_duration_sec * 1000.0

    def finalize(
        self,
        *,
        confidence_level: float,
        slot_duration_sec: float,
        max_harq_rounds: int,
    ) -> dict[str, Any]:
        safe_batches = max(1, int(self.num_batches))
        avg_latency_sec = self.latency_sec_sum / safe_batches
        avg_energy = self.energy_joules / safe_batches
        # Power is now energy over the physical slot time, not over wall clock.
        avg_power = avg_energy / max(slot_duration_sec, 1e-12)

        served_users = np.where(self.user_blocks > 0)[0]
        per_user_bler = (
            self.user_block_errors[served_users] / np.maximum(self.user_blocks[served_users], 1)
            if served_users.size
            else np.array([])
        )
        delivered = self.user_delivered_bits[served_users] if served_users.size else np.array([])

        total_rounds = sum(r * c for r, c in self.round_histogram.items())
        total_delivered_blocks = sum(self.round_histogram.values())
        harq_rounds_mean = (
            (total_rounds + self.failed_blocks * max_harq_rounds)
            / max(total_delivered_blocks + self.failed_blocks, 1)
        )

        nmse = (
            10.0 * math.log10(self.nmse_error_power / self.nmse_signal_power)
            if self.nmse_signal_power > 0.0 and self.nmse_error_power > 0.0
            else float("nan")
        )

        # Calibration: declared error variance against the error actually made,
        # both averaged per resource element.
        calibration = float("nan")
        if self.declared_err_var_batches > 0 and self.nmse_error_power > 0.0:
            elements = self.nmse_elements or 0
            if elements > 0:
                measured_per_element = self.nmse_error_power / elements
                declared_mean = self.declared_err_var_sum / self.declared_err_var_batches
                calibration = declared_mean / measured_per_element

        return {
            "bler": (self.block_errors / self.blocks) if self.blocks > 0 else 0.0,
            "bler_upper_confidence": bler_upper_confidence_bound(
                self.block_errors, self.blocks, confidence_level
            ),
            "worst_user_bler": float(per_user_bler.max()) if per_user_bler.size else 0.0,
            "ber": (self.errors / self.bits) if self.bits > 0 else 0.0,
            "latency_ms": avg_latency_sec * 1000.0,
            "latency_p99_ms": self._latency_percentile(0.99, slot_duration_sec, max_harq_rounds),
            "latency_p999_ms": self._latency_percentile(0.999, slot_duration_sec, max_harq_rounds),
            "harq_rounds_mean": float(harq_rounds_mean),
            "throughput_bits_per_batch": self.throughput / safe_batches,
            "energy_joules_per_batch": avg_energy,
            "avg_power_w": avg_power,
            "radiated_power_w": self.radiated_power_sum / safe_batches,
            "jains_index": jains_fairness_index(delivered),
            "num_scheduled_users": self.scheduled_users_sum / safe_batches,
            "nmse_db": nmse,
            "err_var_calibration": calibration,
            "runtime_sec": float(self.runtime_sec),
            "bit_errors": float(self.errors),
            "total_bits": float(self.bits),
            "block_errors": float(self.block_errors),
            "total_blocks": float(self.blocks),
            "num_batches": float(self.num_batches),
            "stop_reason": str(self.stop_reason or "max_batches"),
            "point_status": classify_point_status(float(self.errors)),
            "batch_block_errors": list(self.batch_block_errors),
            "batch_blocks": list(self.batch_blocks),
        }


def initialize_stage_metrics(methods: list[str]) -> dict[str, dict[str, list[Any]]]:
    return {method: {key: [] for key in METRIC_KEYS} for method in methods}


def classify_point_status(bit_errors: float) -> str:
    if float(bit_errors) < float(MIN_RESOLVED_BIT_ERRORS):
        return POINT_STATUS_UPPER_BOUND_ONLY
    return POINT_STATUS_RESOLVED


def transmitted_ut_mask(
    directives: ResourceDirectives | None,
    num_ut: int,
) -> list[int]:
    base_active = np.ones(max(num_ut, 1), dtype=bool)
    power_active = np.ones(max(num_ut, 1), dtype=bool)

    if directives is not None and directives.active_ut_mask is not None:
        raw = np.asarray(directives.active_ut_mask, dtype=np.float32).reshape(-1)
        if raw.size < num_ut:
            raise ValueError(
                f"active_ut_mask length {raw.size} is smaller than num_ut={num_ut}."
            )
        base_active = raw[:num_ut] > 0.0
    if directives is not None and directives.per_ut_power is not None:
        raw = np.asarray(directives.per_ut_power, dtype=np.float32).reshape(-1)
        if raw.size < num_ut:
            raise ValueError(
                f"per_ut_power length {raw.size} is smaller than num_ut={num_ut}."
            )
        power_active = raw[:num_ut] > 0.0

    return (base_active & power_active).astype(np.int32).tolist()


def extract_error_stats(
    bits: np.ndarray,
    bits_hat: np.ndarray,
    ut_mask: Sequence[int] | None = None,
) -> dict[str, int]:
    diff = np.not_equal(bits, bits_hat)
    if ut_mask is not None:
        if diff.ndim < 2:
            raise ValueError(
                f"Expected at least 2-D bit tensor with user axis at dim=1, got shape {diff.shape}."
            )
        mask = np.asarray(list(ut_mask), dtype=bool).reshape(-1)
        if mask.size != diff.shape[1]:
            raise ValueError(
                f"ut_mask length {mask.size} does not match user axis size {diff.shape[1]}."
            )
        diff = diff[:, mask, ...]

    block_error_mask = np.any(diff, axis=-1)
    return {
        "bit_errors": int(diff.sum()),
        "total_bits": int(diff.size),
        "block_errors": int(block_error_mask.sum()),
        "total_blocks": int(block_error_mask.size),
    }


def clopper_pearson_upper(successes: int, trials: int, confidence_level: float) -> float:
    """Exact one-sided upper confidence bound for a binomial proportion.

    Used for BLER, where the trials (codewords) are close to independent. Unlike
    the Wilson interval used for BER, this is exact and stays valid at zero
    observed errors.
    """
    if trials <= 0:
        return 1.0
    if successes >= trials:
        return 1.0
    alpha = max(1e-12, 1.0 - float(confidence_level))
    # Upper bound = Beta quantile with shape (k+1, n-k)
    return float(betaincinv(successes + 1, trials - successes, 1.0 - alpha))


def bler_upper_confidence_bound(
    block_errors: int, total_blocks: int, confidence_level: float
) -> float:
    """One-sided upper confidence bound on the block error rate."""
    return clopper_pearson_upper(int(block_errors), int(total_blocks), confidence_level)


def jains_fairness_index(values: Sequence[float]) -> float:
    """Jain's fairness index over per-user allocations. 1.0 is perfectly fair."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    total = float(np.sum(arr))
    if abs(total) < 1e-15:
        return 1.0
    return float(total**2 / (arr.size * float(np.sum(arr**2))))


def channel_nmse_db(h_true: np.ndarray, h_hat: np.ndarray) -> float:
    """Normalized MSE of a channel estimate, in dB.

    NMSE = E|h - h_hat|^2 / E|h|^2. This is the standard channel-estimation
    accuracy metric and isolates estimator quality from the equalizer and the
    LDPC decoder, which BER alone cannot.
    """
    true_arr = np.asarray(h_true)
    hat_arr = np.asarray(h_hat)
    if true_arr.shape != hat_arr.shape:
        return float("nan")
    signal = float(np.mean(np.abs(true_arr) ** 2))
    if signal <= 0.0:
        return float("nan")
    error = float(np.mean(np.abs(true_arr - hat_arr) ** 2))
    if error <= 0.0:
        return float("-inf")
    return float(10.0 * math.log10(error / signal))


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    confidence_level: float = 0.95,
    num_resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of paired per-batch differences.

    Every method sees the identical channel, noise and source bits at a given
    (batch, Eb/No) -- common random numbers. Comparing the *paired* differences
    rather than the marginal curves removes the shared Monte Carlo variance and
    gives a far tighter interval, which is what turns "the DRL curve looks lower"
    into a defensible quantitative claim.

    Returns (mean, lower, upper).
    """
    values = np.asarray([v for v in differences if np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(values))
    if values.size == 1:
        return (mean, mean, mean)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(int(num_resamples), values.size))
    means = values[idx].mean(axis=1)
    alpha = 1.0 - float(confidence_level)
    lower = float(np.percentile(means, 100.0 * alpha / 2.0))
    upper = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return (mean, lower, upper)


def slot_duration_seconds(config: dict[str, Any]) -> float:
    """Duration of one transmission time interval, in seconds.

    Derived from the numerology alone so stages do not have to reach into a
    Model instance for it.
    """
    subcarrier_spacing = float(config.get("subcarrier_spacing", 30e3))
    fft_size = float(config.get("fft_size", 512))
    cyclic_prefix_length = float(config.get("cyclic_prefix_length", 20))
    num_ofdm_symbols = float(config.get("num_ofdm_symbols", 14))
    symbol_duration = 1.0 / max(subcarrier_spacing, 1e-12)
    cyclic_prefix_ratio = cyclic_prefix_length / max(fft_size, 1.0)
    return symbol_duration * (1.0 + cyclic_prefix_ratio) * num_ofdm_symbols


def derive_seed(*parts: Any) -> int:
    """Deterministic 32-bit seed derived from the given parts.

    A single global seed means one shared RNG stream for every method and every
    Eb/No point, so adding or removing a method from the enabled list changes
    every channel realisation and runs stop being comparable. Deriving the seed
    from (stage, method-independent point identity, batch index) instead makes a
    point reproducible in isolation, comparable across runs with different method
    lists, and exactly resumable.
    """
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def seed_global_rngs(seed: int) -> None:
    """Reseed Python, NumPy, TensorFlow and Sionna from one derived value."""
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:  # pragma: no cover - TF is always present at runtime
        pass
    try:
        import sionna

        sionna.phy.config.seed = seed
    except (ImportError, AttributeError):  # pragma: no cover
        pass


def zero_error_upper_bound(total_bits: int, confidence_level: float) -> float:
    alpha = max(1e-12, 1.0 - confidence_level)
    return -math.log(alpha) / max(total_bits, 1)


def ber_upper_confidence_bound(bit_errors: int, total_bits: int, confidence_level: float) -> float:
    if total_bits <= 0:
        return 1.0
    if bit_errors <= 0:
        return zero_error_upper_bound(total_bits, confidence_level)
    p_hat = bit_errors / total_bits
    z = NormalDist().inv_cdf(max(1e-12, min(1.0 - 1e-12, confidence_level)))
    denom = 1.0 + (z * z / total_bits)
    center = p_hat + (z * z / (2.0 * total_bits))
    radius = z * math.sqrt(
        (p_hat * (1.0 - p_hat) / total_bits) + ((z * z) / (4.0 * total_bits * total_bits))
    )
    return min(1.0, (center + radius) / denom)


def mc_stop_reason(
    *,
    num_batches: int,
    total_bits: int,
    total_block_errors: int,
    target_block_errors: int | None,
    total_bit_errors: int,
    target_ber: float | None,
    stop_policy: str,
    confidence_level: float,
    min_batches: int,
    min_total_bits: int,
) -> str | None:
    if num_batches < min_batches or total_bits < min_total_bits:
        return None
    if stop_policy == "sweep":
        if target_block_errors is not None and total_block_errors >= target_block_errors:
            return "target_block_errors"
        return None
    if stop_policy != "threshold":
        raise ValueError(f"Unsupported monte carlo stop policy '{stop_policy}'.")
    if target_block_errors is None and target_ber is None:
        return "min_evidence"
    if target_block_errors is not None and total_block_errors >= target_block_errors:
        return "target_block_errors"
    if target_ber is not None:
        ber_upper = ber_upper_confidence_bound(total_bit_errors, total_bits, confidence_level)
        if ber_upper <= target_ber:
            return "target_ber"
    return None


def append_point_metrics(
    aggregate: dict[str, list[Any]],
    *,
    confidence_level: float,
    point: dict[str, Any],
) -> None:
    """Append one finalized point to a method's metric arrays."""
    aggregate["ber_upper_confidence"].append(
        ber_upper_confidence_bound(
            int(point["bit_errors"]),
            int(point["total_bits"]),
            confidence_level,
        )
    )
    for key in METRIC_KEYS:
        if key == "ber_upper_confidence":
            continue
        value = point.get(key)
        if key == "point_status":
            aggregate[key].append(
                str(value)
                if value is not None
                else classify_point_status(float(point.get("bit_errors", 0.0)))
            )
        elif key == "stop_reason":
            aggregate[key].append(str(value) if value is not None else "unknown")
        else:
            aggregate[key].append(float(value) if value is not None else float("nan"))


def compare_methods_paired(
    per_method_points: dict[str, list[dict[str, Any]]],
    *,
    reference: str,
    confidence_level: float = 0.95,
    num_resamples: int = 10000,
) -> dict[str, list[dict[str, Any]]]:
    """Paired per-batch BLER comparison of each method against a reference.

    Every method sees the identical channel realisation, noise draw and source
    bits at a given (batch, Eb/No). Differencing per batch before averaging
    removes that shared variance, so the resulting interval is much tighter than
    comparing the two marginal curves -- and it is what lets a claim be stated as
    "improves BLER by X (95% CI [a, b])" rather than "the curve is lower".

    Only the batch prefix both methods actually ran is used, because the
    early-stopping rule retires points at different batch counts per method.
    """
    if reference not in per_method_points:
        return {}

    reference_points = per_method_points[reference]
    comparisons: dict[str, list[dict[str, Any]]] = {}
    for method, points in per_method_points.items():
        if method == reference:
            continue
        rows: list[dict[str, Any]] = []
        for index, point in enumerate(points):
            if index >= len(reference_points):
                break
            ref = reference_points[index]
            method_errors = point.get("batch_block_errors", [])
            method_blocks = point.get("batch_blocks", [])
            ref_errors = ref.get("batch_block_errors", [])
            ref_blocks = ref.get("batch_blocks", [])
            paired_len = min(len(method_errors), len(ref_errors))
            if paired_len == 0:
                rows.append({"num_paired_batches": 0})
                continue
            differences = [
                (method_errors[i] / max(method_blocks[i], 1))
                - (ref_errors[i] / max(ref_blocks[i], 1))
                for i in range(paired_len)
            ]
            mean, lower, upper = paired_bootstrap_ci(
                differences,
                confidence_level=confidence_level,
                num_resamples=num_resamples,
                seed=derive_seed("paired", method, reference, index),
            )
            rows.append(
                {
                    "num_paired_batches": paired_len,
                    "mean_bler_delta": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    # A CI excluding zero is a statistically significant difference.
                    "significant": bool(np.isfinite(lower) and np.isfinite(upper) and (lower > 0.0 or upper < 0.0)),
                }
            )
        comparisons[method] = rows
    return comparisons


def run_monte_carlo_point(
    *,
    model: "Model",
    batch_size: int,
    ebno_db: float,
    min_batches: int,
    max_mc_batches: int,
    target_block_errors: int | None,
    target_ber: float | None,
    stop_policy: str,
    confidence_level: float,
    min_total_bits: int,
    include_feedback: bool,
    directives_fn: Callable[[Any], ResourceDirectives | None] | None = None,
) -> dict[str, Any]:
    total_errors = 0
    total_bits = 0
    total_block_errors = 0
    total_blocks = 0
    total_throughput = 0.0
    total_latency_sec = 0.0
    total_energy_joules = 0.0
    num_batches_run = 0

    runtime_start = time.perf_counter()
    final_stop_reason = "max_batches"
    for _ in range(max_mc_batches):
        context = model.prepare_batch_context(
            batch_size=batch_size,
            ebno_db=float(ebno_db),
            include_feedback=include_feedback,
        )
        directives = directives_fn(context) if directives_fn is not None else None
        res = model.run_batch(context, directives=directives, include_details=True)
        num_batches_run += 1

        batch_stats = extract_error_stats(res["bits"], res["bits_hat"])
        total_errors += batch_stats["bit_errors"]
        total_bits += batch_stats["total_bits"]
        total_block_errors += batch_stats["block_errors"]
        total_blocks += batch_stats["total_blocks"]
        total_throughput += max(0.0, batch_stats["total_bits"] - batch_stats["bit_errors"])
        total_latency_sec += float(res.get("runtime_latency_sec", 0.0))
        total_energy_joules += float(res.get("energy_joules", 0.0))

        stop_reason = mc_stop_reason(
            num_batches=num_batches_run,
            total_bits=total_bits,
            total_block_errors=total_block_errors,
            target_block_errors=target_block_errors,
            total_bit_errors=total_errors,
            target_ber=target_ber,
            stop_policy=stop_policy,
            confidence_level=confidence_level,
            min_batches=min_batches,
            min_total_bits=min_total_bits,
        )
        if stop_reason is not None:
            final_stop_reason = stop_reason
            break

    runtime_sec = time.perf_counter() - runtime_start
    safe_batches = max(num_batches_run, 1)
    avg_latency_sec = total_latency_sec / safe_batches
    avg_energy_joules = total_energy_joules / safe_batches
    avg_power_w = avg_energy_joules / max(avg_latency_sec, 1e-12)
    return {
        "ber": (total_errors / total_bits) if total_bits > 0 else 0.0,
        "latency_ms": avg_latency_sec * 1000.0,
        "throughput_bits_per_batch": total_throughput / safe_batches,
        "energy_joules_per_batch": avg_energy_joules,
        "avg_power_w": avg_power_w,
        "runtime_sec": runtime_sec,
        "bit_errors": float(total_errors),
        "total_bits": float(total_bits),
        "block_errors": float(total_block_errors),
        "total_blocks": float(total_blocks),
        "num_batches": float(num_batches_run),
        "stop_reason": final_stop_reason,
        "point_status": classify_point_status(float(total_errors)),
    }


def resolve_kwargs(kwargs_config: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    direct = kwargs_config.get(key)
    if isinstance(direct, dict):
        return direct
    return {}
