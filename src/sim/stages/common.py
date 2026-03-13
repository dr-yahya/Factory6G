from __future__ import annotations

import math
import time
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from src.models.resource_manager import ResourceDirectives

if TYPE_CHECKING:
    from src.models.model import Model


MIN_RESOLVED_BIT_ERRORS = 30
POINT_STATUS_RESOLVED = "resolved"
POINT_STATUS_UPPER_BOUND_ONLY = "upper_bound_only"


METRIC_KEYS = (
    "ber",
    "ber_upper_confidence",
    "latency_ms",
    "throughput_bits_per_batch",
    "energy_joules_per_batch",
    "avg_power_w",
    "runtime_sec",
    "bit_errors",
    "total_bits",
    "block_errors",
    "total_blocks",
    "num_batches",
    "stop_reason",
    "point_status",
)


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
    aggregate["ber"].append(float(point["ber"]))
    aggregate["ber_upper_confidence"].append(
        ber_upper_confidence_bound(
            int(point["bit_errors"]),
            int(point["total_bits"]),
            confidence_level,
        )
    )
    aggregate["latency_ms"].append(float(point["latency_ms"]))
    aggregate["throughput_bits_per_batch"].append(float(point["throughput_bits_per_batch"]))
    aggregate["energy_joules_per_batch"].append(float(point["energy_joules_per_batch"]))
    aggregate["avg_power_w"].append(float(point["avg_power_w"]))
    aggregate["runtime_sec"].append(float(point["runtime_sec"]))
    aggregate["bit_errors"].append(float(point["bit_errors"]))
    aggregate["total_bits"].append(float(point["total_bits"]))
    aggregate["block_errors"].append(float(point["block_errors"]))
    aggregate["total_blocks"].append(float(point["total_blocks"]))
    aggregate["num_batches"].append(float(point["num_batches"]))
    aggregate["stop_reason"].append(str(point.get("stop_reason", "unknown")))
    aggregate["point_status"].append(
        str(point.get("point_status", classify_point_status(float(point["bit_errors"]))))
    )


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
