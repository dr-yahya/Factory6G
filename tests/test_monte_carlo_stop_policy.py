from __future__ import annotations

import numpy as np

from factory6g.sim.stages.common import (
    POINT_STATUS_RESOLVED,
    POINT_STATUS_UPPER_BOUND_ONLY,
    append_point_metrics,
    initialize_stage_metrics,
    mc_stop_reason,
    run_monte_carlo_point,
)


class _FakeModel:
    def __init__(self, bit_errors_per_batch: list[int], *, total_bits: int = 32):
        self._bit_errors_per_batch = list(bit_errors_per_batch)
        self._total_bits = int(total_bits)

    def prepare_batch_context(self, batch_size: int, ebno_db: float, include_feedback: bool):
        return {
            "batch_size": batch_size,
            "ebno_db": ebno_db,
            "include_feedback": include_feedback,
        }

    def run_batch(self, batch_context, directives=None, include_details: bool = True):
        bit_errors = self._bit_errors_per_batch.pop(0)
        bits = np.zeros((1, 1, 1, self._total_bits), dtype=np.int32)
        bits_hat = bits.copy()
        bits_hat.reshape(-1)[:bit_errors] = 1
        return {
            "bits": bits,
            "bits_hat": bits_hat,
            "runtime_latency_sec": 1e-4,
            "energy_joules": 1e-6,
        }


def test_sweep_policy_never_stops_on_target_ber():
    reason = mc_stop_reason(
        num_batches=16,
        total_bits=2_000_000,
        total_block_errors=0,
        target_block_errors=1_000,
        total_bit_errors=0,
        target_ber=1e-5,
        stop_policy="sweep",
        confidence_level=0.95,
        min_batches=4,
        min_total_bits=1_000_000,
    )
    assert reason is None


def test_threshold_policy_can_stop_on_target_ber():
    reason = mc_stop_reason(
        num_batches=16,
        total_bits=2_000_000,
        total_block_errors=0,
        target_block_errors=1_000,
        total_bit_errors=0,
        target_ber=1e-5,
        stop_policy="threshold",
        confidence_level=0.95,
        min_batches=4,
        min_total_bits=1_000_000,
    )
    assert reason == "target_ber"


def test_run_monte_carlo_point_records_max_batches_stop_reason():
    point = run_monte_carlo_point(
        model=_FakeModel([0, 0]),
        batch_size=1,
        ebno_db=0.0,
        min_batches=1,
        max_mc_batches=2,
        target_block_errors=None,
        target_ber=None,
        stop_policy="sweep",
        confidence_level=0.95,
        min_total_bits=0,
        include_feedback=False,
    )
    assert point["stop_reason"] == "max_batches"


def test_run_monte_carlo_point_records_target_block_error_stop_reason():
    point = run_monte_carlo_point(
        model=_FakeModel([1, 0]),
        batch_size=1,
        ebno_db=0.0,
        min_batches=1,
        max_mc_batches=4,
        target_block_errors=1,
        target_ber=None,
        stop_policy="sweep",
        confidence_level=0.95,
        min_total_bits=0,
        include_feedback=False,
    )
    assert point["stop_reason"] == "target_block_errors"


def test_append_point_metrics_marks_low_evidence_points_as_upper_bound_only():
    aggregate = initialize_stage_metrics(["m"])["m"]
    append_point_metrics(
        aggregate,
        confidence_level=0.95,
        point={
            "ber": 0.0,
            "latency_ms": 1.0,
            "throughput_bits_per_batch": 10.0,
            "energy_joules_per_batch": 0.1,
            "avg_power_w": 0.1,
            "runtime_sec": 0.1,
            "bit_errors": 29.0,
            "total_bits": 10_000.0,
            "block_errors": 1.0,
            "total_blocks": 1.0,
            "num_batches": 1.0,
            "stop_reason": "max_batches",
        },
    )
    assert aggregate["point_status"][-1] == POINT_STATUS_UPPER_BOUND_ONLY


def test_append_point_metrics_marks_resolved_points_at_threshold():
    aggregate = initialize_stage_metrics(["m"])["m"]
    append_point_metrics(
        aggregate,
        confidence_level=0.95,
        point={
            "ber": 0.1,
            "latency_ms": 1.0,
            "throughput_bits_per_batch": 10.0,
            "energy_joules_per_batch": 0.1,
            "avg_power_w": 0.1,
            "runtime_sec": 0.1,
            "bit_errors": 30.0,
            "total_bits": 10_000.0,
            "block_errors": 1.0,
            "total_blocks": 1.0,
            "num_batches": 1.0,
            "stop_reason": "target_block_errors",
        },
    )
    assert aggregate["point_status"][-1] == POINT_STATUS_RESOLVED
