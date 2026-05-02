from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
import scripts.generate_dataset as dataset_module
from src.models.resource_manager import ResourceDirectives

from .conftest import make_tiny_config, write_config


def _candidate(
    *,
    source: str,
    avg_ber: float,
    ber_upper: float,
    throughput_bits: float,
    latency_ms: float,
) -> dataset_module.CandidateEvaluation:
    return dataset_module.CandidateEvaluation(
        source=source,
        directives=ResourceDirectives(active_ut_mask=[1], per_ut_power=[1.0]),
        utility=0.0,
        avg_ber=avg_ber,
        ber_upper_confidence=ber_upper,
        throughput_eff=0.0,
        throughput_bits=throughput_bits,
        latency_ms=latency_ms,
        bit_errors=0,
        total_bits=1,
    )


def test_ber_first_candidate_sort_uses_reliability_then_throughput_then_latency():
    same_ber_high_upper = _candidate(
        source="same_ber_high_upper",
        avg_ber=0.01,
        ber_upper=0.03,
        throughput_bits=30.0,
        latency_ms=0.5,
    )
    same_ber_low_upper = _candidate(
        source="same_ber_low_upper",
        avg_ber=0.01,
        ber_upper=0.02,
        throughput_bits=5.0,
        latency_ms=5.0,
    )
    same_ber_low_throughput = _candidate(
        source="same_ber_low_throughput",
        avg_ber=0.01,
        ber_upper=0.02,
        throughput_bits=10.0,
        latency_ms=1.0,
    )
    same_ber_high_throughput = _candidate(
        source="same_ber_high_throughput",
        avg_ber=0.01,
        ber_upper=0.02,
        throughput_bits=20.0,
        latency_ms=2.0,
    )
    same_ber_same_throughput_lower_latency = _candidate(
        source="same_ber_same_throughput_lower_latency",
        avg_ber=0.01,
        ber_upper=0.02,
        throughput_bits=20.0,
        latency_ms=1.0,
    )
    lower_ber = _candidate(
        source="lower_ber",
        avg_ber=0.005,
        ber_upper=0.03,
        throughput_bits=5.0,
        latency_ms=10.0,
    )

    best = min(
        [same_ber_low_throughput, same_ber_high_throughput, lower_ber],
        key=lambda candidate: dataset_module._candidate_sort_key(candidate, "ber_first"),
    )
    assert best.source == "lower_ber"

    upper_best = min(
        [same_ber_high_upper, same_ber_low_upper],
        key=lambda candidate: dataset_module._candidate_sort_key(candidate, "ber_first"),
    )
    assert upper_best.source == "same_ber_low_upper"

    tie_best = min(
        [same_ber_low_throughput, same_ber_high_throughput],
        key=lambda candidate: dataset_module._candidate_sort_key(candidate, "ber_first"),
    )
    assert tie_best.source == "same_ber_high_throughput"

    latency_best = min(
        [same_ber_high_throughput, same_ber_same_throughput_lower_latency],
        key=lambda candidate: dataset_module._candidate_sort_key(candidate, "ber_first"),
    )
    assert latency_best.source == "same_ber_same_throughput_lower_latency"


def test_ber_first_candidate_sort_prefers_stable_manager_before_random_on_exact_tie():
    manager_candidate = _candidate(
        source="max_throughput",
        avg_ber=0.0,
        ber_upper=1e-8,
        throughput_bits=100.0,
        latency_ms=2.0,
    )
    random_candidate = _candidate(
        source="random_1",
        avg_ber=0.0,
        ber_upper=1e-8,
        throughput_bits=100.0,
        latency_ms=1.0,
    )

    best = min(
        [random_candidate, manager_candidate],
        key=lambda candidate: dataset_module._candidate_sort_key(candidate, "ber_first"),
    )

    assert best.source == "max_throughput"


def test_candidate_metrics_use_active_ut_mask_like_benchmark():
    result = {
        "bits": np.array([[[[0, 1, 0, 1]], [[1, 1, 0, 0]]]], dtype=np.int32),
        "bits_hat": np.array([[[[0, 1, 0, 1]], [[0, 0, 1, 1]]]], dtype=np.int32),
        "latency_sec": 0.001,
    }

    masked = dataset_module._candidate_metrics(
        result,
        latency_weight=0.0,
        ut_mask=[1, 0],
    )
    unmasked = dataset_module._candidate_metrics(
        result,
        latency_weight=0.0,
        ut_mask=None,
    )

    assert masked["avg_ber"] == 0.0
    assert masked["total_bits"] == 4
    assert unmasked["avg_ber"] == 0.5
    assert unmasked["total_bits"] == 8


@pytest.mark.slow
def test_dataset_generation_uses_one_shared_context_per_sample(monkeypatch, tmp_path):
    config_path = write_config(tmp_path, make_tiny_config(str(tmp_path / "results")))
    output_path = tmp_path / "rm_dataset.parquet"

    prepare_calls = 0
    context_h_ids: list[int] = []
    context_noise_ids: list[int] = []
    feedback_ids: list[int] = []

    original_prepare = dataset_module.Model.prepare_batch_context
    original_run_batch = dataset_module.Model.run_batch

    def counted_prepare(self, *args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(self, *args, **kwargs)

    def recording_run_batch(self, batch_context, *args, **kwargs):
        context_h_ids.append(id(batch_context.h_freq))
        context_noise_ids.append(id(batch_context.data_noise))
        if batch_context.feedback is not None:
            feedback_ids.append(id(batch_context.feedback.h_hat))
        return original_run_batch(self, batch_context, *args, **kwargs)

    monkeypatch.setattr(dataset_module.Model, "prepare_batch_context", counted_prepare)
    monkeypatch.setattr(dataset_module.Model, "run_batch", recording_run_batch)

    dataset_module.generate_dataset(
        output_path=str(output_path),
        samples=1,
        batch_size=1,
        scenario="umi",
        min_ebno=0.0,
        max_ebno=0.0,
        seed=1234,
        tries=3,
        latency_weight=0.002,
        config_path=str(config_path),
    )

    assert prepare_calls == 1
    assert len(set(context_h_ids)) == 1
    assert len(set(context_noise_ids)) == 1
    assert len(set(feedback_ids)) == 1


def test_latency_weight_changes_utility():
    sample_result = {
        "bits": np.array([0, 1, 0, 1]),
        "bits_hat": np.array([0, 1, 1, 1]),
        "latency_sec": 0.001,
    }
    utility_low, *_ = dataset_module._score_candidate(sample_result, latency_weight=0.001)
    utility_high, *_ = dataset_module._score_candidate(sample_result, latency_weight=0.01)
    assert utility_low > utility_high


def test_saved_parquet_columns_and_feature_shape(tmp_path):
    config_path = write_config(tmp_path, make_tiny_config(str(tmp_path / "results")))
    output_path = tmp_path / "rm_dataset.parquet"

    dataset_module.generate_dataset(
        output_path=str(output_path),
        samples=1,
        batch_size=1,
        scenario="umi",
        min_ebno=0.0,
        max_ebno=0.0,
        seed=1234,
        tries=4,
        latency_weight=0.002,
        config_path=str(config_path),
        objective="ber_first",
        candidate_managers="static,round_robin",
        channel="tr38901",
        ebno_grid=[0.0],
        random_active_count=2,
        random_power_min=0.85,
    )

    df = pd.read_parquet(output_path)
    assert df.columns.tolist() == [
        "scenario",
        "channel_model_type",
        "ebno_db",
        "sample_index",
        "channel_energy",
        "active_ut_mask",
        "per_ut_power",
        "oracle_utility",
        "oracle_avg_ber",
        "oracle_ber_upper_confidence",
        "oracle_throughput_eff",
        "oracle_throughput_bits",
        "oracle_latency_ms",
        "oracle_candidates",
        "oracle_eligible_candidates",
        "oracle_objective",
        "oracle_source_manager",
        "oracle_source_priority",
    ]
    assert df.iloc[0]["oracle_objective"] == "ber_first"
    assert df.iloc[0]["channel_model_type"] == "tr38901"
    assert df.iloc[0]["oracle_candidates"] == 4
    assert df.iloc[0]["oracle_eligible_candidates"] >= 1
    assert sum(df.iloc[0]["active_ut_mask"]) == 1
    assert df.iloc[0]["oracle_source_manager"] in {"round_robin", "random_1"}
    feature = np.stack(df.iloc[0]["channel_energy"]).astype(np.float32)
    assert feature.shape == (2, 32)
