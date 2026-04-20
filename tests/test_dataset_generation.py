from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
import scripts.generate_dataset as dataset_module

from .conftest import make_tiny_config, write_config


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
        tries=2,
        latency_weight=0.002,
        config_path=str(config_path),
    )

    df = pd.read_parquet(output_path)
    assert df.columns.tolist() == [
        "scenario",
        "ebno_db",
        "sample_index",
        "channel_energy",
        "active_ut_mask",
        "per_ut_power",
        "oracle_utility",
        "oracle_avg_ber",
        "oracle_throughput_eff",
        "oracle_latency_ms",
        "oracle_candidates",
    ]
    feature = np.array(df.iloc[0]["channel_energy"], dtype=np.float32)
    assert feature.shape == (2, 32)
