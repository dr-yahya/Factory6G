from __future__ import annotations

import numpy as np

from factory6g.models.model import Model

from .conftest import make_tiny_config, set_all_seeds


def _build_model():
    config_data = make_tiny_config("results")
    runtime_config = config_data["system"] | config_data["transceiver"]
    return Model(config=runtime_config, estimator_type="ls", perfect_csi=False)


def test_prepare_batch_context_returns_consistent_shapes():
    model = _build_model()
    context = model.prepare_batch_context(batch_size=1, ebno_db=0.0, include_feedback=True)
    assert tuple(context.h_freq.shape) == (1, 1, 4, 2, 1, 14, 32)
    assert tuple(context.probe_noise.shape) == (1, 1, 4, 14, 32)
    assert tuple(context.data_noise.shape) == (1, 1, 4, 14, 32)
    assert tuple(context.source_bits.shape) == (1, 2, 1, 384)
    assert context.feedback is not None
    assert tuple(context.feedback.h_hat.shape[-2:]) == (14, 32)


def test_reusing_one_batch_context_yields_identical_outputs_for_identical_directives():
    model = _build_model()
    context = model.prepare_batch_context(batch_size=1, ebno_db=0.0, include_feedback=True)
    result_one = model.run_batch(context, include_details=False)
    result_two = model.run_batch(context, include_details=False)
    assert np.array_equal(result_one["bits"], result_two["bits"])
    assert np.array_equal(result_one["bits_hat"], result_two["bits_hat"])


def test_resetting_seeds_reproduces_new_contexts():
    set_all_seeds(2026)
    model_one = _build_model()
    context_one = model_one.prepare_batch_context(batch_size=1, ebno_db=0.0, include_feedback=True)

    set_all_seeds(2026)
    model_two = _build_model()
    context_two = model_two.prepare_batch_context(batch_size=1, ebno_db=0.0, include_feedback=True)

    assert np.allclose(context_one.h_freq.numpy(), context_two.h_freq.numpy())
    assert np.allclose(context_one.data_noise.numpy(), context_two.data_noise.numpy())
    assert np.allclose(context_one.source_bits.numpy(), context_two.source_bits.numpy())
