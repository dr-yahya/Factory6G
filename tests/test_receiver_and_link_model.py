"""Tests for LLR clipping, scheduled-stream masking, HARQ and the energy model."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from factory6g.components.receiver import (  # noqa: E402
    DEFAULT_LLR_CLIP,
    _resolve_llr_clip,
    apply_stream_mask,
)
from factory6g.models.model import Model  # noqa: E402
from factory6g.models.resource_manager import ResourceDirectives  # noqa: E402


def _runtime_config(**overrides) -> dict:
    config = {
        "num_ut": 4,
        "num_bs_ant": 8,
        "fft_size": 32,
        "num_ofdm_symbols": 14,
        "channel_model_type": "rayleigh",
        "num_bits_per_symbol": 2,
        "coderate": 0.5,
    }
    config.update(overrides)
    return config


class TestLLRClipResolution:
    def test_absent_key_uses_wide_default(self):
        assert _resolve_llr_clip({}) == DEFAULT_LLR_CLIP

    def test_default_is_wide_enough_not_to_saturate_high_snr_llrs(self):
        # The historical value of 20 created an artificial high-SNR error floor.
        assert DEFAULT_LLR_CLIP >= 100.0

    def test_null_disables_clipping(self):
        assert _resolve_llr_clip({"llr_clip": None}) is None

    def test_non_positive_disables_clipping(self):
        assert _resolve_llr_clip({"llr_clip": 0}) is None
        assert _resolve_llr_clip({"llr_clip": -5}) is None

    def test_explicit_value_is_honoured(self):
        assert _resolve_llr_clip({"llr_clip": 33.0}) == 33.0


class TestStreamMask:
    def test_muted_transmitter_columns_are_zeroed(self):
        h = tf.ones([2, 1, 4, 3, 1, 14, 8], dtype=tf.complex64)
        err = tf.ones([2, 1, 4, 3, 1, 14, 8], dtype=tf.float32)
        h_masked, err_masked = apply_stream_mask(h, err, [1, 0, 1])

        h_np = h_masked.numpy()
        assert np.all(h_np[:, :, :, 1] == 0)
        assert np.all(h_np[:, :, :, 0] == 1)
        assert np.all(h_np[:, :, :, 2] == 1)
        assert np.all(err_masked.numpy()[:, :, :, 1] == 0)

    def test_none_mask_is_a_no_op(self):
        h = tf.ones([1, 1, 2, 2, 1, 4, 4], dtype=tf.complex64)
        h_masked, err = apply_stream_mask(h, 0.5, None)
        assert h_masked is h
        assert err == 0.5

    def test_scalar_err_var_survives_masking(self):
        h = tf.ones([1, 1, 2, 2, 1, 4, 4], dtype=tf.complex64)
        _, err = apply_stream_mask(h, 0.25, [1, 0])
        assert err == 0.25


class TestEnergyModel:
    """The previous model's terms cancelled to a constant and ignored directives."""

    def test_energy_increases_with_allocated_power(self):
        model = Model(config=_runtime_config(), estimator_type="ls")
        iters = tf.constant([[10.0]])
        low = model._estimate_energy(
            ResourceDirectives(active_ut_mask=[1, 1, 0, 0], per_ut_power=[0.2, 0.2, 0.0, 0.0]),
            iters,
        )
        high = model._estimate_energy(
            ResourceDirectives(active_ut_mask=[1, 1, 0, 0], per_ut_power=[1.0, 1.0, 0.0, 0.0]),
            iters,
        )
        assert high > low

    def test_energy_increases_with_number_of_scheduled_users(self):
        model = Model(config=_runtime_config(), estimator_type="ls")
        iters = tf.constant([[10.0]])
        one = model._estimate_energy(
            ResourceDirectives(active_ut_mask=[1, 0, 0, 0], per_ut_power=[1.0, 0.0, 0.0, 0.0]),
            iters,
        )
        three = model._estimate_energy(
            ResourceDirectives(active_ut_mask=[1, 1, 1, 0], per_ut_power=[1.0, 1.0, 1.0, 0.0]),
            iters,
        )
        assert three > one

    def test_radiated_power_tracks_directives(self):
        model = Model(config=_runtime_config(ut_max_tx_power_w=0.2), estimator_type="ls")
        directives = ResourceDirectives(
            active_ut_mask=[1, 1, 0, 0], per_ut_power=[1.0, 0.5, 0.0, 0.0]
        )
        assert model._radiated_power(directives) == pytest.approx(0.2 * 1.5)


class TestHARQ:
    def test_single_round_is_the_default_and_charges_one_slot(self):
        model = Model(config=_runtime_config(), estimator_type="ls")
        context = model.prepare_batch_context(batch_size=2, ebno_db=20.0, include_feedback=False)
        directives = ResourceDirectives(
            active_ut_mask=[1, 1, 1, 1], per_ut_power=[1.0] * 4
        )
        result = model.run_batch(context, directives=directives, harq_max_rounds=1)
        assert result["harq_rounds_executed"] == 1
        assert result["latency_sec"] == pytest.approx(result["slot_duration_sec"])

    def test_retransmission_never_reduces_delivery_and_costs_latency(self):
        model = Model(config=_runtime_config(), estimator_type="ls")
        context = model.prepare_batch_context(batch_size=4, ebno_db=0.0, include_feedback=False)
        directives = ResourceDirectives(
            active_ut_mask=[1, 1, 1, 1], per_ut_power=[1.0] * 4
        )
        one = model.run_batch(context, directives=directives, harq_max_rounds=1)
        three = model.run_batch(context, directives=directives, harq_max_rounds=3)

        delivered_one = (one["delivery_round"] > 0).sum()
        delivered_three = (three["delivery_round"] > 0).sum()
        assert delivered_three >= delivered_one
        assert three["latency_sec"] >= one["latency_sec"]

    def test_muted_users_are_excluded_from_latency_and_retransmission(self):
        model = Model(config=_runtime_config(), estimator_type="ls")
        context = model.prepare_batch_context(batch_size=2, ebno_db=0.0, include_feedback=False)
        directives = ResourceDirectives(
            active_ut_mask=[1, 1, 0, 0], per_ut_power=[1.0, 1.0, 0.0, 0.0]
        )
        result = model.run_batch(context, directives=directives, harq_max_rounds=3)

        scheduled = result["scheduled_block_mask"]
        assert not scheduled[:, 2, :].any()
        assert not scheduled[:, 3, :].any()
        # Latency is NaN (undefined) for users that never transmitted.
        assert np.all(np.isnan(result["latency_per_block_sec"][:, 2, :]))
        assert np.all(~np.isnan(result["latency_per_block_sec"][:, 0, :]))
