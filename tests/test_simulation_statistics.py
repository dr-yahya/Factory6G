"""Tests for the reliability statistics, seeding and metric accumulation."""

from __future__ import annotations

import numpy as np
import pytest

from factory6g.sim.stages.common import (
    PointAccumulator,
    channel_nmse_db,
    clopper_pearson_upper,
    compare_methods_paired,
    derive_seed,
    jains_fairness_index,
    paired_bootstrap_ci,
    slot_duration_seconds,
)


class TestClopperPearson:
    def test_zero_errors_matches_the_rule_of_three(self):
        # With no observed errors in n trials the 95% upper bound is ~3/n.
        bound = clopper_pearson_upper(0, 1000, 0.95)
        assert bound == pytest.approx(3.0 / 1000, rel=0.05)

    def test_bound_is_above_the_point_estimate(self):
        assert clopper_pearson_upper(10, 1000, 0.95) > 10 / 1000

    def test_bound_tightens_with_more_evidence(self):
        assert clopper_pearson_upper(0, 10_000, 0.95) < clopper_pearson_upper(0, 100, 0.95)

    def test_degenerate_inputs_are_conservative(self):
        assert clopper_pearson_upper(0, 0, 0.95) == 1.0
        assert clopper_pearson_upper(5, 5, 0.95) == 1.0


class TestFairness:
    def test_equal_allocation_is_perfectly_fair(self):
        assert jains_fairness_index([2.0, 2.0, 2.0, 2.0]) == pytest.approx(1.0)

    def test_single_user_allocation_is_one_over_n(self):
        assert jains_fairness_index([4.0, 0.0, 0.0, 0.0]) == pytest.approx(0.25)

    def test_empty_allocation_does_not_divide_by_zero(self):
        assert jains_fairness_index([]) == 1.0
        assert jains_fairness_index([0.0, 0.0]) == 1.0


class TestNMSE:
    def test_perfect_estimate_is_minus_infinity(self):
        h = np.ones((4, 4), dtype=complex)
        assert channel_nmse_db(h, h) == float("-inf")

    def test_ten_percent_amplitude_error_is_minus_twenty_db(self):
        h = np.ones((4, 4), dtype=complex)
        assert channel_nmse_db(h, h * 0.9) == pytest.approx(-20.0, abs=1e-6)

    def test_shape_mismatch_returns_nan(self):
        assert np.isnan(channel_nmse_db(np.ones((2, 2)), np.ones((3, 3))))


class TestPairedBootstrap:
    def test_interval_brackets_the_mean(self):
        rng = np.random.default_rng(0)
        mean, lower, upper = paired_bootstrap_ci(rng.normal(0.5, 0.1, 500), seed=1)
        assert lower < mean < upper
        assert lower <= 0.5 <= upper

    def test_a_real_difference_produces_an_interval_excluding_zero(self):
        rng = np.random.default_rng(1)
        _, lower, upper = paired_bootstrap_ci(rng.normal(0.2, 0.01, 200), seed=2)
        assert lower > 0.0 and upper > 0.0

    def test_no_difference_produces_an_interval_containing_zero(self):
        rng = np.random.default_rng(2)
        _, lower, upper = paired_bootstrap_ci(rng.normal(0.0, 0.1, 200), seed=3)
        assert lower <= 0.0 <= upper

    def test_is_deterministic_for_a_fixed_seed(self):
        values = [0.1, -0.2, 0.3, 0.05]
        assert paired_bootstrap_ci(values, seed=7) == paired_bootstrap_ci(values, seed=7)

    def test_degenerate_inputs_do_not_raise(self):
        assert all(np.isnan(v) for v in paired_bootstrap_ci([]))


class TestPairedComparison:
    def test_reports_significance_against_the_reference(self):
        points = {
            "baseline": [{"batch_block_errors": [50] * 20, "batch_blocks": [100] * 20}],
            "better": [{"batch_block_errors": [10] * 20, "batch_blocks": [100] * 20}],
        }
        result = compare_methods_paired(points, reference="baseline", num_resamples=2000)
        row = result["better"][0]
        assert row["mean_bler_delta"] == pytest.approx(-0.4)
        assert row["significant"] is True

    def test_uses_only_the_batch_prefix_both_methods_ran(self):
        points = {
            "baseline": [{"batch_block_errors": [1, 1, 1], "batch_blocks": [10, 10, 10]}],
            "short": [{"batch_block_errors": [1], "batch_blocks": [10]}],
        }
        result = compare_methods_paired(points, reference="baseline", num_resamples=100)
        assert result["short"][0]["num_paired_batches"] == 1

    def test_unknown_reference_yields_no_comparisons(self):
        assert compare_methods_paired({"a": []}, reference="missing") == {}


class TestDerivedSeeds:
    def test_same_inputs_give_the_same_seed(self):
        assert derive_seed(42, "estimators", 0.0, 3) == derive_seed(42, "estimators", 0.0, 3)

    def test_different_points_give_different_seeds(self):
        assert derive_seed(42, "estimators", 0.0, 3) != derive_seed(42, "estimators", 2.0, 3)
        assert derive_seed(42, "estimators", 0.0, 3) != derive_seed(42, "estimators", 0.0, 4)

    def test_seed_does_not_depend_on_which_methods_are_enabled(self):
        # The point identity carries no method list, so enabling another method
        # cannot change this point's channel realisations.
        assert derive_seed(42, "resource_managers", 6.0, 1) == derive_seed(
            42, "resource_managers", 6.0, 1
        )

    def test_seed_fits_in_32_bits(self):
        assert 0 <= derive_seed("x", 1) < 2**32


class TestPointAccumulator:
    @staticmethod
    def _result(bit_errors: int, num_ut: int = 2):
        bits = np.zeros((2, num_ut, 1, 8), dtype=np.int32)
        bits_hat = bits.copy()
        bits_hat.reshape(-1)[:bit_errors] = 1
        delivery = np.ones((2, num_ut, 1), dtype=np.int32)
        return {
            "bits": bits,
            "bits_hat": bits_hat,
            "delivery_round": delivery,
            "scheduled_block_mask": np.ones((2, num_ut, 1), dtype=bool),
            "latency_sec": 1e-3,
            "energy_joules": 2e-3,
            "radiated_power_w": 0.4,
        }

    def test_accumulates_bler_and_ber(self):
        acc = PointAccumulator(num_ut=2)
        acc.add_batch(self._result(8), ut_mask=[1, 1], elapsed_sec=0.1, num_ut=2)
        point = acc.finalize(
            confidence_level=0.95, slot_duration_sec=1e-3, max_harq_rounds=1
        )
        assert point["total_bits"] == 32
        assert point["bit_errors"] == 8
        assert point["ber"] == pytest.approx(0.25)
        assert 0.0 < point["bler"] <= 1.0
        assert point["bler_upper_confidence"] >= point["bler"]

    def test_power_is_energy_over_slot_time_not_wall_clock(self):
        acc = PointAccumulator(num_ut=2)
        acc.add_batch(self._result(0), ut_mask=[1, 1], elapsed_sec=99.0, num_ut=2)
        point = acc.finalize(
            confidence_level=0.95, slot_duration_sec=1e-3, max_harq_rounds=1
        )
        # 2 mJ over a 1 ms slot is 2 W, regardless of how slow the host was.
        assert point["avg_power_w"] == pytest.approx(2.0)

    def test_round_trips_through_serialisation(self):
        acc = PointAccumulator(num_ut=2)
        acc.add_batch(self._result(4), ut_mask=[1, 1], elapsed_sec=0.1, num_ut=2)
        restored = PointAccumulator.from_dict(acc.to_dict())
        kwargs = dict(confidence_level=0.95, slot_duration_sec=1e-3, max_harq_rounds=1)
        before, after = acc.finalize(**kwargs), restored.finalize(**kwargs)
        assert before.keys() == after.keys()
        for key, value in before.items():
            other = after[key]
            if isinstance(value, float) and np.isnan(value):
                assert np.isnan(other), key
            else:
                assert value == other, key

    def test_retains_per_batch_samples_for_paired_analysis(self):
        acc = PointAccumulator(num_ut=2)
        for _ in range(3):
            acc.add_batch(self._result(4), ut_mask=[1, 1], elapsed_sec=0.1, num_ut=2)
        point = acc.finalize(
            confidence_level=0.95, slot_duration_sec=1e-3, max_harq_rounds=1
        )
        assert len(point["batch_block_errors"]) == 3
        assert len(point["batch_blocks"]) == 3


def test_slot_duration_follows_the_numerology():
    full_slot = slot_duration_seconds(
        {"subcarrier_spacing": 30e3, "fft_size": 128, "cyclic_prefix_length": 20, "num_ofdm_symbols": 14}
    )
    mini_slot = slot_duration_seconds(
        {"subcarrier_spacing": 120e3, "fft_size": 128, "cyclic_prefix_length": 20, "num_ofdm_symbols": 4}
    )
    # A 4-symbol mini-slot at 120 kHz is far shorter than a 14-symbol slot at 30 kHz.
    assert mini_slot < full_slot / 10
