"""Tests for the Monte Carlo evidence ceiling and tail extrapolation."""

from __future__ import annotations

import numpy as np
import pytest

from factory6g.sim.evidence import (
    MIN_RESOLVED_EVENTS,
    batches_needed_for,
    check_reliability_target,
    evidence_ceiling,
    extrapolate_bler,
    info_bits_per_codeword,
)


def _shipped_ceiling():
    """The configuration the repository ships with."""
    return evidence_ceiling(
        batch_size=20,
        max_batches=20,
        num_ut=4,
        fft_size=128,
        num_ofdm_symbols=14,
        num_pilot_symbols=2,
        num_bits_per_symbol=2,
        coderate=0.5,
    )


class TestEvidenceCeiling:
    def test_codeword_size_follows_the_numerology(self):
        # 12 data symbols x 128 subcarriers x 2 bits x rate 1/2.
        assert (
            info_bits_per_codeword(
                fft_size=128,
                num_ofdm_symbols=14,
                num_pilot_symbols=2,
                num_bits_per_symbol=2,
                coderate=0.5,
            )
            == 1536
        )

    def test_shipped_config_cannot_reach_the_urllc_regime(self):
        ceiling = _shipped_ceiling()
        assert ceiling.max_blocks_per_point == 1600
        assert ceiling.max_bits_per_point == 1600 * 1536
        # Two to three orders of magnitude short of a 1e-5..1e-6 target.
        assert ceiling.min_resolvable_bler > 1e-3
        assert ceiling.min_resolvable_ber > 1e-6

    def test_more_batches_lower_the_ceiling(self):
        small = evidence_ceiling(batch_size=20, max_batches=20, num_ut=4)
        large = evidence_ceiling(batch_size=20, max_batches=2000, num_ut=4)
        assert large.min_resolvable_bler < small.min_resolvable_bler
        assert large.max_blocks_per_point > small.max_blocks_per_point

    def test_zero_error_bound_matches_the_rule_of_three(self):
        ceiling = evidence_ceiling(batch_size=10, max_batches=10, num_ut=10)
        assert ceiling.zero_error_bler_bound == pytest.approx(
            3.0 / ceiling.max_blocks_per_point, rel=0.01
        )

    def test_describe_names_both_units(self):
        text = _shipped_ceiling().describe()
        assert "bits" in text and "codewords" in text


class TestReachability:
    def test_reachable_target_is_reported_as_such(self):
        ceiling = _shipped_ceiling()
        report = check_reliability_target(1e-1, ceiling, blocks_per_batch=80)
        assert report["reachable"] is True

    def test_urllc_target_is_flagged_with_the_budget_required(self):
        ceiling = _shipped_ceiling()
        report = check_reliability_target(1e-5, ceiling, blocks_per_batch=80)
        assert report["reachable"] is False
        assert report["batches_needed"] > 10_000
        assert "BELOW the evidence ceiling" in report["message"]

    def test_batches_needed_scales_inversely_with_the_rate(self):
        assert batches_needed_for(1e-4, blocks_per_batch=80) == 10 * batches_needed_for(
            1e-3, blocks_per_batch=80
        )

    def test_batches_needed_for_a_zero_rate_is_undefined(self):
        assert batches_needed_for(0.0, blocks_per_batch=80) == -1

    def test_min_events_threshold_is_the_documented_one(self):
        assert MIN_RESOLVED_EVENTS == 30


class TestTailExtrapolation:
    _EBNO = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    _BLER = np.array([3e-1, 8e-2, 1.5e-2, 2e-3, 3e-4])

    def test_recovers_a_known_log_linear_slope(self):
        ebno = np.arange(0.0, 10.0, 2.0)
        bler = 10.0 ** (-0.5 * ebno - 1.0)
        fit = extrapolate_bler(ebno, bler)
        assert fit["ok"] is True
        assert fit["slope_decades_per_db"] == pytest.approx(-0.5, abs=1e-6)
        assert fit["intercept"] == pytest.approx(-1.0, abs=1e-6)

    def test_prediction_interval_brackets_the_estimate(self):
        fit = extrapolate_bler(self._EBNO, self._BLER, target_ebno_db=14.0)
        assert fit["predicted_bler_lower"] < fit["predicted_bler"] < fit["predicted_bler_upper"]

    def test_flags_predictions_made_outside_the_fitted_range(self):
        beyond = extrapolate_bler(self._EBNO, self._BLER, target_ebno_db=14.0)
        inside = extrapolate_bler(self._EBNO, self._BLER, target_ebno_db=4.0)
        assert beyond["extrapolated_beyond_data"] is True
        assert inside["extrapolated_beyond_data"] is False

    def test_solves_for_the_ebno_a_reliability_target_needs(self):
        fit = extrapolate_bler(self._EBNO, self._BLER, target_bler=1e-6)
        assert fit["required_ebno_db"] > self._EBNO.max()
        assert fit["required_ebno_extrapolated"] is True

    def test_refuses_to_fit_without_enough_usable_points(self):
        fit = extrapolate_bler(np.array([0.0, 2.0]), np.array([0.0, 1.0]))
        assert fit["ok"] is False
        assert "at least" in fit["reason"]

    def test_ignores_zero_and_saturated_points(self):
        ebno = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        bler = np.array([1.0, 8e-2, 1.5e-2, 2e-3, 0.0])
        fit = extrapolate_bler(ebno, bler)
        assert fit["ok"] is True
        assert fit["num_points"] == 3
