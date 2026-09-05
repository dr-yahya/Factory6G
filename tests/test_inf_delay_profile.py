"""Tests for the TR 38.901 Indoor Factory delay profile and its resolvability.

The InF channel was frequency-flat until the delay-spread model was wired in:
the large-scale model was layered onto single-tap block fading and the hall
volume/surface parameters were never read. These tests pin both the model and
the bandwidth condition under which it is actually observable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from factory6g.components.inf_channel import (
    coherence_bandwidth_hz,
    exponential_pdp,
    hall_volume_and_surface,
    inf_delay_spread_seconds,
)

tf = pytest.importorskip("tensorflow")
pytest.importorskip("sionna")

from sionna.phy.ofdm import ResourceGrid  # noqa: E402

from factory6g.components.antenna import AntennaConfig  # noqa: E402
from factory6g.components.channel import ChannelModel  # noqa: E402

SMALL_HALL = [15.0, 15.0, 5.0]
LARGE_HALL = [40.0, 40.0, 8.0]


class TestHallGeometry:
    def test_volume_and_surface_of_a_rectangular_hall(self):
        volume, surface = hall_volume_and_surface(SMALL_HALL)
        assert volume == pytest.approx(15 * 15 * 5)
        # Floor + ceiling + four walls.
        assert surface == pytest.approx(2 * 15 * 15 + 2 * 5 * (15 + 15))

    def test_bigger_hall_has_bigger_volume_to_surface_ratio(self):
        vs_small = np.divide(*hall_volume_and_surface(SMALL_HALL))
        vs_large = np.divide(*hall_volume_and_surface(LARGE_HALL))
        assert vs_large > vs_small


class TestDelaySpreadModel:
    def test_matches_the_tr38901_formula(self):
        volume, surface = hall_volume_and_surface(SMALL_HALL)
        expected = 10.0 ** (math.log10(26.0 * (volume / surface) + 14.0) - 9.35)
        assert float(inf_delay_spread_seconds(volume, surface)[0]) == pytest.approx(expected)

    def test_small_hall_lands_in_the_expected_tens_of_nanoseconds(self):
        spread = float(inf_delay_spread_seconds(*hall_volume_and_surface(SMALL_HALL))[0])
        assert 15e-9 < spread < 35e-9

    def test_delay_spread_grows_with_hall_size(self):
        small = float(inf_delay_spread_seconds(*hall_volume_and_surface(SMALL_HALL))[0])
        large = float(inf_delay_spread_seconds(*hall_volume_and_surface(LARGE_HALL))[0])
        assert large > small

    def test_rng_draws_a_lognormal_spread_around_the_mean(self):
        volume, surface = hall_volume_and_surface(SMALL_HALL)
        rng = np.random.default_rng(0)
        draws = inf_delay_spread_seconds(volume, surface, rng=rng, num_links=4000)
        assert draws.shape == (4000,)
        assert np.all(draws > 0)
        # sigma_lgDS = 0.15 in log10 space.
        assert np.std(np.log10(draws)) == pytest.approx(0.15, rel=0.12)

    def test_deterministic_without_an_rng(self):
        volume, surface = hall_volume_and_surface(SMALL_HALL)
        a = inf_delay_spread_seconds(volume, surface, num_links=3)
        assert np.allclose(a, a[0])


class TestPowerDelayProfile:
    def test_profile_is_normalised_and_decaying(self):
        pdp = exponential_pdp(30e-9, 20, 1.0 / 61.44e6)
        assert pdp.sum() == pytest.approx(1.0)
        assert np.all(np.diff(pdp) <= 0)

    def test_longer_spread_puts_more_energy_in_later_taps(self):
        sample = 1.0 / 61.44e6
        short = exponential_pdp(10e-9, 20, sample)
        long_ = exponential_pdp(60e-9, 20, sample)
        assert long_[1:].sum() > short[1:].sum()

    def test_zero_spread_collapses_to_a_single_tap(self):
        pdp = exponential_pdp(0.0, 16, 1e-8)
        assert pdp[0] == pytest.approx(1.0)
        assert pdp[1:].sum() == pytest.approx(0.0)


class TestCoherenceBandwidth:
    def test_follows_the_one_over_five_ds_rule(self):
        assert coherence_bandwidth_hz(20e-9) == pytest.approx(1.0 / (5 * 20e-9))

    def test_shorter_spread_gives_wider_coherence_bandwidth(self):
        assert coherence_bandwidth_hz(10e-9) > coherence_bandwidth_hz(40e-9)

    def test_flat_channel_has_unbounded_coherence_bandwidth(self):
        assert coherence_bandwidth_hz(0.0) == float("inf")


def _config(fft_size: int, num_symbols: int, scs: float, room):
    return {
        "num_ut": 2,
        "num_ut_ant": 1,
        "num_bs_ant": 4,
        "fft_size": fft_size,
        "num_ofdm_symbols": num_symbols,
        "subcarrier_spacing": scs,
        "cyclic_prefix_length": 20,
        "pilot_ofdm_symbol_indices": [0, 2] if num_symbols <= 4 else [2, 11],
        "direction": "uplink",
        "carrier_frequency": 3.5e9,
        "tx_pattern": "tr38901",
        "tx_polarization": "cross",
        "rx_pattern": "iso",
        "rx_polarization": "V",
        "antenna_spacing": 0.5,
        "seed": 5,
        "channel_model_type": "inf",
        "scenario": "inf_dh",
        "enable_pathloss": True,
        "enable_shadow_fading": True,
        "num_machines": 5,
        "machine_size_range": [[0.5, 2.0], [0.5, 2.0], [0.5, 1.5]],
        "room_dimensions": room,
    }


def _channel(cfg):
    rg = ResourceGrid(
        num_ofdm_symbols=cfg["num_ofdm_symbols"],
        fft_size=cfg["fft_size"],
        subcarrier_spacing=cfg["subcarrier_spacing"],
        num_tx=cfg["num_ut"],
        num_streams_per_tx=1,
        cyclic_prefix_length=cfg["cyclic_prefix_length"],
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=cfg["pilot_ofdm_symbol_indices"],
    )
    return ChannelModel(cfg, AntennaConfig(cfg), rg)


def _lag1_correlation(h: np.ndarray) -> float:
    a, b = h[..., :-1], h[..., 1:]
    return float(np.abs(np.mean(a * np.conj(b))) / max(np.mean(np.abs(h) ** 2), 1e-30))


class TestSelectivityReport:
    def test_narrow_carrier_cannot_resolve_the_hall(self):
        report = _channel(_config(128, 14, 30e3, SMALL_HALL)).frequency_selectivity_report()
        assert report["signal_bandwidth_hz"] == pytest.approx(128 * 30e3)
        # 3.84 MHz against a 5-8 MHz coherence bandwidth.
        assert report["selectivity_ratio"] < 1.0
        assert report["delay_spread_samples"] < 0.2

    def test_wide_carrier_resolves_it(self):
        report = _channel(_config(512, 4, 120e3, SMALL_HALL)).frequency_selectivity_report()
        assert report["signal_bandwidth_hz"] == pytest.approx(512 * 120e3)
        assert report["selectivity_ratio"] > 5.0
        assert report["delay_spread_samples"] > 1.0

    def test_larger_hall_is_more_selective_at_fixed_bandwidth(self):
        narrow = _channel(_config(512, 4, 120e3, SMALL_HALL)).frequency_selectivity_report()
        wide = _channel(_config(512, 4, 120e3, LARGE_HALL)).frequency_selectivity_report()
        assert wide["selectivity_ratio"] > narrow["selectivity_ratio"]


class TestGeneratedChannel:
    def test_channel_is_power_normalised(self):
        h = _channel(_config(512, 4, 120e3, SMALL_HALL)).sample_frequency_response(6).numpy()
        assert np.mean(np.abs(h) ** 2) == pytest.approx(1.0, rel=0.35)

    def test_wide_carrier_produces_frequency_selectivity(self):
        """The bug this guards: the channel used to be flat at every bandwidth."""
        narrow = _channel(_config(128, 14, 30e3, SMALL_HALL)).sample_frequency_response(6).numpy()
        wide = _channel(_config(512, 4, 120e3, LARGE_HALL)).sample_frequency_response(6).numpy()
        # Adjacent-subcarrier correlation drops once the delay profile is resolved.
        assert _lag1_correlation(narrow) > 0.99
        assert _lag1_correlation(wide) < _lag1_correlation(narrow)

    def test_diagnostics_report_the_delay_spread_used(self):
        channel = _channel(_config(512, 4, 120e3, SMALL_HALL))
        channel.sample_frequency_response(4)
        diagnostics = channel.last_large_scale_diagnostics()
        assert "rms_delay_spread_sec" in diagnostics
        assert np.all(diagnostics["rms_delay_spread_sec"] > 0)
