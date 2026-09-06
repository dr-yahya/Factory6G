"""Tests for Rician LOS structure, TR 38.901 InF propagation and CSI ageing."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("sionna")

from sionna.phy.ofdm import ResourceGrid  # noqa: E402

from factory6g.components.antenna import AntennaConfig  # noqa: E402
from factory6g.components.channel import ChannelModel  # noqa: E402
from factory6g.components.inf_channel import (  # noqa: E402
    clutter_density_from_layout,
    inf_los_probability,
    inf_path_loss_db,
    inf_shadow_sigma_db,
)
from factory6g.models.model import Model  # noqa: E402

_BASE = {
    "num_ut": 4,
    "num_ut_ant": 1,
    "num_bs_ant": 8,
    "fft_size": 32,
    "num_ofdm_symbols": 14,
    "subcarrier_spacing": 30e3,
    "cyclic_prefix_length": 20,
    "direction": "uplink",
    "seed": 7,
    "carrier_frequency": 3.5e9,
    "tx_pattern": "tr38901",
    "tx_polarization": "cross",
    "rx_pattern": "iso",
    "rx_polarization": "V",
    "antenna_spacing": 0.5,
}


def _resource_grid() -> ResourceGrid:
    return ResourceGrid(
        num_ofdm_symbols=14,
        fft_size=32,
        subcarrier_spacing=30e3,
        num_tx=4,
        num_streams_per_tx=1,
        cyclic_prefix_length=20,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[2, 11],
    )


def _sample(config: dict, batch: int = 8) -> np.ndarray:
    return ChannelModel(config, AntennaConfig(config), _resource_grid()).sample_frequency_response(
        batch
    ).numpy()


def _mean_cross_user_correlation(h: np.ndarray) -> float:
    """Mean |correlation| between users' spatial signatures."""
    vectors = h.mean(axis=(5, 6))[0, 0, :, :, 0]
    vectors = vectors / np.linalg.norm(vectors, axis=0, keepdims=True)
    gram = np.abs(vectors.conj().T @ vectors)
    return float(gram[~np.eye(gram.shape[0], dtype=bool)].mean())


class TestRicianLOS:
    def test_blend_preserves_unit_average_power(self):
        h = _sample({**_BASE, "channel_model_type": "rician", "rician_k_factor": 5.0})
        assert np.mean(np.abs(h) ** 2) == pytest.approx(1.0, rel=0.15)

    def test_users_stay_spatially_separable(self):
        """A scalar LOS term made every user's LOS component identical."""
        rician = _sample({**_BASE, "channel_model_type": "rician", "rician_k_factor": 5.0})
        rayleigh = _sample({**_BASE, "channel_model_type": "rayleigh"})

        k = 5.0
        scalar_los = np.sqrt(k / (k + 1)) + np.sqrt(1 / (k + 1)) * rayleigh

        assert _mean_cross_user_correlation(rician) < 0.6
        assert _mean_cross_user_correlation(scalar_los) > 0.8
        assert _mean_cross_user_correlation(rician) < _mean_cross_user_correlation(scalar_los)

    def test_higher_k_factor_still_preserves_power(self):
        h = _sample({**_BASE, "channel_model_type": "rician", "rician_k_factor": 20.0})
        assert np.mean(np.abs(h) ** 2) == pytest.approx(1.0, rel=0.15)


class TestInFPropagation:
    def test_los_probability_falls_with_distance(self):
        distances = np.array([1.0, 5.0, 20.0, 50.0])
        probability = inf_los_probability(
            distances,
            scenario="inf_sl",
            clutter_density=0.2,
            clutter_size_m=2.0,
            bs_height_m=8.0,
            ut_height_m=1.5,
            clutter_height_m=2.0,
        )
        assert np.all(np.diff(probability) < 0)
        assert np.all((probability >= 0) & (probability <= 1))

    def test_denser_clutter_lowers_los_probability(self):
        kwargs = dict(
            scenario="inf_dl",
            clutter_size_m=2.0,
            bs_height_m=8.0,
            ut_height_m=1.5,
            clutter_height_m=2.0,
        )
        distances = np.array([5.0, 20.0])
        sparse = inf_los_probability(distances, clutter_density=0.2, **kwargs)
        dense = inf_los_probability(distances, clutter_density=0.8, **kwargs)
        assert np.all(dense < sparse)

    def test_high_high_scenario_is_always_los(self):
        probability = inf_los_probability(
            np.array([1.0, 100.0]),
            scenario="inf_hh",
            clutter_density=0.5,
            clutter_size_m=2.0,
            bs_height_m=8.0,
            ut_height_m=1.5,
            clutter_height_m=2.0,
        )
        assert np.all(probability == 1.0)

    def test_nlos_path_loss_is_never_below_los(self):
        distances = np.array([1.0, 10.0, 100.0])
        for scenario in ("inf_sl", "inf_dl", "inf_sh", "inf_dh"):
            los = inf_path_loss_db(
                distances,
                scenario=scenario,
                carrier_frequency_hz=3.5e9,
                is_los=np.ones_like(distances, dtype=bool),
            )
            nlos = inf_path_loss_db(
                distances,
                scenario=scenario,
                carrier_frequency_hz=3.5e9,
                is_los=np.zeros_like(distances, dtype=bool),
            )
            assert np.all(nlos >= los - 1e-9), scenario

    def test_path_loss_grows_with_distance_and_frequency(self):
        near = inf_path_loss_db(
            np.array([5.0]), scenario="inf_dl", carrier_frequency_hz=3.5e9, is_los=np.array([False])
        )
        far = inf_path_loss_db(
            np.array([50.0]), scenario="inf_dl", carrier_frequency_hz=3.5e9, is_los=np.array([False])
        )
        high_band = inf_path_loss_db(
            np.array([5.0]), scenario="inf_dl", carrier_frequency_hz=28e9, is_los=np.array([False])
        )
        assert far > near
        assert high_band > near

    def test_shadow_sigma_differs_between_los_and_nlos(self):
        sigma = inf_shadow_sigma_db("inf_dl", np.array([True, False]))
        assert sigma[0] != sigma[1]

    def test_unknown_scenario_is_rejected(self):
        with pytest.raises(ValueError):
            inf_path_loss_db(
                np.array([1.0]), scenario="inf_zz", carrier_frequency_hz=3.5e9, is_los=np.array([True])
            )

    def test_clutter_density_tracks_the_hall_layout(self):
        small = clutter_density_from_layout(5, [[0.5, 2.0], [0.5, 2.0]], [15, 15, 5])
        crowded = clutter_density_from_layout(40, [[0.5, 2.0], [0.5, 2.0]], [15, 15, 5])
        assert crowded > small
        assert 0.0 < small < 1.0 and 0.0 < crowded < 1.0

    def test_channel_is_power_normalised_and_spreads_users(self):
        config = {
            **_BASE,
            "channel_model_type": "inf",
            "scenario": "inf_dl",
            "room_dimensions": [15.0, 15.0, 5.0],
            "num_machines": 5,
            "machine_size_range": [[0.5, 2.0], [0.5, 2.0], [0.5, 1.5]],
            "enable_pathloss": True,
            "enable_shadow_fading": True,
        }
        model = ChannelModel(config, AntennaConfig(config), _resource_grid())
        h = model.sample_frequency_response(8).numpy()
        assert np.mean(np.abs(h) ** 2) == pytest.approx(1.0, rel=0.2)

        diagnostics = model.last_large_scale_diagnostics()
        assert diagnostics is not None
        assert set(diagnostics) >= {"is_los", "path_loss_db", "distance_3d_m"}
        # Users must not all see the same gain, or there is nothing to schedule.
        assert np.std(20 * np.log10(diagnostics["amplitude_gain"])) > 0.5

    def test_hall_size_changes_the_link_distances(self):
        def distances(room):
            config = {
                **_BASE,
                "channel_model_type": "inf",
                "scenario": "inf_dl",
                "room_dimensions": room,
                "num_machines": 5,
                "machine_size_range": [[0.5, 2.0], [0.5, 2.0], [0.5, 1.5]],
            }
            model = ChannelModel(config, AntennaConfig(config), _resource_grid())
            model.sample_frequency_response(32)
            return model.last_large_scale_diagnostics()["distance_2d_m"].mean()

        assert distances([40.0, 40.0, 8.0]) > distances([15.0, 15.0, 5.0])


class TestCSIAgeing:
    @staticmethod
    def _model(**overrides) -> Model:
        config = {
            "num_ut": 4,
            "num_bs_ant": 8,
            "fft_size": 32,
            "num_ofdm_symbols": 14,
            "channel_model_type": "rayleigh",
            "num_bits_per_symbol": 2,
            "coderate": 0.5,
            "carrier_frequency": 3.5e9,
            "subcarrier_spacing": 30e3,
            "cyclic_prefix_length": 20,
        }
        config.update(overrides)
        return Model(config=config, estimator_type="ls")

    def test_static_users_have_perfectly_fresh_csi(self):
        assert self._model(max_ut_velocity=0.0, csi_feedback_delay_slots=4).csi_correlation() == 1.0

    def test_zero_delay_means_fresh_csi_even_when_moving(self):
        assert self._model(max_ut_velocity=3.0, csi_feedback_delay_slots=0).csi_correlation() == 1.0

    def test_correlation_falls_with_speed_and_delay(self):
        slow = self._model(max_ut_velocity=1.5, csi_feedback_delay_slots=4).csi_correlation()
        fast = self._model(max_ut_velocity=3.0, csi_feedback_delay_slots=4).csi_correlation()
        delayed = self._model(max_ut_velocity=3.0, csi_feedback_delay_slots=8).csi_correlation()
        assert 1.0 > slow > fast > delayed

    def test_aged_feedback_decorrelates_from_the_scheduled_channel(self):
        model = self._model(max_ut_velocity=3.0, csi_feedback_delay_slots=40)
        context = model.prepare_batch_context(batch_size=4, ebno_db=20.0, include_feedback=True)
        assert context.feedback is not None
        # Feedback exists but is drawn from an aged realisation, so it is no
        # longer a near-perfect copy of the channel that carries the data.
        assert abs(model.csi_correlation()) < 0.5
