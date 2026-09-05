from __future__ import annotations

import tensorflow as tf
from sionna.phy.channel import (
    ApplyOFDMChannel,
    GenerateOFDMChannel,
    RayleighBlockFading,
    gen_single_sector_topology as gen_topology,
)
from sionna.phy.channel.tr38901 import RMa, UMa, UMi
from sionna.phy.ofdm import ResourceGrid

import numpy as np

from .antenna import AntennaConfig
from .inf_channel import (
    INF_SCENARIOS,
    clutter_density_from_layout,
    coherence_bandwidth_hz,
    exponential_pdp,
    hall_volume_and_surface,
    inf_delay_spread_seconds,
    mean_clutter_size_m,
    sample_inf_large_scale_gain,
)


class ChannelModel:
    """Channel wrapper that separates topology, channel sampling, and channel application."""

    def __init__(self, config: dict, antenna_config: AntennaConfig, resource_grid: ResourceGrid):
        self.config = config
        self.antenna_config = antenna_config
        self.resource_grid = resource_grid

        channel_model_type = self.config.get("channel_model_type", "tr38901")
        if channel_model_type == "rayleigh":
            self._channel_model = self._create_rayleigh_channel()
            self._generator = GenerateOFDMChannel(self._channel_model, resource_grid, normalize_channel=True)
        elif channel_model_type == "rician":
            # Rician = structured LOS + Rayleigh NLOS, blended by the K-factor.
            self._channel_model = self._create_rayleigh_channel()
            self._generator = GenerateOFDMChannel(self._channel_model, resource_grid, normalize_channel=True)
        elif channel_model_type == "awgn":
            self._channel_model = None
            self._generator = None
        elif channel_model_type == "inf":
            # TR 38.901 Indoor Factory: small-scale fading from the block-fading
            # generator, large-scale statistics from the InF model.
            self._channel_model = self._create_rayleigh_channel()
            self._generator = GenerateOFDMChannel(
                self._channel_model, resource_grid, normalize_channel=True
            )
        else:  # tr38901
            self._channel_model = self._create_tr38901_channel()
            self._generator = GenerateOFDMChannel(self._channel_model, resource_grid, normalize_channel=True)
        self._rician_k_factor = float(self.config.get("rician_k_factor", 1.0))
        self._applier = ApplyOFDMChannel()
        self._rng = np.random.default_rng(int(self.config.get("seed", 0)) or None)
        self._last_large_scale: dict[str, object] | None = None

    def _create_tr38901_channel(self):
        channel_params = {
            "carrier_frequency": self.config.get("carrier_frequency", 3.5e9),
            "o2i_model": self.config.get("o2i_model", "low"),
            "ut_array": self.antenna_config.get_ut_array(),
            "bs_array": self.antenna_config.get_bs_array(),
            "direction": self.config.get("direction", "uplink"),
            "enable_pathloss": self.config.get("enable_pathloss", False),
            "enable_shadow_fading": self.config.get("enable_shadow_fading", False),
        }
        scenario_lower = self.config.get("scenario", "umi").lower()
        if scenario_lower == "umi":
            return UMi(**channel_params)
        if scenario_lower == "uma":
            return UMa(**channel_params)
        if scenario_lower == "rma":
            return RMa(**channel_params)
        raise ValueError(f"Unknown scenario: {scenario_lower}. Supported: umi, uma, rma.")

    def _create_rayleigh_channel(self):
        if self.config.get("direction", "uplink") == "uplink":
            num_rx = 1
            num_rx_ant = int(self.config.get("num_bs_ant", 32))
            num_tx = int(self.config.get("num_ut", 8))
            num_tx_ant = int(self.config.get("num_ut_ant", 1))
        else:
            num_rx = int(self.config.get("num_ut", 8))
            num_rx_ant = int(self.config.get("num_ut_ant", 1))
            num_tx = 1
            num_tx_ant = int(self.config.get("num_bs_ant", 32))
        return RayleighBlockFading(
            num_rx=num_rx,
            num_rx_ant=num_rx_ant,
            num_tx=num_tx,
            num_tx_ant=num_tx_ant,
        )

    def set_topology(self, batch_size: int) -> None:
        if self.config.get("channel_model_type", "tr38901") in (
            "rayleigh",
            "rician",
            "awgn",
            "inf",
        ):
            return
        topology = gen_topology(
            batch_size,
            self.config.get("num_ut", 8),
            self.config.get("scenario", "umi"),
            min_ut_velocity=self.config.get("min_ut_velocity", 0.0),
            max_ut_velocity=self.config.get("max_ut_velocity", 0.0),
        )
        if hasattr(self._channel_model, "set_topology"):
            self._channel_model.set_topology(*topology)

    def sample_frequency_response(self, batch_size: int) -> tf.Tensor:
        cmt = self.config.get("channel_model_type", "tr38901")
        if cmt == "awgn":
            direction = self.config.get("direction", "uplink")
            if direction == "uplink":
                num_rx, num_rx_ant = 1, int(self.config.get("num_bs_ant", 8))
                num_tx, num_tx_ant = int(self.config.get("num_ut", 4)), int(self.config.get("num_ut_ant", 1))
            else:
                num_rx, num_rx_ant = int(self.config.get("num_ut", 4)), int(self.config.get("num_ut_ant", 1))
                num_tx, num_tx_ant = 1, int(self.config.get("num_bs_ant", 8))
            shape = [
                batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,
                self.resource_grid.num_ofdm_symbols,
                self.resource_grid.fft_size,
            ]
            return tf.ones(shape, dtype=tf.complex64)
        with tf.device("/CPU:0"):
            h = self._generator(batch_size)
        if cmt == "rician":
            h = self._apply_rician_los(h)
        elif cmt == "inf":
            h = self._apply_inf_large_scale(h)
        return h

    def _apply_rician_los(self, h: tf.Tensor) -> tf.Tensor:
        """Blend in a spatially structured LOS component.

        An earlier revision added a *scalar* sqrt(K/(K+1)) to every antenna pair,
        subcarrier and symbol -- and to every user. Total power was right, but the
        LOS part was then the same rank-one all-ones vector for all users, which
        makes the multi-user channel matrix artificially ill-conditioned and has
        nothing to do with Rician fading.

        The LOS component of a link is a rank-one outer product of the receive and
        transmit array responses with a per-link phase:

            h_LOS = a_rx(theta_rx) a_tx(theta_tx)^H e^{j*phi}

        so each user gets its own angle of arrival and its own phase, and users
        remain spatially separable.
        """
        shape = h.shape.as_list()
        batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_sym, num_sc = shape
        k = float(self._rician_k_factor)
        los_scale = np.sqrt(k / (k + 1.0))
        nlos_scale = np.sqrt(1.0 / (k + 1.0))

        antenna_spacing = float(self.config.get("antenna_spacing", 0.5))
        # Independent AoA/AoD and phase per (batch, rx, tx) link.
        aoa = self._rng.uniform(-np.pi / 2, np.pi / 2, size=(batch, num_rx, num_tx))
        aod = self._rng.uniform(-np.pi / 2, np.pi / 2, size=(batch, num_rx, num_tx))
        phase = self._rng.uniform(-np.pi, np.pi, size=(batch, num_rx, num_tx))

        rx_index = np.arange(num_rx_ant).reshape(1, 1, 1, num_rx_ant)
        tx_index = np.arange(num_tx_ant).reshape(1, 1, 1, num_tx_ant)
        # Uniform linear array steering vectors. Entries are unit modulus so each
        # channel coefficient carries unit LOS power, matching the unit-power
        # convention of the normalised NLOS component; the K-factor blend then
        # preserves total power exactly.
        a_rx = np.exp(2j * np.pi * antenna_spacing * rx_index * np.sin(aoa)[..., None])
        a_tx = np.exp(2j * np.pi * antenna_spacing * tx_index * np.sin(aod)[..., None])

        # [batch, num_rx, num_tx, num_rx_ant, num_tx_ant] -> channel axis order.
        outer = a_rx[..., :, None] * np.conj(a_tx)[..., None, :]
        outer = outer * np.exp(1j * phase)[..., None, None]
        los = np.transpose(outer, (0, 1, 3, 2, 4))[:, :, :, :, :, None, None]
        los = np.broadcast_to(los, (batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_sym, num_sc))

        los_tf = tf.constant(los.astype(np.complex64))
        return tf.cast(los_scale, tf.complex64) * los_tf + tf.cast(nlos_scale, tf.complex64) * h

    def _inf_frequency_selectivity(
        self,
        shape: list[int],
        delay_spread_sec: np.ndarray,
    ) -> np.ndarray:
        """Frequency response of an InF tapped-delay-line channel.

        The large-scale model alone leaves the channel frequency-flat, because it
        only scales a single-tap block-fading realisation. A factory hall does
        have a delay profile -- TR 38.901 ties its RMS spread to the hall's
        volume-to-surface ratio -- so the small-scale channel is built here as an
        exponential PDP sampled at the signal bandwidth, with independent Rayleigh
        taps per link.

        Whether that produces *usable* frequency selectivity depends on the
        bandwidth: an indoor hall's 24-40 ns spread has a coherence bandwidth of
        5-8 MHz, so a narrow carrier sees a flat channel no matter how correct
        this model is. `frequency_selectivity_report()` quantifies that for the
        configured numerology.
        """
        batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_sym, num_sc = shape
        bandwidth = float(num_sc) * float(self.config.get("subcarrier_spacing", 30e3))
        sample_duration = 1.0 / max(bandwidth, 1.0)
        num_taps = max(int(self.config.get("cyclic_prefix_length", 20)), 1)

        link_shape = (batch, num_rx, num_rx_ant, num_tx, num_tx_ant)
        num_links = int(np.prod(link_shape))
        spreads = np.asarray(delay_spread_sec, dtype=np.float64).reshape(-1)
        if spreads.size == 1:
            spreads = np.repeat(spreads, num_links)
        elif spreads.size != num_links:
            # One spread per (batch, rx, tx) link; broadcast over the antennas.
            spreads = np.repeat(spreads, max(num_links // spreads.size, 1))[:num_links]

        # Independent Rayleigh taps, shaped by each link's power delay profile.
        taps = (
            self._rng.normal(size=(num_links, num_taps))
            + 1j * self._rng.normal(size=(num_links, num_taps))
        ) / np.sqrt(2.0)
        profiles = np.stack(
            [exponential_pdp(float(ds), num_taps, sample_duration) for ds in spreads]
        )
        taps = taps * np.sqrt(profiles)

        # Frequency response: zero-padded DFT over the delay axis.
        response = np.fft.fft(taps, n=num_sc, axis=-1)
        response = response.reshape(*link_shape, 1, num_sc)
        return np.broadcast_to(response, (*link_shape, num_sym, num_sc))

    def frequency_selectivity_report(self) -> dict[str, float]:
        """Is the configured carrier wide enough to see the hall's delay spread?"""
        room_dimensions = list(self.config.get("room_dimensions", [15.0, 15.0, 5.0]))
        volume, surface = hall_volume_and_surface(room_dimensions)
        delay_spread = float(inf_delay_spread_seconds(volume, surface)[0])
        bandwidth = float(self.resource_grid.fft_size) * float(
            self.config.get("subcarrier_spacing", 30e3)
        )
        coherence = coherence_bandwidth_hz(delay_spread)
        return {
            "hall_volume_m3": volume,
            "hall_surface_m2": surface,
            "rms_delay_spread_sec": delay_spread,
            "signal_bandwidth_hz": bandwidth,
            "coherence_bandwidth_hz": coherence,
            "delay_spread_samples": delay_spread * bandwidth,
            # Below ~1 the carrier cannot resolve the delay profile and every
            # frequency-domain estimator converges to the same answer.
            "selectivity_ratio": bandwidth / max(coherence, 1e-9),
        }

    def _apply_inf_large_scale(self, h: tf.Tensor) -> tf.Tensor:
        """Apply the TR 38.901 Indoor Factory channel: delay profile plus pathloss."""
        shape = h.shape.as_list()
        batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_sym, num_sc = shape

        room_dimensions = list(self.config.get("room_dimensions", [15.0, 15.0, 5.0]))
        machine_size_range = list(
            self.config.get("machine_size_range", [[0.5, 2.0], [0.5, 2.0], [0.5, 1.5]])
        )
        num_machines = int(self.config.get("num_machines", 5))
        clutter_density = float(
            self.config.get(
                "inf_clutter_density",
                clutter_density_from_layout(num_machines, machine_size_range, room_dimensions),
            )
        )
        large_scale = sample_inf_large_scale_gain(
            num_links=batch * num_rx * num_tx,
            scenario=str(self.config.get("scenario", "inf_dl")).lower(),
            carrier_frequency_hz=float(self.config.get("carrier_frequency", 3.5e9)),
            room_dimensions=room_dimensions,
            bs_height_m=float(room_dimensions[2]) - float(self.config.get("tx_height_offset", 1.0)),
            ut_height_m=float(self.config.get("rx_height", 1.0)),
            clutter_density=clutter_density,
            clutter_size_m=mean_clutter_size_m(machine_size_range),
            clutter_height_m=float(self.config.get("inf_clutter_height_m", 2.0)),
            enable_pathloss=bool(self.config.get("enable_pathloss", True)),
            enable_shadow_fading=bool(self.config.get("enable_shadow_fading", True)),
            rng=self._rng,
        )
        volume, surface = hall_volume_and_surface(room_dimensions)
        delay_spread = inf_delay_spread_seconds(
            float(self.config.get("inf_hall_volume_m3", volume)),
            float(self.config.get("inf_hall_surface_m2", surface)),
            rng=self._rng,
            num_links=batch * num_rx * num_tx,
        )
        large_scale["rms_delay_spread_sec"] = delay_spread
        self._last_large_scale = large_scale

        selective = self._inf_frequency_selectivity(shape, delay_spread)
        gain = large_scale["amplitude_gain"].reshape(batch, num_rx, 1, num_tx, 1, 1, 1)
        return tf.constant((selective * gain).astype(np.complex64))

    def last_large_scale_diagnostics(self) -> dict[str, object] | None:
        """LOS flags, distances and path loss from the most recent InF draw."""
        return self._last_large_scale

    def apply_frequency_response(self, x_rg: tf.Tensor, h_freq: tf.Tensor) -> tf.Tensor:
        with tf.device("/CPU:0"):
            return self._applier(x_rg, h_freq, no=None)

    def sample_noise(self, y_shape: tf.TensorShape | tf.Tensor, noise_var: tf.Tensor) -> tf.Tensor:
        if isinstance(y_shape, tf.TensorShape):
            if None in y_shape:
                raise ValueError("Cannot sample noise from a partially defined TensorShape.")
            shape = y_shape.as_list()
        else:
            shape = y_shape

        with tf.device("/CPU:0"):
            noise_std = tf.sqrt(tf.cast(noise_var, tf.float32) / 2.0)
            noise_re = tf.random.normal(shape, stddev=noise_std, dtype=tf.float32)
            noise_im = tf.random.normal(shape, stddev=noise_std, dtype=tf.float32)
        return tf.complex(noise_re, noise_im)

    def received_shape_from_response(self, h_freq: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(h_freq)
        return tf.stack([shape[0], shape[1], shape[2], shape[5], shape[6]])

    def get_channel_model(self):
        return self._channel_model
