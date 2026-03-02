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

from .antenna import AntennaConfig


class ChannelModel:
    """Channel wrapper that separates topology, channel sampling, and channel application."""

    def __init__(self, config: dict, antenna_config: AntennaConfig, resource_grid: ResourceGrid):
        self.config = config
        self.antenna_config = antenna_config
        self.resource_grid = resource_grid

        if self.config.get("channel_model_type", "tr38901") == "rayleigh":
            self._channel_model = self._create_rayleigh_channel()
        else:
            self._channel_model = self._create_tr38901_channel()

        self._generator = GenerateOFDMChannel(
            self._channel_model,
            resource_grid,
            normalize_channel=True,
        )
        self._applier = ApplyOFDMChannel()

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
        if self.config.get("channel_model_type") == "rayleigh":
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
        with tf.device("/CPU:0"):
            return self._generator(batch_size)

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
