from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid

logger = logging.getLogger(__name__)

# Must match src/training/train_estimator.py
_EBNO_MIN_DB: float = 0.0
_EBNO_MAX_DB: float = 20.0


def build_neural_estimator_model(num_ofdm_symbols: int, fft_size: int) -> tf.keras.Model:
    """Build the SNR-conditioned residual Conv2D network used for channel estimation.

    Input shape:  [N, num_ofdm_symbols, fft_size, 3]
                  channels: re(h_ls), im(h_ls), snr_map (normalised Eb/No)
    Output shape: [N, num_ofdm_symbols, fft_size, 2]
                  channels: re(delta), im(delta)  — the correction to add to h_ls

    At inference: h_hat = h_ls + complex(re(delta), im(delta))
    """
    inputs = tf.keras.layers.Input(
        shape=(num_ofdm_symbols, fft_size, 3), name="h_ls_snr"
    )
    x = tf.keras.layers.Conv2D(64, (9, 9), padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu", name="conv2")(x)
    outputs = tf.keras.layers.Conv2D(2, (5, 5), padding="same", activation="linear", name="conv3")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="neural_channel_estimator")


class NeuralChannelEstimator(Block):
    """Nonlinear channel estimator using a pre-trained SNR-conditioned Conv2D network.

    The network takes the LS channel estimate together with a normalised SNR
    map and predicts the *residual correction* (h_true - h_ls).  The final
    estimate is h_hat = h_ls + delta, which is easier to learn than direct
    channel regression because the corrections are small at high SNR.

    Training:
        Use ``docker compose run train-estimator`` (or ``python train.py``) to
        generate training data and train the network.  Weights are saved to the
        path specified by ``model_path`` (default: ``models/neural_estimator.keras``).

    If no weights file is found at ``model_path``, falls back to LS estimation
    and emits a warning.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        config: dict | None = None,
        model_path: str = "models/neural_estimator.keras",
        nn_err_var_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self._rg = resource_grid
        self._ls_estimator = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self.num_ofdm_symbols = int(resource_grid.num_ofdm_symbols)
        self.fft_size = int(resource_grid.fft_size)
        self.nn_err_var_scale = float(nn_err_var_scale)
        self._nn_available = False
        self._nn_model: tf.keras.Model | None = None

        try:
            self._nn_model = tf.keras.models.load_model(model_path, compile=False)
            self._nn_available = True
            logger.info("NeuralChannelEstimator: loaded weights from '%s'.", model_path)
        except (OSError, ValueError) as exc:
            logger.warning(
                "NeuralChannelEstimator: could not load model from '%s' (%s). "
                "Falling back to plain LS. Run 'python train.py' inside Docker to train.",
                model_path,
                exc,
            )

    def _apply_network(self, h_ls: tf.Tensor, ebno_db: tf.Tensor) -> tf.Tensor:
        """Reshape h_ls + SNR map, run the network, add residual back to h_ls."""
        original_shape = tf.shape(h_ls)
        # Flatten all leading dims into a batch dimension
        flat = tf.reshape(h_ls, [-1, self.num_ofdm_symbols, self.fft_size])
        # Split complex → real/imag: [N, ofdm_syms, fft_size, 2]
        ri = tf.stack([tf.math.real(flat), tf.math.imag(flat)], axis=-1)
        ri = tf.cast(ri, tf.float32)

        # Build normalised SNR map: scalar → [N, ofdm_syms, fft_size, 1]
        snr_norm = tf.clip_by_value(
            (tf.cast(ebno_db, tf.float32) - _EBNO_MIN_DB) / (_EBNO_MAX_DB - _EBNO_MIN_DB),
            0.0, 1.0,
        )
        n_flat = tf.shape(flat)[0]
        snr_map = tf.ones([n_flat, self.num_ofdm_symbols, self.fft_size, 1]) * snr_norm

        ri3 = tf.concat([ri, snr_map], axis=-1)  # [N, ofdm_syms, fft_size, 3]

        # Network predicts the correction (residual)
        delta_ri = self._nn_model(ri3, training=False)  # [N, ofdm_syms, fft_size, 2]

        delta = tf.complex(delta_ri[..., 0], delta_ri[..., 1])
        delta = tf.cast(tf.reshape(delta, tf.shape(flat)), h_ls.dtype)

        h_refined_flat = flat + delta
        return tf.reshape(h_refined_flat, original_shape)

    def call(self, y: tf.Tensor, no: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        h_ls, err_var_ls = self._ls_estimator(y, no)

        if not self._nn_available:
            return h_ls, err_var_ls

        # Derive Eb/No (dB) from noise variance for SNR conditioning
        no_mean = tf.reduce_mean(tf.cast(no, tf.float32))
        # Guard against log(0): clamp noise to a small positive value
        no_safe = tf.maximum(no_mean, 1e-10)
        ebno_db = -10.0 * tf.math.log(no_safe) / tf.math.log(10.0)

        h_hat = self._apply_network(h_ls, ebno_db)
        err_var = err_var_ls * tf.cast(self.nn_err_var_scale, err_var_ls.dtype)
        return h_hat, err_var
