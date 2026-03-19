from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid

logger = logging.getLogger(__name__)


def build_neural_estimator_model(num_ofdm_symbols: int, fft_size: int) -> tf.keras.Model:
    """Build the SRCNN-style Conv2D network used for channel estimation.

    Input shape:  [num_ofdm_symbols, fft_size, 2]  (real/imag channels)
    Output shape: [num_ofdm_symbols, fft_size, 2]
    """
    inputs = tf.keras.layers.Input(shape=(num_ofdm_symbols, fft_size, 2), name="h_ls_ri")
    x = tf.keras.layers.Conv2D(64, (9, 9), padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu", name="conv2")(x)
    outputs = tf.keras.layers.Conv2D(2, (5, 5), padding="same", activation="linear", name="conv3")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="neural_channel_estimator")


class NeuralChannelEstimator(Block):
    """Nonlinear channel estimator using a pre-trained Conv2D neural network.

    Applies an SRCNN-style network to refine the LS channel estimate.
    The network treats the 2D OFDM grid (time × frequency) as an image,
    with real and imaginary parts as separate channels.

    Training:
        Use ``docker compose run train-estimator`` to generate training data
        and train the network. Weights are saved to the path specified by
        ``model_path`` (default: ``models/neural_estimator.keras``).

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
                "Falling back to plain LS. Run 'docker compose run train-estimator' to train.",
                model_path,
                exc,
            )

    def _apply_network(self, h_ls: tf.Tensor) -> tf.Tensor:
        """Reshape h_ls to image format, run the network, reshape back."""
        original_shape = tf.shape(h_ls)
        # h_ls: [..., num_ofdm_symbols, fft_size] complex
        # Flatten all leading dims into a batch dimension
        flat = tf.reshape(h_ls, [-1, self.num_ofdm_symbols, self.fft_size])
        # Split complex → real/imag: [N, ofdm_syms, fft_size, 2]
        ri = tf.stack([tf.math.real(flat), tf.math.imag(flat)], axis=-1)
        ri = tf.cast(ri, tf.float32)
        # Run network
        ri_out = self._nn_model(ri, training=False)
        # Reassemble complex
        h_refined_flat = tf.complex(ri_out[..., 0], ri_out[..., 1])
        h_refined_flat = tf.cast(h_refined_flat, h_ls.dtype)
        # Restore original shape
        return tf.reshape(h_refined_flat, original_shape)

    def call(self, y: tf.Tensor, no: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        h_ls, err_var_ls = self._ls_estimator(y, no)

        if not self._nn_available:
            return h_ls, err_var_ls

        h_hat = self._apply_network(h_ls)
        err_var = err_var_ls * tf.cast(self.nn_err_var_scale, err_var_ls.dtype)
        return h_hat, err_var
