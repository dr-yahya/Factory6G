"""Neural network-based channel estimator for the OFDM MIMO receiver."""

from __future__ import annotations

import tensorflow as tf
from pathlib import Path
from typing import Iterable, Optional, Sequence

from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid

from ..config import SystemConfig


class NeuralChannelEstimator(Block):
    """Channel estimator that refines LS estimates using a small neural network.

    The estimator first runs a conventional LS estimator to obtain an initial
    channel estimate and then applies a point-wise neural network that learns a
    nonlinear correction based on the real and imaginary parts of the LS
    estimate.  The network predicts refined real/imaginary parts for each
    resource element independently, which keeps the model lightweight while
    remaining fully differentiable.
    """

    def __init__(
        self,
        config: SystemConfig,
        resource_grid: ResourceGrid,
        hidden_units: Sequence[int] | None = None,
        activation: str = "relu",
        weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.resource_grid = resource_grid
        self.hidden_units = list(hidden_units or [32, 32])
        self.activation = activation
        self.weights_path = Path(weights_path) if weights_path else None

        # Base estimator (LS) used for initial estimation
        self._base_estimator = LSChannelEstimator(
            resource_grid,
            interpolation_type="nn",
        )

        # Build point-wise neural network
        self.model = self._build_network()

        self._weights_loaded = False
        if self.weights_path and self.weights_path.exists():
            self.load_weights(self.weights_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def call(self, y: tf.Tensor, noise_variance: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Estimate the channel response using neural refinement.

        Args:
            y: Received resource grid (complex tensor).
            noise_variance: Noise variance for each resource element.

        Returns:
            Tuple ``(h_pred, err_var)`` containing the refined channel estimate
            and the residual error variance (currently identical to the LS
            estimator's variance).
        """
        h_ls, err_var = self._base_estimator(y, noise_variance)

        # Build neural network input (real + imag stacked on last axis)
        features = tf.stack(
            [tf.math.real(h_ls), tf.math.imag(h_ls)],
            axis=-1,
        )

        refined = self.model(features)
        real_part = refined[..., 0]
        imag_part = refined[..., 1]
        h_pred = tf.complex(real_part, imag_part)

        return h_pred, err_var

    def save_weights(self, path: str | Path) -> None:
        """Save neural network weights to *path*."""
        self.model.save_weights(str(path))
        self._weights_loaded = True

    def load_weights(self, path: str | Path) -> None:
        """Load neural network weights from *path*."""
        self.model.build(input_shape=self._dummy_input_shape())
        self.model.load_weights(str(path))
        self._weights_loaded = True

    def compile(self, *args, **kwargs) -> None:
        """Proxy ``compile`` to the underlying keras model."""
        self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):  # pragma: no cover - passthrough
        """Proxy ``fit`` to the underlying keras model."""
        return self.model.fit(*args, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_network(self) -> tf.keras.Model:
        model = tf.keras.Sequential(name="neural_channel_estimator")
        for idx, units in enumerate(self.hidden_units):
            model.add(
                tf.keras.layers.Dense(
                    units,
                    activation=self.activation,
                    name=f"dense_{idx+1}",
                )
            )
        model.add(tf.keras.layers.Dense(2, activation=None, name="dense_out"))
        return model

    def _dummy_input_shape(self) -> tuple[int, ...]:
        """Return a dummy input shape for weight loading."""
        # The network operates point-wise on the last dimension of size 2.
        # We provide a shape with arbitrary batch size for building.
        return (None, 1, self.config.num_bs_ant, self.config.num_ut, 1,
                self.config.num_ofdm_symbols, self.config.fft_size, 2)


# --------------------------------------------------------------------------
# Utility functions for dataset generation (used by the training script)
# --------------------------------------------------------------------------
def stack_complex(real: tf.Tensor, imag: tf.Tensor) -> tf.Tensor:
    """Utility to stack real/imaginary parts along the last dimension."""
    return tf.stack([real, imag], axis=-1)
