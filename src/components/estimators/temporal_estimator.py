"""Temporal (EMA) LS-based channel estimator across OFDM symbols."""

from __future__ import annotations

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid

from ..config import SystemConfig


class TemporalEstimator(Block):
    """
    Apply exponential moving average over the OFDM symbol dimension to track
    slowly-varying channels. Uses LS per symbol then smooths temporally.
    """

    def __init__(self, config: SystemConfig, resource_grid: ResourceGrid,
                 alpha: float = 0.7) -> None:
        super().__init__()
        self._base = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def _ema_time(self, x: tf.Tensor) -> tf.Tensor:
        """EMA along the OFDM symbol dimension using tf.scan (graph-safe).
        x shape: [B, rx, tx, ut, stream, n_sym, n_sc]
        """
        # Move symbol axis to first for scan: (..., n_sym, n_sc) -> (n_sym, ... , n_sc)
        x_t = tf.transpose(x, perm=[-2, 0, 1, 2, 3, 4, -1])
        def scan_fn(prev, cur):
            return self.alpha * prev + (1.0 - self.alpha) * cur
        y_t = tf.scan(scan_fn, x_t)
        # Transpose back to original
        y = tf.transpose(y_t, perm=[1, 2, 3, 4, 5, 0, 6])
        return y

    def call(self, y: tf.Tensor, noise_variance: tf.Tensor):
        h_ls, err_var = self._base(y, noise_variance)
        real = self._ema_time(tf.math.real(h_ls))
        imag = self._ema_time(tf.math.imag(h_ls))
        h_ema = tf.complex(real, imag)
        return h_ema, err_var
