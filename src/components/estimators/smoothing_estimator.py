"""Smoothed LS-based channel estimator using 2D convolutional smoothing."""

from __future__ import annotations

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid

from ..config import SystemConfig


class SmoothedLSEstimator(Block):
    """
    Apply a separable 2D smoothing filter to the LS estimate across
    (ofdm_symbol, subcarrier) dimensions.
    """

    def __init__(self, config: SystemConfig, resource_grid: ResourceGrid,
                 kernel_time: int = 3, kernel_freq: int = 5) -> None:
        super().__init__()
        self._base = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self.kernel_time = kernel_time
        self.kernel_freq = kernel_freq
        # Build normalized box filters
        kt = tf.ones([kernel_time, 1, 1, 1], dtype=tf.float32) / float(kernel_time)
        kf = tf.ones([1, kernel_freq, 1, 1], dtype=tf.float32) / float(kernel_freq)
        self.kt = kt
        self.kf = kf

    def _smooth(self, x: tf.Tensor) -> tf.Tensor:
        """Apply separable smoothing on the last two dims: (symbols, subcarriers)."""
        # Expect shape: [B, rx, tx, ut, stream, n_sym, n_sc]
        # Move (sym, sc) to spatial dims for conv2d: NHWC
        real = tf.math.real(x)
        imag = tf.math.imag(x)
        for t in [real, imag]:
            pass
        def smooth_real_imag(t):
            t4 = tf.reshape(t, [-1, tf.shape(t)[-2], tf.shape(t)[-1], 1])
            # time smoothing (vertical)
            y = tf.nn.conv2d(t4, self.kt, strides=1, padding='SAME')
            # freq smoothing (horizontal)
            y = tf.nn.conv2d(y, self.kf, strides=1, padding='SAME')
            y = tf.reshape(y, tf.shape(t))
            return y
        real_s = smooth_real_imag(real)
        imag_s = smooth_real_imag(imag)
        return tf.complex(real_s, imag_s)

    def call(self, y: tf.Tensor, noise_variance: tf.Tensor):
        h_ls, err_var = self._base(y, noise_variance)
        h_sm = self._smooth(h_ls)
        return h_sm, err_var
