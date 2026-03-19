from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid


class ISTAChannelEstimator(Block):
    """Nonlinear sparse channel estimator using ISTA (Iterative Shrinkage-Thresholding).

    Instead of hard tap truncation (DFT estimator), solves a sparse recovery
    problem in the delay domain via iterative soft-thresholding:

        min_h  ||h||_1   s.t.   IFFT(h_ls) ≈ h

    The soft-threshold operator  S_λ(x) = sign(x) * max(|x| - λ, 0)  is
    genuinely nonlinear, giving better denoising for sparse multipath channels
    compared to the linear DFT truncation approach.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        config: dict | None = None,
        num_iterations: int = 10,
        lambda_scale: float = 1.0,
        step_size: float = 0.5,
    ) -> None:
        super().__init__()
        self._rg = resource_grid
        self._ls_estimator = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self.fft_size = int(resource_grid.fft_size)
        self.num_iterations = int(max(num_iterations, 1))
        self.lambda_scale = float(lambda_scale)
        self.step_size = float(step_size)

    def _soft_threshold(self, x: tf.Tensor, threshold: tf.Tensor) -> tf.Tensor:
        """Complex soft-thresholding: shrink magnitude, preserve phase."""
        magnitude = tf.abs(x)
        shrunk = tf.maximum(magnitude - threshold, 0.0)
        # Avoid division by zero for zero-magnitude entries
        safe_mag = tf.maximum(magnitude, 1e-10)
        phase = x / tf.cast(safe_mag, x.dtype)
        return phase * tf.cast(shrunk, x.dtype)

    def _ista_refine(self, h_ls: tf.Tensor, no: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply ISTA in the delay domain to denoise h_ls.

        Returns the refined frequency-domain estimate and an approximate
        active-tap count for error-variance scaling.
        """
        # Broadcast noise std to complex dtype for threshold computation
        no_scalar = tf.reduce_mean(no)
        threshold = self.lambda_scale * tf.cast(tf.sqrt(no_scalar), h_ls.dtype)

        h_delay = tf.signal.ifft(h_ls)
        h = h_delay

        for _ in range(self.num_iterations):
            # Gradient step: residual in frequency domain, pulled back to delay
            residual = tf.signal.ifft(h_ls - tf.signal.fft(h))
            h = self._soft_threshold(
                h + tf.cast(self.step_size, h.dtype) * residual,
                threshold * tf.cast(self.step_size, threshold.dtype),
            )

        # Estimate active taps (magnitude above threshold) for error-variance scaling
        active_fraction = tf.reduce_mean(
            tf.cast(tf.abs(h) > tf.cast(threshold * self.step_size, tf.float32), tf.float32)
        )
        # Clamp to [1/fft_size, 1] to avoid degenerate scaling
        active_fraction = tf.clip_by_value(active_fraction, 1.0 / self.fft_size, 1.0)

        h_hat = tf.signal.fft(h)
        return h_hat, active_fraction

    def call(self, y: tf.Tensor, no: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        h_ls, err_var_ls = self._ls_estimator(y, no)
        h_hat, active_fraction = self._ista_refine(h_ls, no)
        err_var = err_var_ls * tf.cast(active_fraction, err_var_ls.dtype)
        return h_hat, err_var
