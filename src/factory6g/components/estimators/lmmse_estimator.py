from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid


class LMMSEChannelEstimator(Block):
    """
    Frequency-domain LMMSE smoother with spectral shrinkage acceleration.

    It computes LS once and applies:
      H_hat = H_ls * U * diag(lambda/(lambda+sigma_n)) * U^H
    where U/lambda come from one-time eigendecomposition of the Hermitian
    correlation matrix R_freq.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        config: dict | None = None,
        r_freq: float = 0.98,
        noise_bin_db: float = 0.5,
    ) -> None:
        super().__init__()
        self._rg = resource_grid
        self._ls_estimator = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self.fft_size = int(resource_grid.fft_size)
        self.r_freq = float(r_freq)
        self.noise_bin_db = float(noise_bin_db)

        k_indices = np.arange(self.fft_size)
        delta_k = np.abs(k_indices[:, None] - k_indices[None, :])
        correlation = (self.r_freq ** delta_k).astype(np.float32)
        self.R_freq = tf.cast(correlation, dtype=tf.complex64)

        eigvals, eigvecs = np.linalg.eigh(correlation.astype(np.float64))
        eigvals = np.maximum(eigvals.astype(np.float32), 1e-9)
        self._eigvals = tf.constant(eigvals, dtype=tf.float32)
        self._eigvecs = tf.constant(eigvecs.astype(np.complex64), dtype=tf.complex64)
        self._shrinkage_cache: dict[float, tf.Tensor] = {}

    def _avg_noise(self, no: tf.Tensor) -> float:
        """Eager-only scalar noise level, used for the shrinkage cache."""
        avg_no = float(tf.reduce_mean(tf.cast(no, tf.float32)).numpy())
        return max(avg_no, 1e-12)

    def _avg_noise_tensor(self, no: tf.Tensor) -> tf.Tensor:
        """Traceable mean noise level."""
        return tf.maximum(tf.reduce_mean(tf.cast(no, tf.float32)), 1e-12)

    def _shrinkage_tensor(self, no: tf.Tensor) -> tf.Tensor:
        """Shrinkage computed with TF ops only, so the estimator can be traced.

        The cached variant below keys a Python dict by a float noise level, which
        cannot work under tf.function. Recomputing is an elementwise op over
        fft_size values -- negligible next to the two N x N matmuls that follow.
        """
        noise = self._avg_noise_tensor(no)
        if self.noise_bin_db > 0.0:
            # Quantise to the same noise bins the cached path uses, so the two
            # modes agree.
            noise_db = 10.0 * (tf.math.log(noise) / tf.math.log(10.0))
            binned_db = tf.round(noise_db / self.noise_bin_db) * self.noise_bin_db
            noise = tf.pow(10.0, binned_db / 10.0)
        return tf.cast(self._eigvals / (self._eigvals + noise), tf.complex64)

    def _noise_bin_key(self, noise_linear: float) -> float | None:
        if self.noise_bin_db <= 0.0:
            return None
        noise_db = 10.0 * np.log10(max(noise_linear, 1e-12))
        return round(noise_db / self.noise_bin_db) * self.noise_bin_db

    def _shrinkage_for_noise(self, noise_linear: float) -> tf.Tensor:
        key = self._noise_bin_key(noise_linear)
        if key is None:
            noise_eff = noise_linear
        else:
            if key in self._shrinkage_cache:
                return self._shrinkage_cache[key]
            noise_eff = 10.0 ** (key / 10.0)

        shrinkage = self._eigvals / (self._eigvals + noise_eff)
        shrinkage = tf.cast(shrinkage, tf.complex64)
        if key is not None:
            self._shrinkage_cache[key] = shrinkage
        return shrinkage

    def estimate_from_ls(
        self,
        h_ls: tf.Tensor,
        err_var_ls: tf.Tensor,
        no: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if tf.executing_eagerly():
            shrinkage = self._shrinkage_for_noise(self._avg_noise(no))
        else:
            shrinkage = self._shrinkage_tensor(no)

        flat = tf.reshape(tf.cast(h_ls, tf.complex64), [-1, self.fft_size])
        projected = tf.matmul(flat, self._eigvecs)
        projected *= shrinkage
        h_lmmse_flat = tf.matmul(projected, tf.transpose(tf.math.conj(self._eigvecs)))
        h_lmmse = tf.reshape(h_lmmse_flat, tf.shape(h_ls))

        smoothing_factor = tf.math.real(tf.reduce_mean(shrinkage))
        err_var_lmmse = err_var_ls * smoothing_factor
        return h_lmmse, err_var_lmmse

    def call(self, y: tf.Tensor, no: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        h_ls, err_var_ls = self._ls_estimator(y, no)
        return self.estimate_from_ls(h_ls, err_var_ls, no)
