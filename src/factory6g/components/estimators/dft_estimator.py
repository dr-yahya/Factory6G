from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid


class DFTChannelEstimator(Block):
    """DFT-denoising estimator using delay-domain tap truncation."""

    def __init__(
        self,
        resource_grid: ResourceGrid,
        config: dict | None = None,
        dft_tap_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self._rg = resource_grid
        self._ls_estimator = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self.fft_size = int(resource_grid.fft_size)
        self.cp_length = int(resource_grid.cyclic_prefix_length)
        self.dft_tap_ratio = float(dft_tap_ratio)
        target_taps = int(round(self.cp_length * max(self.dft_tap_ratio, 0.0)))
        self.tap_count = int(min(max(target_taps, 1), self.fft_size))

    def estimate_from_ls(
        self,
        h_ls: tf.Tensor,
        err_var_ls: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Truncate the delay profile, and report the error that actually causes.

        The declared error variance has two terms, and an earlier revision
        reported only the first:

        1. **Retained noise.** Keeping `tap_count` of `fft_size` delay taps keeps
           that fraction of the LS noise power.
        2. **Truncation bias.** The taps thrown away are not pure noise -- in a
           channel whose delay spread exceeds the truncation window they carry
           real signal, and discarding it is an error that does not shrink with
           SNR.

        Omitting term 2 made this estimator declare, at 20 dB on TR 38.901 UMi,
        about one seventy-first of its actual mean squared error. Because the
        LMMSE equalizer folds `err_var` into the effective noise, that
        understatement inflated its LLRs and flattered its BER relative to
        estimators that were more honest -- confounding the estimator comparison.

        Term 2 is directly measurable. By Parseval under TensorFlow's FFT
        convention, the per-subcarrier error power from dropping a set of taps is
        the summed energy of those taps. Their noise content is subtracted so the
        result estimates discarded *signal*.
        """
        h_delay = tf.signal.ifft(h_ls)
        tap_idx = tf.range(self.fft_size)
        keep = tap_idx < self.tap_count
        mask = tf.cast(keep, dtype=h_delay.dtype)
        h_filtered_delay = h_delay * mask
        h_dft = tf.signal.fft(h_filtered_delay)

        retained_fraction = float(self.tap_count) / float(self.fft_size)
        discarded_taps = float(self.fft_size - self.tap_count)

        # Energy in the discarded taps, per link and OFDM symbol.
        discarded_energy = tf.reduce_sum(
            tf.abs(h_delay * (1.0 - mask)) ** 2, axis=-1, keepdims=True
        )
        # Those taps also hold their share of the LS noise: with per-subcarrier
        # noise variance s^2, each delay tap carries s^2 / fft_size.
        err_var_ls_f = tf.cast(err_var_ls, tf.float32)
        noise_in_discarded = tf.reduce_mean(err_var_ls_f, axis=-1, keepdims=True) * (
            discarded_taps / float(self.fft_size)
        )
        truncation_bias = tf.maximum(discarded_energy - noise_in_discarded, 0.0)

        err_var_dft = err_var_ls_f * retained_fraction + truncation_bias
        return h_dft, err_var_dft

    def call(self, y: tf.Tensor, no: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        h_ls, err_var_ls = self._ls_estimator(y, no)
        return self.estimate_from_ls(h_ls, err_var_ls)
