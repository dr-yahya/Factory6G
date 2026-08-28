from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid

from .dft_estimator import DFTChannelEstimator
from .lmmse_estimator import LMMSEChannelEstimator


def select_quality_branch(quality: float, quality_low: float, quality_high: float) -> str:
    if quality < quality_low:
        return "low"
    if quality < quality_high:
        return "mid"
    return "high"


class AdaptiveHybridChannelEstimator(Block):
    """
    Adaptive LS/DFT/LMMSE hybrid:
    - low quality  -> full LMMSE smoothing
    - mid quality  -> blended DFT + LMMSE
    - high quality -> DFT-only denoising
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        config: dict | None = None,
        quality_low: float = 3.0,
        quality_high: float = 12.0,
        blend_mid_weight: float = 0.5,
        dft_tap_ratio: float = 1.0,
        lmmse_r_freq: float = 0.98,
        noise_bin_db: float = 0.5,
    ) -> None:
        super().__init__()
        self._ls_estimator = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self._dft = DFTChannelEstimator(resource_grid, dft_tap_ratio=dft_tap_ratio)
        self._lmmse = LMMSEChannelEstimator(
            resource_grid,
            r_freq=lmmse_r_freq,
            noise_bin_db=noise_bin_db,
        )
        self.quality_low = float(quality_low)
        self.quality_high = float(quality_high)
        self.blend_mid_weight = float(min(max(blend_mid_weight, 0.0), 1.0))
        self.last_branch = "high"

    @staticmethod
    def quality_proxy(h_ls: tf.Tensor, no: tf.Tensor) -> float:
        signal_power = tf.reduce_mean(tf.abs(h_ls) ** 2)
        noise_power = tf.reduce_mean(tf.cast(no, tf.float32))
        ratio = signal_power / tf.maximum(noise_power, tf.constant(1e-12, tf.float32))
        return float(ratio.numpy())

    def estimate_from_ls(
        self,
        h_ls: tf.Tensor,
        err_var_ls: tf.Tensor,
        no: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, str]:
        quality = self.quality_proxy(h_ls, no)
        branch = select_quality_branch(quality, self.quality_low, self.quality_high)
        self.last_branch = branch

        if branch == "low":
            h_hat, err_var = self._lmmse.estimate_from_ls(h_ls, err_var_ls, no)
            return h_hat, err_var, branch

        h_dft, err_dft = self._dft.estimate_from_ls(h_ls, err_var_ls)
        if branch == "high":
            return h_dft, err_dft, branch

        h_lmmse, err_lmmse = self._lmmse.estimate_from_ls(h_ls, err_var_ls, no)
        w = tf.cast(self.blend_mid_weight, h_dft.dtype)
        h_mid = w * h_lmmse + (1.0 - w) * h_dft
        err_mid = self.blend_mid_weight * err_lmmse + (1.0 - self.blend_mid_weight) * err_dft
        return h_mid, err_mid, branch

    def call(self, y: tf.Tensor, no: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        h_ls, err_var_ls = self._ls_estimator(y, no)
        h_hat, err_var, _ = self.estimate_from_ls(h_ls, err_var_ls, no)
        return h_hat, err_var
