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


def delay_spread_ratio(h_ls: tf.Tensor, cp_length: int) -> tf.Tensor:
    """Fraction of delay-domain energy that falls outside the cyclic prefix.

    This is the statistic that actually distinguishes channels. A short,
    well-behaved delay profile concentrates its energy inside the CP window, so
    DFT truncation is nearly lossless; energy spilling past the CP means either a
    genuinely long profile or a noise-dominated estimate, and truncation would
    then discard signal. Either way it is a property of *this* channel
    realisation rather than of the Eb/No point.

    Returned per (batch, rx, rx_ant, tx, tx_ant, symbol) so the branch decision
    can be made per user rather than once for the whole batch.
    """
    h_delay = tf.signal.ifft(tf.cast(h_ls, tf.complex64))
    power = tf.abs(h_delay) ** 2
    fft_size = tf.shape(power)[-1]
    taps = tf.range(fft_size)
    inside = tf.cast(taps < tf.cast(cp_length, taps.dtype), power.dtype)
    inside_energy = tf.reduce_sum(power * inside, axis=-1)
    total_energy = tf.reduce_sum(power, axis=-1)
    return 1.0 - inside_energy / tf.maximum(total_energy, 1e-12)


def per_user_quality(h_ls: tf.Tensor, no: tf.Tensor, num_tx_axis: int = 3) -> tf.Tensor:
    """Per-user estimated SNR of the LS channel estimate.

    The previous proxy reduced the whole batch to one scalar,
    ``mean|h_ls|^2 / mean(no)``, which under a power-normalised channel is
    essentially ``1/no`` -- a deterministic function of the Eb/No point. That
    makes the "adaptive" estimator an SNR-threshold switch, not something that
    adapts to the channel. Reducing per user instead lets different users take
    different branches in the same slot, which is what adaptivity means here.
    """
    power = tf.abs(tf.cast(h_ls, tf.complex64)) ** 2
    rank = len(power.shape)
    reduce_axes = [axis for axis in range(rank) if axis != num_tx_axis]
    signal = tf.reduce_mean(power, axis=reduce_axes)
    noise = tf.maximum(tf.reduce_mean(tf.cast(no, tf.float32)), 1e-12)
    return signal / noise


class AdaptiveHybridChannelEstimator(Block):
    """
    Adaptive LS/DFT/LMMSE hybrid.

    Two selection modes:

    ``per_user`` (default)
        The blend weight is decided *per user* from two statistics the channel
        actually varies: the per-user LS SNR and the fraction of delay-domain
        energy falling outside the cyclic prefix. Different users can take
        different branches within the same slot, and the decision is made with
        TensorFlow ops only, so the estimator remains graph-traceable.

    ``scalar`` (legacy)
        One branch for the whole batch from a scalar SNR proxy. Under a
        power-normalised channel that proxy is essentially ``1/no``, i.e. a
        deterministic function of the Eb/No point -- so this mode is an SNR
        threshold switch rather than genuine channel adaptation. Retained for
        reproducing earlier result families.
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
        selection_mode: str = "per_user",
        leakage_reference: float = 0.25,
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
        self.selection_mode = str(selection_mode).lower()
        if self.selection_mode not in {"per_user", "scalar"}:
            raise ValueError(
                f"Unknown adaptive selection_mode '{selection_mode}'. Use 'per_user' or 'scalar'."
            )
        # Delay-domain leakage at which the DFT branch is fully distrusted.
        self.leakage_reference = float(leakage_reference)
        self._cp_length = int(resource_grid.cyclic_prefix_length)
        self.last_branch = "high"

    @staticmethod
    def quality_proxy(h_ls: tf.Tensor, no: tf.Tensor) -> float:
        signal_power = tf.reduce_mean(tf.abs(h_ls) ** 2)
        noise_power = tf.reduce_mean(tf.cast(no, tf.float32))
        ratio = signal_power / tf.maximum(noise_power, tf.constant(1e-12, tf.float32))
        return float(ratio.numpy())

    def _per_user_blend_weight(self, h_ls: tf.Tensor, no: tf.Tensor) -> tf.Tensor:
        """Weight on the LMMSE branch, per user, in [0, 1].

        Combines two observable statistics:

        * per-user LS SNR -- low SNR favours the heavier LMMSE smoothing;
        * delay-domain leakage past the CP -- high leakage means DFT truncation
          would discard signal, so it also favours LMMSE.

        Both are computed with TensorFlow ops only, so the whole estimator stays
        traceable. The old implementation called ``.numpy()`` mid-forward-pass,
        which forces eager execution for the entire simulation.
        """
        quality = per_user_quality(h_ls, no)  # [num_tx]
        # Linear ramp from full LMMSE below quality_low to pure DFT above quality_high.
        span = tf.maximum(self.quality_high - self.quality_low, 1e-6)
        snr_weight = tf.clip_by_value((self.quality_high - quality) / span, 0.0, 1.0)

        leakage = delay_spread_ratio(h_ls, self._cp_length)
        rank = len(leakage.shape)
        reduce_axes = [axis for axis in range(rank) if axis != 3]
        leakage_per_user = tf.reduce_mean(leakage, axis=reduce_axes)
        leakage_weight = tf.clip_by_value(
            leakage_per_user / tf.maximum(self.leakage_reference, 1e-6), 0.0, 1.0
        )

        weight = tf.maximum(snr_weight, leakage_weight)
        # Snap to the hard branches at the extremes so the cheap DFT-only and
        # LMMSE-only regimes are still exactly recovered.
        weight = tf.where(weight < 1e-3, tf.zeros_like(weight), weight)
        return tf.where(weight > 1.0 - 1e-3, tf.ones_like(weight), weight)

    def estimate_from_ls(
        self,
        h_ls: tf.Tensor,
        err_var_ls: tf.Tensor,
        no: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, str]:
        if self.selection_mode == "scalar":
            # Legacy behaviour: one branch for the whole batch, chosen from a
            # scalar SNR proxy. Kept for reproducing older result families.
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

        weight = self._per_user_blend_weight(h_ls, no)
        h_dft, err_dft = self._dft.estimate_from_ls(h_ls, err_var_ls)
        h_lmmse, err_lmmse = self._lmmse.estimate_from_ls(h_ls, err_var_ls, no)

        # Broadcast the per-user weight over the channel tensor's transmitter axis.
        rank = len(h_dft.shape)
        shape = [-1 if axis == 3 else 1 for axis in range(rank)]
        weight_b = tf.reshape(weight, shape)
        weight_c = tf.cast(weight_b, h_dft.dtype)

        h_hat = weight_c * h_lmmse + (1.0 - weight_c) * h_dft
        err_lmmse_t = tf.cast(err_lmmse, tf.float32)
        err_dft_t = tf.cast(err_dft, tf.float32)
        err_var = weight_b * err_lmmse_t + (1.0 - weight_b) * err_dft_t

        # Report a branch label with the same meaning as the legacy mode:
        # weight 1 is full LMMSE ("low" quality), weight 0 is pure DFT ("high").
        if tf.executing_eagerly():
            mean_weight = float(tf.reduce_mean(weight))
            self.last_branch = (
                "low"
                if mean_weight >= 1.0 - 1e-3
                else "high"
                if mean_weight <= 1e-3
                else "mid"
            )
        return h_hat, err_var, self.last_branch

    def call(self, y: tf.Tensor, no: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        h_ls, err_var_ls = self._ls_estimator(y, no)
        h_hat, err_var, _ = self.estimate_from_ls(h_ls, err_var_ls, no)
        return h_hat, err_var
