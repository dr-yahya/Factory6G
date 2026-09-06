"""LMMSE with the covariance that actually generated the channel.

`LMMSEChannelEstimator` assumes a fixed frequency correlation, `R = r^|dk|` with
`r = 0.98`, fixed at construction and independent of the channel being estimated.
Every claim measured against it -- that NMSE does not predict coded BER, that the
adaptive branch never wins, that sizing the truncation window beats smoothing --
is therefore measured against a *mismatched* LMMSE, and the first question an
examiner asks is whether the result is a property of LMMSE or of that particular
misspecification.

This estimator removes the question. It builds the frequency covariance from the
same exponential power delay profile `ChannelModel._apply_inf_large_scale` uses
to generate the channel, so it is the genie-matched linear MMSE estimator: the
best any linear estimator can do given second-order statistics. It is a
*reference arm*, not a proposed method -- nothing in the field gets the true PDP
for free.

Reading the comparison:

* if the adaptive window still wins on BLER against this, the contribution
  survives the strongest linear baseline available and the claim is safe;
* if this wins, it is the real bound and the gap is what the window has left to
  close.

Either way the estimator chapter gains the number it is currently missing.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from sionna.phy import Block
from sionna.phy.ofdm import LMMSEInterpolator, LSChannelEstimator, ResourceGrid

from ..inf_channel import (
    exponential_pdp,
    hall_volume_and_surface,
    inf_delay_spread_seconds,
)

SPEED_OF_LIGHT = 299_792_458.0


def genie_frequency_covariance(
    power_delay_profile: np.ndarray,
    num_subcarriers: int,
    fft_size: int,
) -> np.ndarray:
    r"""Frequency covariance of a channel with the given power delay profile.

    For taps of average power :math:`P_\ell` at delays :math:`\tau_\ell`,

    .. math::
        R^{(f)}_{u,v} = \sum_\ell P_\ell e^{-j 2 \pi \tau_\ell \Delta_f (u-v)}

    which is what `sionna.phy.ofdm.tdl_freq_cov_mat` computes for a TDL profile.
    That helper is not usable here: it takes a TDL model letter ("A".."E") and
    our factory channel is generated from an exponential PDP, not a TDL profile.
    The sum is the same either way.

    The taps sit on the sampling grid, :math:`\tau_\ell = \ell / (N \Delta_f)`
    with `N` the FFT size, so the exponent collapses to
    :math:`2 \pi \ell (u-v) / N` -- the covariance is the DFT of the PDP.
    """
    profile = np.asarray(power_delay_profile, dtype=np.float64)
    lag = np.arange(num_subcarriers)[:, None] - np.arange(num_subcarriers)[None, :]
    taps = np.arange(profile.size)[:, None, None]
    phase = -2.0j * np.pi * taps * lag[None, :, :] / float(fft_size)
    return np.sum(profile[:, None, None] * np.exp(phase), axis=0)


class LMMSEGenieChannelEstimator(Block):
    """LMMSE interpolation over Sionna's own interpolator, with a matched covariance.

    Unlike the other custom estimators, this one does not post-process an LS
    grid: the covariance goes into `LMMSEInterpolator` and Sionna performs the
    interpolation, which also means the error variance it declares is computed by
    the same code path that computes the estimate. `err_var_calibration` should
    therefore sit far closer to 1.0 than the hand-rolled estimators manage.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        config: dict | None = None,
        *,
        delay_spread_sec: float | None = None,
        order: str = "f-t",
        time_correlation_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        cfg = dict(config or {})
        self._rg = resource_grid
        self.fft_size = int(resource_grid.fft_size)

        pilot_pattern = resource_grid.pilot_pattern
        num_subcarriers = int(pilot_pattern.num_effective_subcarriers)
        num_ofdm_symbols = int(pilot_pattern.num_ofdm_symbols)

        self.delay_spread_sec = (
            float(delay_spread_sec)
            if delay_spread_sec is not None
            else self._delay_spread_from_config(cfg)
        )

        # The generating PDP: same construction, tap count and sampling grid as
        # ChannelModel._inf_frequency_selectivity.
        bandwidth = float(self.fft_size) * float(cfg.get("subcarrier_spacing", 30e3))
        sample_duration = 1.0 / max(bandwidth, 1.0)
        num_taps = max(int(cfg.get("cyclic_prefix_length", 20)), 1)
        self.power_delay_profile = exponential_pdp(
            self.delay_spread_sec, num_taps, sample_duration
        )

        cov_freq = genie_frequency_covariance(
            self.power_delay_profile, num_subcarriers, self.fft_size
        )
        cov_time = self._time_covariance(cfg, num_ofdm_symbols, time_correlation_floor)

        self._interpolator = LMMSEInterpolator(
            pilot_pattern,
            cov_mat_time=tf.constant(cov_time, dtype=tf.complex64),
            cov_mat_freq=tf.constant(cov_freq.astype(np.complex64)),
            order=order,
        )
        self._estimator = LSChannelEstimator(
            resource_grid, interpolator=self._interpolator
        )

    @staticmethod
    def _delay_spread_from_config(cfg: dict) -> float:
        """The hall's delay spread, resolved exactly as the channel resolves it."""
        room_dimensions = list(cfg.get("room_dimensions", [15.0, 15.0, 5.0]))
        volume, surface = hall_volume_and_surface(room_dimensions)
        override_volume = cfg.get("inf_hall_volume_m3")
        override_surface = cfg.get("inf_hall_surface_m2")
        if override_volume is not None and override_surface is not None:
            volume, surface = float(override_volume), float(override_surface)
        return float(inf_delay_spread_seconds(volume, surface)[0])

    def _time_covariance(
        self, cfg: dict, num_ofdm_symbols: int, floor: float
    ) -> np.ndarray:
        r"""Jakes time correlation across OFDM symbols.

        `sionna.phy.ofdm.tdl_time_cov_mat` computes the same
        :math:`J_0(\nu \Delta_t (u-v))`, but again only for a TDL model letter.
        Computing it directly keeps this consistent with the Jakes model
        `system.csi_feedback_delay_slots` already uses to age the feedback
        channel.

        With every UT static -- the default for the estimator families -- the
        matrix is all ones and therefore singular, so a small diagonal loading
        keeps the interpolator's inversion well posed. It is a regularizer, not a
        model term: at 1e-3 it is far below the estimation noise it competes with.
        """
        from scipy.special import j0  # scipy is already a hard dependency

        speed = float(cfg.get("max_ut_velocity", 0.0))
        carrier_frequency = float(cfg.get("carrier_frequency", 3.5e9))
        symbol_duration = float(self._rg.ofdm_symbol_duration)
        doppler = 2.0 * np.pi * speed / SPEED_OF_LIGHT * carrier_frequency

        lag = np.arange(num_ofdm_symbols)[:, None] - np.arange(num_ofdm_symbols)[None, :]
        cov = j0(doppler * symbol_duration * lag).astype(np.float64)
        cov += floor * np.eye(num_ofdm_symbols)
        return cov.astype(np.complex64)

    def call(self, y: tf.Tensor, no: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return self._estimator(y, no)
