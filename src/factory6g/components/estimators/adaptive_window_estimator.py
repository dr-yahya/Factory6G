"""DFT channel estimation with a delay-truncation window fitted to the channel.

Why this exists
---------------
Three measurements pointed here.

1. On TR 38.901 UMi, LMMSE has the best NMSE at every Eb/No and a coded BER two
   orders of magnitude worse than DFT truncation.
2. On a TR 38.901 Indoor Factory hall, DFT wins on both metrics, and a
   clairvoyant DFT-versus-LMMSE branch selector beats plain DFT by 0.2%.
3. The difference between those two channels is not SNR. It is the delay spread
   against the truncation window: an indoor hall's profile fits inside a 20-tap
   window with an order of magnitude to spare, while UMi's overruns it.

So the quantity worth adapting is the **window**, not the choice of smoother.

Method
------
A fixed window trades two errors against each other, and the balance point moves
with SNR. Every retained tap admits a tap's worth of estimation noise; every
discarded tap throws away whatever signal it held. For a window of ``L`` taps
the mean squared error of the estimate is

    J(L) = sum_{k < L} nu_k + sum_{k >= L} s_k

with ``nu_k`` the estimation-noise power in delay tap ``k`` and ``s_k`` the
signal power there. This estimator evaluates ``J`` at every admissible window
length and keeps the minimiser, per user, per slot.

That rule is self-correcting in the direction a fixed window cannot be. Where
noise dominates the window tightens, because each extra tap costs more noise
than it recovers signal; where noise is small it opens out toward the cyclic
prefix, because the residual tail is then worth more than the noise it carries.
A detection test -- keep the taps that rise above the noise floor -- gets the
first regime right and the second one wrong, which is what measurement showed
before this was rewritten: an earlier threshold-plus-dynamic-range detector won
up to 2.3 dB at 0 dB Eb/No and lost 1.9 dB at 20 dB, because its window
*narrowed* as SNR rose instead of widening.

The noise is not white along the delay axis
-------------------------------------------
Getting ``nu_k`` right is the whole difficulty, and assuming it constant is
wrong by an order of magnitude. Each user's pilots occupy every ``D``-th
subcarrier, ``D`` being the number of co-scheduled transmitters, and the LS
estimate is carried across the gaps by nearest-neighbour interpolation. Holding
one value over ``D`` subcarriers is a rectangular filter in frequency, so along
the delay axis the interpolated noise is shaped by the Dirichlet kernel

    g(k) = | sin(pi k D / N) / (D sin(pi k / N)) |^2,

flat near zero delay and falling to its first null at ``k = N/D``. Its level is
fixed by fitting ``nu_k = A g(k)`` to the profile beyond the cyclic prefix,
which carries no usable channel energy.

This is not a small correction. Measured on this simulator, the ratio of
in-window noise to the level just past the cyclic prefix is 1.35 with four
co-scheduled users and 12.6 with sixteen; the kernel predicts 1.32 and 13. A
constant floor read from anywhere outside the window therefore undercharges
every retained tap, by a factor that changes with the user count -- so no single
fudge factor repairs it, and the window opens to the cap at every SNR.

The kernel alone is not the whole story either. Two errors reach the delay
profile and they are shaped differently: the interpolated pilot noise, which
follows ``g`` and shrinks with SNR, and the hold's own bias -- the channel's
variation between one pilot subcarrier and the next -- which does not shrink
with SNR and is far flatter in delay. Attributing all of the error to ``g``
makes the window collapse once the second term takes over, by up to 4 dB at
20 dB Eb/No on a large hall. So the fit carries both terms,

    nu_k = A g(k) + B,

with ``A`` and ``B`` recovered per user by least squares against the profile
beyond the cyclic prefix, constrained non-negative. Against an exhaustive search
over window length, that cost lands within 0.1 dB of the best achievable window
at every Eb/No from 0 to 20 dB, on both a small and a large factory hall.

Where the method applies
------------------------
The fit reads the noise off the delay taps beyond the cyclic prefix, so it
assumes the channel has no usable energy there. That is the design assumption of
OFDM itself and it holds by construction in the factory: on the mini-slot
numerology the measured energy past the cyclic prefix is 0.000% in both the
small and the large hall. It does not hold for TR 38.901 UMi at this numerology,
where 3.4% of the channel energy overruns the prefix, contaminates the fit, and
makes the window over-tighten at high SNR. UMi is the comparison arm rather than
a factory claim, but the limit is a real one and belongs with the result.

Everything is TensorFlow, so the estimator stays graph-traceable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid

# Axes of the 7-D channel tensor that are not the transmitter and not the
# subcarrier/tap axis. Reducing over these leaves one profile per user.
_REDUCE_AXES = [0, 1, 2, 4, 5]
_TX_AXIS = 3


class AdaptiveWindowChannelEstimator(Block):
    """DFT denoising whose truncation window is fitted per user, per slot."""

    def __init__(
        self,
        resource_grid: ResourceGrid,
        config: dict | None = None,
        min_taps: int = 1,
        max_taps: int | None = None,
        pilot_decimation: int | None = None,
        noise_margin: float = 1.0,
        min_relative_gain: float = 0.05,
    ) -> None:
        """
        Args:
            min_taps: floor on the window. One by default, because a single
                tap is the right answer on a channel flat across the signal
                bandwidth -- the narrowband control is exactly that, and a floor
                of two costs 3 dB there.
            max_taps: ceiling on the window. Defaults to the cyclic prefix
                length, which is the principled bound: energy arriving later
                than the CP causes inter-symbol interference and is not
                something the receiver can use anyway. Capping there means the
                window can only tighten below a fixed CP-length window, never
                widen past it, so at worst this matches plain DFT truncation.
            pilot_decimation: how many subcarriers apart this user's pilots sit,
                which sets the width of the nearest-neighbour hold and hence the
                Dirichlet shape of the interpolated noise. Defaults to the
                number of co-scheduled streams on the grid, which is how Sionna
                keeps the users' pilots orthogonal. Pass 1 for a grid whose
                estimation noise really is white in delay.
            noise_margin: multiplier on the fitted noise level. One is the
                unbiased rule and the default; above one biases the window
                tighter, below one looser.
            min_relative_gain: how much better than the full ``max_taps`` window
                the chosen one must be predicted to be, as a fraction of the
                full window's cost, before the estimator will tighten. The cost
                is an estimate, so where its minimum is shallow the ranking it
                gives is inside its own error and the fixed window is the safer
                bet. Five percent is the smallest margin that removes the
                high-SNR losses on a large hall -- 0.24 dB at 20 dB Eb/No with
                no margin -- while leaving the low-SNR gains untouched.
        """
        super().__init__()
        self._rg = resource_grid
        self._ls_estimator = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self.fft_size = int(resource_grid.fft_size)
        self.min_taps = max(int(min_taps), 1)
        cp_length = int(resource_grid.cyclic_prefix_length)
        self.max_taps = max(int(max_taps) if max_taps else cp_length, self.min_taps)
        if pilot_decimation is None:
            pilot_decimation = int(resource_grid.num_tx) * int(resource_grid.num_streams_per_tx)
        self.pilot_decimation = max(int(pilot_decimation), 1)
        self.noise_margin = float(noise_margin)
        self.min_relative_gain = max(float(min_relative_gain), 0.0)

        self._noise_shape = tf.constant(self._dirichlet_shape(), tf.float32)  # [N]
        # Diagnostic: mean window length chosen on the most recent call.
        self.last_mean_taps = float(self.max_taps)

    def _dirichlet_shape(self) -> np.ndarray:
        """Delay-domain power response of a ``D``-wide nearest-neighbour hold.

        ``|sin(pi k D / N) / (D sin(pi k / N))|^2``, normalised to one at zero
        delay. With ``D = 1`` this is flat, which is the white-noise case.
        """
        k = np.arange(self.fft_size, dtype=np.float64)
        denominator = self.pilot_decimation * np.sin(np.pi * k / self.fft_size)
        numerator = np.sin(np.pi * k * self.pilot_decimation / self.fft_size)
        shape = np.ones(self.fft_size, dtype=np.float64)
        nonzero = np.abs(denominator) > 1e-12
        shape[nonzero] = (numerator[nonzero] / denominator[nonzero]) ** 2
        # Exact nulls would make the fit ill-conditioned and would claim that a
        # tap costs nothing to keep; hold the shape above a small fraction of
        # its mean instead.
        return np.maximum(shape, shape.mean() * 1e-3)

    def _noise_per_tap(self, power: tf.Tensor, err_var_ls: tf.Tensor) -> tf.Tensor:
        """Per-user estimation-noise power in each delay tap, [num_tx, fft_size].

        Fits ``nu_k = A g(k) + B`` per user by least squares against the
        observed profile beyond the cyclic prefix, which carries no usable
        channel energy. Only the two levels are estimated; the shapes are known
        -- the Dirichlet kernel for the interpolated pilot noise, flat for the
        hold bias -- which is what lets the noise inside the window be inferred
        at all. It can never be measured there directly, because that is exactly
        where the signal is.
        """
        observed = power[..., self.max_taps :]  # [num_tx, M]
        shape = self._noise_shape[tf.newaxis, self.max_taps :]  # [1, M]

        shape_mean = tf.reduce_mean(shape)
        observed_mean = tf.reduce_mean(observed, axis=-1, keepdims=True)
        centred_shape = shape - shape_mean
        covariance = tf.reduce_sum(
            (observed - observed_mean) * centred_shape, axis=-1, keepdims=True
        )
        variance = tf.reduce_sum(centred_shape**2) + 1e-30
        # Both terms are powers, so a negative fit is noise in the fit itself
        # rather than a real result; clipping keeps the cost function positive.
        kernel_level = tf.maximum(covariance / variance, 0.0)
        flat_level = tf.maximum(observed_mean - kernel_level * shape_mean, 0.0)

        noise = kernel_level * self._noise_shape[tf.newaxis, :] + flat_level
        # Guard against a degenerate fit: the declared LS error variance is an
        # underestimate of the truth, so it is a valid floor and never a cap.
        declared = tf.reduce_mean(tf.cast(err_var_ls, tf.float32)) / float(self.fft_size)
        return tf.maximum(noise, declared * 1e-3) * self.noise_margin

    def _window_mask(self, h_delay: tf.Tensor, err_var_ls: tf.Tensor) -> tf.Tensor:
        """Per-user keep-mask over delay taps, shape [num_tx, fft_size].

        The profile is averaged over the batch, receive antennas and OFDM
        symbols first, which keeps its variance small enough that the cost curve
        has a well-defined minimum rather than a noisy one.
        """
        power = tf.reduce_mean(tf.abs(h_delay) ** 2, axis=_REDUCE_AXES)  # [num_tx, N]
        noise = self._noise_per_tap(power, err_var_ls)  # [num_tx, N]

        # Signal power per tap: the profile with the noise removed. The profile
        # is an estimate, so taps below the noise come out negative; clipping at
        # zero is what stops them from paying a spurious dividend for being
        # discarded.
        signal = tf.maximum(power - noise, 0.0)

        # J(L) = sum_{k < L} nu_k + sum_{k >= L} s_k, for every L at once: an
        # exclusive forward cumulative sum of the noise plus an inclusive
        # reverse cumulative sum of the signal.
        cost = tf.cumsum(noise, axis=-1, exclusive=True) + tf.cumsum(
            signal, axis=-1, reverse=True
        )

        admissible = tf.logical_and(
            tf.range(self.fft_size) >= self.min_taps,
            tf.range(self.fft_size) <= self.max_taps,
        )
        masked = tf.where(admissible[tf.newaxis, :], cost, tf.fill(tf.shape(cost), cost.dtype.max))
        window = tf.argmin(masked, axis=-1, output_type=tf.int32)  # [num_tx]

        # Tighten only on a clear margin. Where the cost curve is shallow near
        # the full window, the ordering it reports is inside the fit's own
        # error, and the incumbent is the safer choice.
        best = tf.reduce_min(masked, axis=-1)
        full = cost[:, self.max_taps]
        window = tf.where(
            best < (1.0 - self.min_relative_gain) * full, window, self.max_taps
        )

        index = tf.range(self.fft_size)
        return tf.cast(index[tf.newaxis, :] < window[:, tf.newaxis], tf.float32)

    def estimate_from_ls(
        self,
        h_ls: tf.Tensor,
        err_var_ls: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        h_delay = tf.signal.ifft(tf.cast(h_ls, tf.complex64))
        keep = self._window_mask(h_delay, err_var_ls)  # [num_tx, N]

        rank = len(h_delay.shape)
        shape = [-1 if axis == _TX_AXIS else 1 for axis in range(rank)]
        shape[-1] = self.fft_size
        keep_b = tf.reshape(keep, shape)

        h_filtered = h_delay * tf.cast(keep_b, tf.complex64)
        h_hat = tf.signal.fft(h_filtered)

        # Declared error: retained noise plus the signal the window discarded.
        # Reporting only the first term is what let fixed-window DFT understate
        # its error seventy-one-fold and flatter its BER through the equalizer.
        num_kept = tf.reduce_sum(keep, axis=-1, keepdims=True)  # [num_tx, 1]
        err_var_f = tf.cast(err_var_ls, tf.float32)
        retained_noise = tf.reshape(
            num_kept / float(self.fft_size), [-1 if a == _TX_AXIS else 1 for a in range(rank)]
        ) * tf.reduce_mean(err_var_f, axis=-1, keepdims=True)

        discarded = tf.reduce_sum(
            tf.abs(h_delay * tf.cast(1.0 - keep_b, tf.complex64)) ** 2, axis=-1, keepdims=True
        )
        noise_in_discarded = tf.reduce_mean(err_var_f, axis=-1, keepdims=True) * tf.reshape(
            (float(self.fft_size) - num_kept) / float(self.fft_size),
            [-1 if a == _TX_AXIS else 1 for a in range(rank)],
        )
        truncation_bias = tf.maximum(discarded - noise_in_discarded, 0.0)

        if tf.executing_eagerly():
            self.last_mean_taps = float(tf.reduce_mean(num_kept))

        return h_hat, retained_noise + truncation_bias

    def call(self, y: tf.Tensor, no: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        h_ls, err_var_ls = self._ls_estimator(y, no)
        return self.estimate_from_ls(h_ls, err_var_ls)
