"""Tests for the delay-window-adapting DFT estimator.

The estimator's claim is that it sizes the truncation window to the channel
rather than to the resource grid. These tests pin the three things that claim
rests on: the window responds to the delay spread, it responds to the SNR in the
right direction, and it never exceeds the cyclic prefix.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
from sionna.phy.ofdm import ResourceGrid

from factory6g.components.estimators import AdaptiveWindowChannelEstimator

FFT = 64
CP = 16
NUM_TX = 2


def _white_noise_estimator(**kwargs) -> AdaptiveWindowChannelEstimator:
    """An estimator told its noise is white in delay.

    The synthetic fixtures below add white noise directly to the channel rather
    than running it through Sionna's pilot interpolation, so the Dirichlet
    shaping the estimator assumes by default is not present. ``pilot_decimation
    = 1`` makes the shape flat and matches the fixture.
    """
    return AdaptiveWindowChannelEstimator(_grid(), pilot_decimation=1, **kwargs)


def _grid(num_tx: int = NUM_TX) -> ResourceGrid:
    return ResourceGrid(
        num_ofdm_symbols=4,
        fft_size=FFT,
        subcarrier_spacing=30000.0,
        num_tx=num_tx,
        num_streams_per_tx=1,
        cyclic_prefix_length=CP,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=[0, 2],
    )


def _synthetic_ls(taps_per_user, noise_std, seed=0):
    """LS estimate of an exponentially decaying channel plus white noise.

    Returns ``(h_ls, err_var_ls)`` shaped as Sionna's 7-D channel tensor.
    """
    rng = np.random.default_rng(seed)
    shape = (4, 1, 2, len(taps_per_user), 1, 4, FFT)  # batch, rx, rx_ant, tx, tx_ant, sym, sc
    delay = np.zeros(shape, dtype=np.complex64)
    for tx, num_taps in enumerate(taps_per_user):
        profile = np.exp(-np.arange(num_taps) / max(num_taps / 3.0, 1.0))
        profile = profile / np.sqrt(profile.sum())
        head = list(shape[:3]) + [1] + list(shape[4:6]) + [num_taps]
        draw = rng.normal(size=head) + 1j * rng.normal(size=head)
        delay[:, :, :, tx : tx + 1, :, :, :num_taps] = draw * profile / np.sqrt(2.0)
    h_clean = np.fft.fft(delay, axis=-1)
    noise = noise_std * (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)
    h_ls = tf.constant(h_clean + noise, dtype=tf.complex64)
    err_var = tf.fill(shape, tf.constant(noise_std**2, tf.float32))
    return h_ls, err_var


def _windows(estimator, h_ls, err_var):
    mask = estimator._window_mask(tf.signal.ifft(h_ls), err_var)
    return np.sum(mask.numpy(), axis=-1)


def test_window_defaults_to_the_cyclic_prefix_and_never_exceeds_it():
    """Energy past the CP is inter-symbol interference, so it is never retained."""
    estimator = _white_noise_estimator()
    assert estimator.max_taps == CP

    # A channel whose taps run the full length of the FFT: the window must still
    # stop at the cyclic prefix rather than chase the energy beyond it.
    h_ls, err_var = _synthetic_ls([FFT, FFT], noise_std=1e-4)
    assert np.all(_windows(estimator, h_ls, err_var) <= CP)


def test_window_tracks_the_per_user_delay_spread():
    """Two users on the same grid, different delay spreads, different windows."""
    estimator = _white_noise_estimator()
    h_ls, err_var = _synthetic_ls([2, 14], noise_std=0.05)
    short, long_ = _windows(estimator, h_ls, err_var)
    assert short < long_, f"short-spread user got {short} taps, long-spread user {long_}"


def test_window_widens_as_noise_falls():
    """The MMSE trade-off must move the right way with SNR.

    Each retained tap costs one tap of noise and buys back whatever signal it
    holds, so a quieter channel should support a longer window. An earlier
    detection-threshold version of this estimator got this backwards, which is
    what cost it 1.9 dB at 20 dB Eb/No.
    """
    estimator = _white_noise_estimator()
    # A long, slowly decaying profile, so the window has somewhere to grow into
    # rather than stopping at the last non-zero tap of the channel itself.
    noisy = _windows(estimator, *_synthetic_ls([40, 40], noise_std=2.0))
    quiet = _windows(estimator, *_synthetic_ls([40, 40], noise_std=0.02))
    assert np.mean(noisy) < np.mean(quiet), f"noisy {noisy} vs quiet {quiet}"


def test_declared_error_variance_includes_the_truncation_bias():
    """A window that discards signal must own up to it.

    Declaring only the retained noise is what let the fixed-window estimator
    understate its error and flatter its BER through the equalizer.
    """
    estimator = _white_noise_estimator(max_taps=3)
    h_ls, err_var = _synthetic_ls([12, 12], noise_std=0.01)
    _, declared = estimator.estimate_from_ls(h_ls, err_var)
    # The window keeps at most 3 of 64 taps, so retained noise alone would be
    # under a twentieth of the LS error variance. The discarded signal dominates.
    retained_only = float(tf.reduce_mean(err_var)) * 3.0 / FFT
    assert float(tf.reduce_mean(declared)) > 10.0 * retained_only


def test_shapes_dtypes_and_graph_mode():
    estimator = AdaptiveWindowChannelEstimator(_grid())  # default Dirichlet shape
    h_ls, err_var = _synthetic_ls([6, 6], noise_std=0.05)

    h_hat, declared = estimator.estimate_from_ls(h_ls, err_var)
    assert h_hat.shape == h_ls.shape
    assert h_hat.dtype == tf.complex64
    assert declared.dtype == tf.float32
    assert np.all(np.isfinite(declared.numpy()))

    compiled = tf.function(estimator.estimate_from_ls)
    h_graph, declared_graph = compiled(h_ls, err_var)
    np.testing.assert_allclose(h_graph.numpy(), h_hat.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(declared_graph.numpy(), declared.numpy(), rtol=1e-5, atol=1e-6)


def test_min_taps_floor_survives_a_pure_noise_profile():
    estimator = _white_noise_estimator(min_taps=4)
    shape = (2, 1, 1, NUM_TX, 1, 4, FFT)
    rng = np.random.default_rng(3)
    h_ls = tf.constant(rng.normal(size=shape) + 1j * rng.normal(size=shape), tf.complex64)
    err_var = tf.ones(shape, tf.float32)
    assert np.all(_windows(estimator, h_ls, err_var) >= 4)


def test_dirichlet_noise_shape_matches_the_pilot_hold():
    """The assumed delay-domain noise shape is the hold's own power response.

    Nearest-neighbour interpolation across pilots ``D`` subcarriers apart is a
    ``D``-wide rectangular hold in frequency, so the interpolated noise follows
    the Dirichlet kernel in delay: flat at zero delay, first null at ``N / D``.
    Getting this wrong is what made an earlier flat-floor version undercharge
    every retained tap by up to a factor of twelve.
    """
    estimator = AdaptiveWindowChannelEstimator(_grid(num_tx=4), pilot_decimation=4)
    shape = estimator._noise_shape.numpy()

    assert shape[0] == pytest.approx(1.0, rel=1e-6)
    assert shape[FFT // 4] < 1e-2 * shape[0]  # the first null, at N / D
    assert np.all(np.diff(shape[: FFT // 4]) <= 1e-6)  # monotone into that null

    # With one stream the hold is a single subcarrier wide and the shape is flat.
    flat = AdaptiveWindowChannelEstimator(_grid(num_tx=1), pilot_decimation=1)
    np.testing.assert_allclose(flat._noise_shape.numpy(), np.ones(FFT), rtol=1e-6)


def test_pilot_decimation_defaults_to_the_co_scheduled_stream_count():
    grid = _grid(num_tx=4)
    estimator = AdaptiveWindowChannelEstimator(grid)
    assert estimator.pilot_decimation == 4
