"""The genie LMMSE reference arm and the covariance it is built from.

The point of this estimator is to be a *correct* LMMSE, so the covariance
construction is the part worth testing: if it is wrong, the reference arm is just
another mismatched smoother and settles nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from factory6g.components.estimators import genie_frequency_covariance
from factory6g.components.inf_channel import (
    exponential_pdp,
    hall_volume_and_surface,
    inf_delay_spread_seconds,
)

FFT_SIZE = 512
CYCLIC_PREFIX = 20
SUBCARRIER_SPACING = 120e3


def _small_hall_pdp() -> np.ndarray:
    volume, surface = hall_volume_and_surface([15.0, 15.0, 5.0])
    delay_spread = float(inf_delay_spread_seconds(volume, surface)[0])
    sample_duration = 1.0 / (FFT_SIZE * SUBCARRIER_SPACING)
    return exponential_pdp(delay_spread, CYCLIC_PREFIX, sample_duration)


def test_covariance_is_a_valid_hermitian_psd_matrix():
    cov = genie_frequency_covariance(_small_hall_pdp(), 64, FFT_SIZE)
    assert np.allclose(cov, cov.conj().T)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert eigenvalues.min() > -1e-9


def test_unit_power_profile_gives_unit_diagonal():
    """A normalised PDP means every subcarrier carries unit average power."""
    cov = genie_frequency_covariance(_small_hall_pdp(), 32, FFT_SIZE)
    assert np.allclose(np.diag(cov), 1.0, atol=1e-9)


def test_single_tap_channel_is_perfectly_correlated():
    """A flat channel has the all-ones covariance: one tap, no decorrelation."""
    sample_duration = 1.0 / (FFT_SIZE * SUBCARRIER_SPACING)
    flat = exponential_pdp(0.0, CYCLIC_PREFIX, sample_duration)
    cov = genie_frequency_covariance(flat, 8, FFT_SIZE)
    assert np.allclose(cov, np.ones((8, 8)))


def test_longer_delay_spread_decorrelates_faster():
    """More delay spread means a narrower coherence bandwidth."""
    sample_duration = 1.0 / (FFT_SIZE * SUBCARRIER_SPACING)
    narrow = exponential_pdp(20e-9, CYCLIC_PREFIX, sample_duration)
    wide = exponential_pdp(80e-9, CYCLIC_PREFIX, sample_duration)
    cov_narrow = genie_frequency_covariance(narrow, 64, FFT_SIZE)
    cov_wide = genie_frequency_covariance(wide, 64, FFT_SIZE)
    # Correlation with the most distant subcarrier in the block.
    assert abs(cov_wide[0, -1]) < abs(cov_narrow[0, -1])


def test_covariance_matches_the_generating_channel():
    """The covariance must match what the channel's own tap model produces.

    This is the load-bearing assertion: the analytic matrix is compared against
    a Monte Carlo estimate drawn with the same construction
    `ChannelModel._inf_frequency_selectivity` uses -- Rayleigh taps shaped by the
    PDP, zero-padded FFT to the subcarrier grid.
    """
    profile = _small_hall_pdp()
    num_subcarriers = 8
    trials = 200_000

    rng = np.random.default_rng(0)
    taps = (
        rng.normal(size=(trials, CYCLIC_PREFIX))
        + 1j * rng.normal(size=(trials, CYCLIC_PREFIX))
    ) / np.sqrt(2.0)
    taps = taps * np.sqrt(profile)
    freq = np.fft.fft(taps, n=FFT_SIZE, axis=-1)[:, :num_subcarriers]
    empirical = (freq.T @ freq.conj()) / trials

    analytic = genie_frequency_covariance(profile, num_subcarriers, FFT_SIZE)
    # Sampling error of a covariance estimate is order 1/sqrt(trials).
    assert np.abs(analytic - empirical).max() < 3.0 / np.sqrt(trials)


@pytest.mark.parametrize("num_subcarriers", [4, 16, 64])
def test_covariance_is_toeplitz_in_subcarrier_lag(num_subcarriers):
    """Correlation depends only on the lag, so every diagonal is constant."""
    cov = genie_frequency_covariance(_small_hall_pdp(), num_subcarriers, FFT_SIZE)
    for lag in range(num_subcarriers):
        diagonal = np.diag(cov, k=lag)
        assert np.allclose(diagonal, diagonal[0])
