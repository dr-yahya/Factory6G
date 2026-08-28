"""
MMSE channel estimation.

Translated from MMSE_CE.m (Athirah Mohd Ramly, UKM).
"""

import numpy as np


def mmse_ce(Y: np.ndarray, Xp: np.ndarray, pilot_loc: np.ndarray,
            Nfft: int, Nps: int, snr_db: float) -> np.ndarray:
    """
    MMSE channel estimation using Wiener filtering.

    MATLAB original: MMSE_CE.m

    Args:
        Y: Frequency-domain received signal, 1-D complex array of length Nfft.
        Xp: Pilot symbols, 1-D complex array of length Np.
        pilot_loc: 0-based pilot subcarrier indices, length Np.
        Nfft: Total FFT size (e.g. 640).
        Nps: Pilot spacing (e.g. 4).
        snr_db: SNR in dB at the current Eb/N0 point.

    Returns:
        H_MMSE: MMSE channel estimate at all Nfft subcarriers, 1-D complex array.
    """
    snr = 10.0 ** (snr_db * 0.1)
    Np = 512 // 4  # = 128

    # LS estimate at pilot positions
    H_tilde = Y[pilot_loc[:Np]] / Xp[:Np]  # shape (Np,)

    # Wiener filter matrices
    tau_rms = 1.05e-6    # RMS delay spread
    df = 1.0 / Nfft

    # Rhp: cross-correlation matrix (Nfft × Np)
    K1 = np.arange(Nfft)[:, np.newaxis]    # (Nfft, 1)
    K2 = np.arange(Np)[np.newaxis, :]      # (1, Np)
    j2pi_tau_df = 1j * 2 * np.pi * tau_rms * df
    Rhp = 1.0 / (1.0 + j2pi_tau_df * Nps * (K1 - K2))  # (Nfft, Np)

    # Rpp: auto-correlation matrix at pilots (Np × Np)
    K3 = np.arange(Np)[:, np.newaxis]
    K4 = np.arange(Np)[np.newaxis, :]
    rf2 = 1.0 / (1.0 + j2pi_tau_df * Nps * (K3 - K4))  # (Np, Np)
    Rpp = rf2 + np.eye(Np) / snr            # regularized

    # MMSE estimate: H = Rhp * inv(Rpp) * H_tilde
    H_MMSE = (Rhp @ np.linalg.solve(Rpp, H_tilde)).flatten()
    return H_MMSE
