"""
Least-Squares channel estimation with interpolation.

Translated from LS_CE.m and interpolate.m (Athirah Mohd Ramly, UKM).
"""

import numpy as np
from scipy.interpolate import interp1d


def interpolate(H: np.ndarray, pilot_loc: np.ndarray,
                Nfft: int, method: str = 'linear') -> np.ndarray:
    """
    Interpolate channel estimates from pilot locations to all subcarriers.

    MATLAB original: interpolate.m

    Args:
        H: Channel estimate at pilot locations (complex 1-D array).
        pilot_loc: 0-based pilot indices.
        Nfft: Total number of subcarriers.
        method: 'linear' or 'spline'.

    Returns:
        H_interpolated: Channel estimate at all Nfft subcarriers.
    """
    H = np.asarray(H, dtype=np.complex128)
    pilot_loc = np.asarray(pilot_loc, dtype=np.float64)

    # Extrapolate at left edge if pilots don't start at index 0
    if pilot_loc[0] > 0:
        slope = (H[1] - H[0]) / (pilot_loc[1] - pilot_loc[0])
        H = np.concatenate([[H[0] - slope * pilot_loc[0]], H])
        pilot_loc = np.concatenate([[0.0], pilot_loc])

    # Extrapolate at right edge
    if pilot_loc[-1] < Nfft - 1:
        slope = (H[-1] - H[-2]) / (pilot_loc[-1] - pilot_loc[-2])
        H = np.concatenate([H, [H[-1] + slope * (Nfft - 1 - pilot_loc[-1])]])
        pilot_loc = np.concatenate([pilot_loc, [float(Nfft - 1)]])

    kind = 'linear' if method.lower().startswith('l') else 'cubic'
    f = interp1d(pilot_loc, H, kind=kind)
    H_interpolated = f(np.arange(Nfft, dtype=np.float64))
    return H_interpolated


def ls_ce(Y: np.ndarray, Xp: np.ndarray, pilot_loc: np.ndarray,
          Nfft: int, Nps: int, int_opt: str = 'linear') -> np.ndarray:
    """
    Least-Squares channel estimation.

    MATLAB original: LS_CE.m

    Args:
        Y: Frequency-domain received signal, 1-D array of length Nfft.
        Xp: Pilot symbols, 1-D array of length Np.
        pilot_loc: 0-based pilot subcarrier indices, length Np.
        Nfft: FFT size (total subcarriers, e.g. 640).
        Nps: Pilot spacing (e.g. 4).
        int_opt: 'linear' or 'spline'.

    Returns:
        H_LS: Channel estimate at all Nfft subcarriers.
    """
    Np = 512 // 4   # = 128, matches MATLAB: Np=512/4
    k = np.arange(Np)

    # LS estimate at pilot positions
    LS_est = Y[pilot_loc[:Np]] / Xp[:Np]

    method = 'linear' if int_opt.lower().startswith('l') else 'spline'
    H_LS = interpolate(LS_est, pilot_loc[:Np].astype(np.float64), Nfft, method)
    return H_LS
