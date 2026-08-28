"""
Pilot insertion and removal for SCMA-OFDM.

Translated from add_pilot.m and remove_pilot.m (Athirah Mohd Ramly, UKM).

The original uses global variables (pilot_loc, Xp, data_loc).
Here they are returned explicitly from add_pilot() and passed to remove_pilot().
"""

import numpy as np
from typing import Tuple


def add_pilot(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Insert CAZAC pilot symbols into the frequency-domain signal.

    Pilot spacing (Nps) = 4: one pilot every 4 data subcarriers.
    Input: 4×128 SCMA output (K×N) → serialized to 512 elements → 640 with pilots.

    MATLAB original: add_pilot.m

    Args:
        u: SCMA output, complex array of shape (K, N) where K*N = 512.

    Returns:
        out: Frequency-domain signal with pilots, 1-D array of length 640.
        pilot_loc: 0-based indices of pilot subcarriers in `out` (length 128).
        Xp: Pilot symbol values (all 1.0+0j), length 128.
        data_loc: 0-based indices of data subcarriers in `out` (length 512).
    """
    # Serialize: MATLAB reshape(u', 512, 1)' — u is K×N, u.T is N×K
    # MATLAB reshape reads column-major from u' (N×K):
    #   column 0 of u' = row 0 of u = resource 0 over all N time slots
    #   column 1 of u' = row 1 of u = resource 1 over all N time slots, etc.
    # This is identical to reading u row-by-row (row-major = C order).
    x = u.flatten(order='C')  # shape (512,), complex

    Nps = 4
    Np = len(x) // Nps   # 128 pilots
    N_total = len(x) + Np  # 640

    pilot_loc = []
    data_loc = []
    Xp = []
    xp = np.zeros(N_total, dtype=np.complex128)

    j = 0  # pilot counter — increments at PILOT positions (matching MATLAB)
    for m in range(N_total):  # m is 0-based (MATLAB m=1..N → 0..N-1)
        if (m + 1) % (Nps + 1) == 1:  # MATLAB: mod(m, Nps+1) == 1
            # Pilot subcarrier
            xp[m] = 1.0 + 0j
            pilot_loc.append(m)
            Xp.append(xp[m])
            j += 1  # MATLAB: j increments here so next data uses x(m-j) correctly
        else:
            # Data subcarrier: MATLAB x(m-j) with 1-based m → x[m-j] with 0-based m
            xp[m] = x[m - j]
            data_loc.append(m)

    pilot_loc = np.array(pilot_loc, dtype=np.int32)
    data_loc = np.array(data_loc, dtype=np.int32)
    Xp = np.array(Xp, dtype=np.complex128)

    return xp, pilot_loc, Xp, data_loc


def remove_pilot(u: np.ndarray, data_loc: np.ndarray) -> np.ndarray:
    """
    Extract data subcarriers, removing pilot symbols.

    MATLAB original: remove_pilot.m

    Args:
        u: Equalized frequency-domain signal, 1-D array of length 640.
        data_loc: 0-based indices of data subcarriers (length 512).

    Returns:
        out: Data subcarriers reshaped to (4, 128), complex array.
    """
    uu = u.reshape(640)  # ensure 1-D
    data_temp = uu[data_loc]  # shape (512,)

    # MATLAB: Data = reshape(Data_temp', 128, 4)'  → shape (4, 128)
    # reshape(Data_temp', 128, 4) in MATLAB is column-major reshape of 512 elements
    # into 128 rows × 4 cols, then transposed to 4 rows × 128 cols.
    out = data_temp.reshape(128, 4, order='F').T  # shape (4, 128)
    return out
