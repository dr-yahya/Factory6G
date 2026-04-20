"""
Cyclic prefix operations for SCMA-OFDM.

Translated from add_CP.m and remove_CP.m (Athirah Mohd Ramly, UKM).
"""

import numpy as np


def add_cp(x: np.ndarray) -> np.ndarray:
    """
    Add cyclic prefix of length Ng = len(x) / 4.

    MATLAB original (add_CP.m):
        Ng = length(x)/4;
        y  = [x(:, end-Ng+1:end) x];   % shape 1×800

    Args:
        x: Time-domain signal, 1-D or row array of length L.

    Returns:
        y: Signal with CP prepended, length L + L//4.
    """
    x = x.flatten()
    Ng = len(x) // 4
    y = np.concatenate([x[-Ng:], x])  # CP from last Ng samples
    return y


def remove_cp(x: np.ndarray) -> np.ndarray:
    """
    Remove cyclic prefix of length Ng = 640/4 = 160.

    MATLAB original (remove_CP.m):
        Ng   = 640/4;   % 160
        Noff = 0;
        y    = x(:, Ng+1-Noff : Ng+640);   % output 1×640

    Args:
        x: Received time-domain signal with CP, 1-D array of length 800.

    Returns:
        y: Signal with CP removed, 1-D array of length 640.
    """
    x = x.flatten()
    Ng = 640 // 4    # 160
    Noff = 0
    # MATLAB: x(Ng+1-Noff : Ng+640)  → 0-based: x[Ng-Noff : Ng+640]
    y = x[Ng - Noff: Ng + 640]
    return y
