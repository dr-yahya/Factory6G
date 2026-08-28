"""
Utility math functions translated from Dr. Athirah's MATLAB.

Sources:
  log_sum_exp.m  (Alexander B. Sergienko, 2015, SPbETU)
  bertoper.m     (Athirah Mohd Ramly, UKM)
"""

import numpy as np


def log_sum_exp(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable log-sum-exp along axis 0.

    MATLAB original (log_sum_exp.m):
        xm = max(x);
        x  = x - repmat(xm, size(x,1), 1);
        y  = xm + log(sum(exp(x)));

    Args:
        x: 2-D array of shape (rows, cols). Each column is treated independently.

    Returns:
        1-D array of shape (cols,) with log(sum(exp(x), axis=0)).
    """
    xm = np.max(x, axis=0)                  # shape (cols,)
    x_shifted = x - xm[np.newaxis, :]       # broadcast subtract max
    y = xm + np.log(np.sum(np.exp(x_shifted), axis=0))
    return y


def ber_to_per(ber: float, packet_size: int) -> float:
    """
    Convert BER to PER.

    MATLAB original (bertoper.m):
        per = 1 - (1 - ber).^PacketSize;
    """
    return 1.0 - (1.0 - ber) ** packet_size
