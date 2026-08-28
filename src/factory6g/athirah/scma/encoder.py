"""
SCMA encoder.

Translated from scmaenc.m
(Vyacheslav P. Klimentyev and Alexander B. Sergienko, SPbETU, 2015).
"""

import numpy as np


def scmaenc(x: np.ndarray, CB: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    SCMA encoder: maps user symbols to resource grid via codebook.

    MATLAB original: scmaenc.m

    Args:
        x: Input symbols, int array of shape (V, N).
           Values in range [0, M-1].
        CB: SCMA codebook, complex array of shape (K, M, V).
            K = number of resources, M = codewords per codebook, V = users.
        h: Channel coefficients, complex array of shape (K, V, N).

    Returns:
        y: SCMA encoded signal after fading, complex array of shape (K, N).
    """
    K = CB.shape[0]
    V = CB.shape[2]
    N = x.shape[1]

    y = np.zeros((K, N), dtype=np.complex128)

    for n in range(N):
        for k in range(V):
            # MATLAB: y(:,n) += CB(:, x(k,n)+1, k) .* h(:,k,n)
            # x(k,n) is 0-based here (converted from MATLAB's 1-based x(k,n)+1)
            y[:, n] += CB[:, x[k, n], k] * h[:, k, n]

    return y
