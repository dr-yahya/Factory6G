"""
Polar encoder with optional CRC.

Translated from pencode.m (Athirah Mohd Ramly, UKM).
"""

import numpy as np


def _crc8(u: np.ndarray) -> np.ndarray:
    """CRC-8 polynomial: x^8 + x^2 + x + 1 → [1 0 0 0 0 0 1 1 1]"""
    crc_gen = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    L = len(u)
    left_shift = np.zeros(9, dtype=np.int32)
    left_shift[0] = 1
    a = np.convolve(u, left_shift).astype(np.int32)
    for i in range(L):
        if a[i] == 1:
            a[i:i + 9] ^= crc_gen
    return a[L:L + 8]


def _crc16(u: np.ndarray) -> np.ndarray:
    """CRC-16: x^16 + x^15 + x^2 + 1 → [1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1]"""
    crc_gen = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int32)
    L = len(u)
    left_shift = np.zeros(17, dtype=np.int32)
    left_shift[0] = 1
    a = np.convolve(u, left_shift).astype(np.int32)
    for i in range(L):
        if a[i] == 1:
            a[i:i + 17] ^= crc_gen
    return a[L:L + 16]


def _crc32(u: np.ndarray) -> np.ndarray:
    """CRC-32."""
    crc_gen = np.array(
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=np.int32)
    L = len(u)
    left_shift = np.zeros(33, dtype=np.int32)
    left_shift[0] = 1
    a = np.convolve(u, left_shift).astype(np.int32)
    for i in range(L):
        if a[i] == 1:
            a[i:i + 33] ^= crc_gen
    return a[L:L + 32]


def pencode(u: np.ndarray, fz_lookup: np.ndarray, crc_size: int,
            bit_reversed_indices: np.ndarray, f_kron_n: np.ndarray) -> np.ndarray:
    """
    Polar encoder with optional CRC.

    MATLAB original: pencode.m

    Args:
        u: Information bits, 1-D int array of length K.
        fz_lookup: FZ lookup from init_pc, length N. -1 = info position.
        crc_size: 0, 8, 16, or 32.
        bit_reversed_indices: 0-based bit-reversal permutation, length N.
        f_kron_n: N×N generator matrix.

    Returns:
        y: Encoded codeword, 1-D int array of length N.
    """
    # Compute CRC and append to info bits
    if crc_size == 0:
        crc_bits = np.array([], dtype=np.int32)
    elif crc_size == 8:
        crc_bits = _crc8(u)
    elif crc_size == 16:
        crc_bits = _crc16(u)
    elif crc_size == 32:
        crc_bits = _crc32(u)
    else:
        raise ValueError(f"Unsupported crc_size: {crc_size}")

    u_full = np.concatenate([u, crc_bits]).astype(np.int32)

    # Place info bits into frozen-bit vector
    # MATLAB: x(x == -1) = u;  (x is fz_lookup copy, -1 positions get info bits)
    x = fz_lookup.copy().astype(np.int32)
    x[x == -1] = u_full  # replace -1 positions with message bits in order

    # Bit-reversal permutation
    x = x[bit_reversed_indices]

    # Encode: y = mod(x * F^⊗n, 2)
    y = np.mod(x @ f_kron_n, 2).astype(np.int32)
    return y
