"""
Polar code initialization.

Translated from initPC.m (Athirah Mohd Ramly, UKM).
"""

import os
import numpy as np


def init_pc(N: int, K: int, n: int, construction_method: int,
            design_snr_db: float, sigma: float, crc_size: int,
            data_dir: str):
    """
    Initialize polar code parameters.

    MATLAB original: initPC.m

    Args:
        N: Block length (e.g. 256).
        K: Information bits (e.g. 128).
        n: log2(N).
        construction_method: 0=BhattaBound, 1=MC, 2=GA.
        design_snr_db: Design SNR in dB (used by BA and MC).
        sigma: Sigma value (used by GA).
        crc_size: CRC size (0, 8, 16, or 32).
        data_dir: Directory containing the polar code .txt files.

    Returns:
        fz_lookup (np.ndarray): 1-D int array of length N.
            -1 at information bit positions, 0 at frozen bit positions.
        bit_reversed_indices (np.ndarray): 1-D int array of length N,
            0-based bit-reversal permutation.
        f_kron_n (np.ndarray): N×N generator matrix (Kronecker product of F).
    """
    # Build generator matrix F^⊗n
    F = np.array([[1, 0], [1, 1]], dtype=np.int32)
    BB = np.array([[1]], dtype=np.int32)
    for _ in range(n):
        BB = np.kron(BB, F)
    f_kron_n = BB  # shape (N, N)

    # Bit-reversal permutation (0-based)
    bit_reversed_indices = np.zeros(N, dtype=np.int32)
    for index in range(N):
        # MATLAB: bin2dec(wrev(dec2bin(index-1, n))) — 1-based → convert
        bits = format(index, f'0{n}b')      # n-bit binary string, 0-based
        reversed_bits = bits[::-1]
        bit_reversed_indices[index] = int(reversed_bits, 2)

    # Select construction file
    if construction_method == 0:
        fname = (f"PolarCode_block_length_{N}_designSNR_{design_snr_db:.2f}dB"
                 f"_method_BhattaBound.txt")
    elif construction_method == 1:
        fname = (f"PolarCode_block_length_{N}_designSNR_{design_snr_db:.2f}dB"
                 f"_method_MC.txt")
    elif construction_method == 2:
        fname = (f"PolarCode_block_length_{N}_sigma_{sigma:.2f}"
                 f"_method_GA.txt")
    else:
        raise ValueError("construction_method must be 0, 1, or 2")

    # MATLAB original uses constructedCode\ subdirectory
    code_file = os.path.join(data_dir, "constructedCode", fname)
    indices = np.loadtxt(code_file, dtype=np.int32)  # 1-based indices

    # Build FZ lookup: -1 = information bit, 0 = frozen bit
    fz_lookup = np.zeros(N, dtype=np.int32)
    num_info = K + crc_size if crc_size > 0 else K
    # MATLAB indices are 1-based; convert to 0-based
    fz_lookup[indices[:num_info] - 1] = -1

    return fz_lookup, bit_reversed_indices, f_kron_n
