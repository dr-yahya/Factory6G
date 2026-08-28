"""
SCAN polar decoder with alpha-scaling.

Translated from:
  polar_SCAN_decode_alpha.m
  updateLLRMap.m
  updateBitMap.m
  fFunction.m
(Athirah Mohd Ramly, UKM)
"""

import numpy as np


def f_function(a: float, b: float) -> float:
    """
    Min-sum f-function used in SCAN decoding.

    MATLAB original (fFunction.m):
        c = sign(a)*sign(b)*min(abs(a), abs(b));
    """
    return np.sign(a) * np.sign(b) * min(abs(a), abs(b))


def update_llr_map(lam: int, phi: int, n: int,
                   L: np.ndarray, B: np.ndarray) -> None:
    """
    Recursive LLR-map update (left messages).

    MATLAB original: updateLLRMap.m
    Note: L and B are modified in-place (shape N×(n+1), 0-based indices).

    MATLAB uses 1-based: L(idx+1, col+1) → Python: L[idx, col]
    MATLAB column mapping:
        n+1-lambda  → n - lam   (0-based)
        n+2-lambda  → n+1-lam   (0-based)
    """
    if lam == 0:
        return

    psi = phi // 2

    if phi % 2 == 0:
        update_llr_map(lam - 1, psi, n, L, B)

    stride = 2 ** lam
    half_stride = 2 ** (lam - 1)
    num_omega = 2 ** (n - lam)

    col_cur  = n - lam        # n+1-lambda - 1 (0-based)
    col_next = n + 1 - lam    # n+2-lambda - 1 (0-based)

    for omega in range(num_omega):
        if phi % 2 == 0:
            idx_L  = phi  + omega * stride          # MATLAB: phi+omega*2^lam+1 → 0-based
            idx_La = psi  + 2 * omega * half_stride  # upper child
            idx_Lb = psi  + (2 * omega + 1) * half_stride  # lower child
            idx_B  = phi + 1 + omega * stride        # MATLAB: phi+1+omega*2^lam+1 → 0-based
            L[idx_L, col_cur] = f_function(
                L[idx_La, col_next],
                L[idx_Lb, col_next] + B[idx_B, col_cur]
            )
        else:
            idx_L  = phi  + omega * stride
            idx_La = psi  + 2 * omega * half_stride
            idx_Lb = psi  + (2 * omega + 1) * half_stride
            idx_B  = phi - 1 + omega * stride
            L[idx_L, col_cur] = (
                L[idx_Lb, col_next]
                + f_function(L[idx_La, col_next], B[idx_B, col_cur])
            )


def update_bit_map(lam: int, phi: int, n: int,
                   L: np.ndarray, B: np.ndarray) -> None:
    """
    Recursive bit-map update (right messages).

    MATLAB original: updateBitMap.m
    Modifies L and B in-place.
    """
    psi = phi // 2

    if phi % 2 != 0:
        stride     = 2 ** lam
        half_stride = 2 ** (lam - 1)
        num_omega  = 2 ** (n - lam)

        col_cur  = n - lam      # n+1-lambda - 1
        col_next = n + 1 - lam  # n+2-lambda - 1

        for omega in range(num_omega):
            idx_B_even = phi - 1 + omega * stride   # MATLAB: phi-1+omega*2^lam+1
            idx_B_odd  = phi     + omega * stride   # MATLAB: phi+omega*2^lam+1
            idx_B_up   = psi + 2 * omega * half_stride
            idx_B_dn   = psi + (2 * omega + 1) * half_stride
            idx_L_dn   = psi + (2 * omega + 1) * half_stride
            idx_L_up   = psi + 2 * omega * half_stride

            B[idx_B_up, col_next] = f_function(
                B[idx_B_even, col_cur],
                B[idx_B_odd,  col_cur] + L[idx_L_dn, col_next]
            )
            B[idx_B_dn, col_next] = (
                B[idx_B_odd, col_cur]
                + f_function(B[idx_B_even, col_cur], L[idx_L_up, col_next])
            )

        if psi % 2 != 0:
            update_bit_map(lam - 1, psi, n, L, B)


def polar_scan_decode_alpha(y_llr: np.ndarray, iter_num: int,
                             alpha: float, fz_lookup: np.ndarray,
                             N: int):
    """
    SCAN polar decoder with alpha-scaling.

    MATLAB original: polar_SCAN_decode_alpha.m

    Args:
        y_llr: Received LLR values, 1-D array of length N.
        iter_num: Number of SCAN iterations.
        alpha: Alpha scaling factor for soft output.
        fz_lookup: FZ lookup (-1 = info, 0 = frozen), length N.
        N: Block length.

    Returns:
        u_llr: Soft LLRs for decoded info bits, 1-D array of length N.
        c_llr: Soft LLRs for codeword bits, 1-D array of length N.
    """
    n = int(np.log2(N))
    plus_infinity = 1000.0

    # L[i, j]: left message at bit i, stage j  (0-based, shape N×(n+1))
    # B[i, j]: right message at bit i, stage j
    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)

    # Initial conditions
    L[:, n] = y_llr                           # MATLAB: L(:, n+1) = y_llr'
    B[fz_lookup == 0, 0] = plus_infinity      # MATLAB: B(FZlookup==0, 1) = +inf

    # SCAN iterations
    for _ in range(iter_num):
        for phi in range(N):
            update_llr_map(n, phi, n, L, B)
            if phi % 2 != 0:
                update_bit_map(n, phi, n, L, B)

    mean_B = np.mean(np.abs(B[:, n]))   # MATLAB: mean(abs(B(:, n+1)))
    mean_L = np.mean(np.abs(L[:, n]))   # MATLAB: mean(abs(L(:, n+1)))

    u_llr = L[:, 0] + B[:, 0]           # MATLAB: L(:,1) + B(:,1)

    # Alpha-scaled soft codeword output
    if mean_L > 0:
        c_llr = B[:, n] + alpha * (mean_B / mean_L) * L[:, n]
    else:
        c_llr = B[:, n]

    return u_llr, c_llr
