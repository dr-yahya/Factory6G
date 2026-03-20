"""
Joint Iterative Detection and Decoding (JIDD) for polar-coded SCMA-OFDM.

Translated from JIDD.m (Athirah Mohd Ramly, UKM).
"""

import numpy as np
from src.athirah.utils.math import log_sum_exp
from src.athirah.polar.scan import polar_scan_decode_alpha


def jidd(y: np.ndarray, polar_N: int, polar_K: int, fz_lookup: np.ndarray,
         K: int, V: int, M: int, N: int, CB: np.ndarray, N0: float,
         h: np.ndarray, iter_num: int, is_interleaver: int,
         interleaver: np.ndarray, alpha: float) -> np.ndarray:
    """
    Joint iterative detection and decoding (message passing).

    MATLAB original: JIDD.m

    Args:
        y: Received signal after equalization, complex array (K, N).
           K = resources, N = SCMA symbols per user.
        polar_N: Polar code block length (e.g. 256).
        polar_K: Polar code info bits (e.g. 128).
        fz_lookup: FZ lookup from init_pc, length polar_N.
        K: Number of SCMA resources (e.g. 4).
        V: Number of users (e.g. 6).
        M: Codebook size / SCMA order (e.g. 4 for QPSK).
        N: Number of SCMA symbols per user (= polar_N / log2(M)).
        CB: SCMA codebook, complex array (K, M, V).
        N0: Noise variance (linear).
        h: Channel coefficients, complex array (K, V, N).
        iter_num: Number of outer JIDD iterations.
        is_interleaver: 1 if interleaving is used, 0 otherwise.
        interleaver: Interleaver permutation indices, int array (V, polar_N).
                     0-based.
        alpha: Alpha scaling factor for SCAN decoder.

    Returns:
        mhat_llr: Soft LLRs for decoded info bits, float array (V, polar_K).
    """
    log2M = int(np.log2(M))

    # ─── Factor graph: F[k, v] = 1 if user v uses resource k ───────────────
    # MATLAB: F = zeros(K,V); IND = find(CB(:,:,k)); [I,~]=ind2sub(s,IND); F(unique(I),k)=1
    # CB shape: (K, M, V) — for user v (0-based): CB[:, :, v]
    F_graph = np.zeros((K, V), dtype=np.int32)
    for v in range(V):
        codebook_v = CB[:, :, v]                     # shape (K, M)
        rows_with_data = np.where(codebook_v.any(axis=1))[0]
        F_graph[rows_with_data, v] = 1

    # ─── Pre-compute log-likelihood tensor f[m1, m2, m3, k, jj] ────────────
    # f(m1,m2,m3,k,jj) = -(1/N0) * |y[k,jj] - sum_over_active_users(CB[k,m_u,u]*h[k,u,jj])|^2
    # Each resource k is connected to exactly 3 users (SCMA with overloading 6/4).
    # We store active user indices per resource.
    resource_users = {}  # k → list of user indices connected to resource k
    for k in range(K):
        resource_users[k] = list(np.where(F_graph[k, :] == 1)[0])

    # f tensor: (M, M, M, K, N)
    f = np.zeros((M, M, M, K, N), dtype=np.float64)
    for jj in range(N):
        for k in range(K):
            ind = resource_users[k]  # 3 users connected to resource k
            for m1 in range(M):
                for m2 in range(M):
                    for m3 in range(M):
                        signal = (CB[k, m1, ind[0]] * h[k, ind[0], jj]
                                  + CB[k, m2, ind[1]] * h[k, ind[1], jj]
                                  + CB[k, m3, ind[2]] * h[k, ind[2], jj])
                        f[m1, m2, m3, k, jj] = -(1.0 / N0) * abs(y[k, jj] - signal) ** 2

    # ─── Message initialisation ─────────────────────────────────────────────
    LLR = np.zeros((V, log2M * N), dtype=np.float64)

    # Ap[1, v, m, jj]: prior probability — uniform 1/M
    Ap = (1.0 / M) * np.ones((1, V, M, N), dtype=np.float64)

    # Ivg: variable-to-factor log messages, shape (K, V, M, N) — log uniform
    Ivg = np.log((1.0 / M) * np.ones((K, V, M, N), dtype=np.float64))

    # Igv: factor-to-variable log messages, shape (K, V, M, N)
    Igv = np.zeros((K, V, M, N), dtype=np.float64)

    c_llr = np.zeros((polar_N, V), dtype=np.float64)
    u_llr = np.zeros((polar_N, V), dtype=np.float64)
    mhat_llr = np.zeros((V, polar_K), dtype=np.float64)

    # ─── Outer iteration loop ────────────────────────────────────────────────
    for _ in range(iter_num):

        # ── Factor-to-variable messages (Igv) ──────────────────────────────
        for k in range(K):
            ind = resource_users[k]   # [u0, u1, u2]

            for m1 in range(M):
                sIgv = np.zeros((M * M, N), dtype=np.float64)
                for m2 in range(M):
                    for m3 in range(M):
                        sIgv[(m2) * M + m3, :] = (
                            f[m1, m2, m3, k, :]
                            + Ivg[k, ind[1], m2, :]
                            + Ivg[k, ind[2], m3, :]
                        )
                Igv[k, ind[0], m1, :] = log_sum_exp(sIgv)

            for m2 in range(M):
                sIgv = np.zeros((M * M, N), dtype=np.float64)
                for m1 in range(M):
                    for m3 in range(M):
                        sIgv[(m1) * M + m3, :] = (
                            f[m1, m2, m3, k, :]
                            + Ivg[k, ind[0], m1, :]
                            + Ivg[k, ind[2], m3, :]
                        )
                Igv[k, ind[1], m2, :] = log_sum_exp(sIgv)

            for m3 in range(M):
                sIgv = np.zeros((M * M, N), dtype=np.float64)
                for m1 in range(M):
                    for m2 in range(M):
                        sIgv[(m1) * M + m2, :] = (
                            f[m1, m2, m3, k, :]
                            + Ivg[k, ind[0], m1, :]
                            + Ivg[k, ind[1], m2, :]
                        )
                Igv[k, ind[2], m3, :] = log_sum_exp(sIgv)

        # ── Combine factor messages to compute Q[m, v, jj] ─────────────────
        Q = np.zeros((M, V, N), dtype=np.float64)
        for v in range(V):
            ind = np.where(F_graph[:, v] == 1)[0]  # resources connected to user v
            for m in range(M):
                Q[m, v, :] = Igv[ind[0], v, m, :] + Igv[ind[1], v, m, :]

        # ── Compute LLRs from Q (QPSK: log2(M)=2 bits per symbol) ──────────
        for jj in range(N):
            for v in range(V):
                # Bit 0 (MSB): symbols 0,1 vs 2,3
                LLR[v, 2 * jj] = np.log(
                    (np.exp(Q[0, v, jj]) + np.exp(Q[1, v, jj]))
                    / (np.exp(Q[2, v, jj]) + np.exp(Q[3, v, jj]) + 1e-300)
                )
                # Bit 1 (LSB): symbols 0,2 vs 1,3
                LLR[v, 2 * jj + 1] = np.log(
                    (np.exp(Q[0, v, jj]) + np.exp(Q[2, v, jj]))
                    / (np.exp(Q[1, v, jj]) + np.exp(Q[3, v, jj]) + 1e-300)
                )

        # ── De-interleave and polar decode ──────────────────────────────────
        if is_interleaver != 0:
            LLR_deint = np.zeros((V, polar_N), dtype=np.float64)
            for ii in range(V):
                LLR_deint[ii, interleaver[ii, :]] = LLR[ii, :]
        else:
            LLR_deint = LLR

        for user in range(V):
            u_llr[:, user], c_llr[:, user] = polar_scan_decode_alpha(
                LLR_deint[user, :], 1, alpha, fz_lookup, polar_N
            )

        # ── Re-interleave codeword LLRs for feedback ───────────────────────
        c_llr_T = c_llr.T  # shape (V, polar_N)
        if is_interleaver != 0:
            c_llr_int = np.zeros_like(c_llr_T)
            for ii in range(V):
                c_llr_int[ii, :] = c_llr_T[ii, interleaver[ii, :]]
        else:
            c_llr_int = c_llr_T

        # ── Convert codeword LLRs to symbol probabilities (polar → SCMA) ───
        # MATLAB: polar2scma(1,v,1,:) = P(b0=0)*P(b1=0) = σ(c0)*σ(c1)  (symbol 00)
        #         polar2scma(1,v,2,:) = P(b0=0)*P(b1=1) = σ(c0)*(1-σ(c1)) (symbol 01)
        #         polar2scma(1,v,3,:) = P(b0=1)*P(b1=0)  (symbol 10)
        #         polar2scma(1,v,4,:) = P(b0=1)*P(b1=1)  (symbol 11)
        polar2scma = np.zeros((1, V, M, N), dtype=np.float64)
        for v in range(V):
            llr_msb = c_llr_int[v, 0::2]   # bits 0,2,4,...  shape (N,)
            llr_lsb = c_llr_int[v, 1::2]   # bits 1,3,5,...  shape (N,)

            sig_msb  = np.exp(llr_msb) / (np.exp(llr_msb) + 1.0)
            sig_lsb  = np.exp(llr_lsb) / (np.exp(llr_lsb) + 1.0)
            nsg_msb  = 1.0 / (np.exp(llr_msb) + 1.0)
            nsg_lsb  = 1.0 / (np.exp(llr_lsb) + 1.0)

            polar2scma[0, v, 0, :] = sig_msb * sig_lsb   # 00
            polar2scma[0, v, 1, :] = sig_msb * nsg_lsb   # 01
            polar2scma[0, v, 2, :] = nsg_msb * sig_lsb   # 10
            polar2scma[0, v, 3, :] = nsg_msb * nsg_lsb   # 11

        Ap = polar2scma

        # ── Update variable-to-factor messages ──────────────────────────────
        # MATLAB: Ivg = log(repmat(Ap, K, 1))
        Ivg = np.log(np.tile(Ap, (K, 1, 1, 1)) + 1e-300)  # shape (K,V,M,N)

    # ─── Extract decoded info bit LLRs ──────────────────────────────────────
    # MATLAB: mhat_llr(user,:) = u_llr(fz_lookup==-1, user)'
    for user in range(V):
        mhat_llr[user, :] = u_llr[fz_lookup == -1, user]

    return mhat_llr
