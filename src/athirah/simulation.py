"""
Main JIDD-SCMA simulation loop.

Translated from main_JIDD.m (Athirah Mohd Ramly, UKM).

Entry point: run_jidd_scma(cfg) → dict of per-Eb/N0 results.
"""

import os
import time
import logging

import numpy as np
import scipy.io

from src.athirah.polar.init_pc import init_pc
from src.athirah.polar.pencode import pencode
from src.athirah.scma.encoder import scmaenc
from src.athirah.ofdm.pilot import add_pilot, remove_pilot
from src.athirah.ofdm.cp import add_cp, remove_cp
from src.athirah.estimation.mmse_ce import mmse_ce
from src.athirah.estimation.ls_ce import ls_ce
from src.athirah.jidd import jidd

logger = logging.getLogger(__name__)


def _awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add AWGN at given SNR (dB), measured from signal power.

    Equivalent to MATLAB's awgn(signal, snr_db, 'measured').
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise


def _bi2de_left_msb(bits: np.ndarray) -> np.ndarray:
    """
    Convert rows of bits to decimal integers, MSB first.

    Equivalent to MATLAB's bi2de(bits, 'left-msb').
    bits: shape (rows, cols)
    Returns: shape (rows,)
    """
    ncols = bits.shape[1]
    weights = 1 << np.arange(ncols - 1, -1, -1)
    return bits @ weights


def run_jidd_scma(cfg: dict, data_dir: str, ce_method: str | None = None) -> dict:
    """
    Run the JIDD SCMA-OFDM BER simulation.

    Args:
        cfg: jidd_scma section from config.json.
        data_dir: Path to directory containing polar code .txt files and
                  the SCMA codebook .mat file (i.e. 'dr_athirah_simulation/PHY layer').

    Returns:
        Dictionary keyed by Eb/N0 value (float) with fields:
            ber, bit_errors, total_bits, runtime_sec
    """
    # ── Simulation parameters (from config) ──────────────────────────────────
    polar_N            = cfg.get("polar_N", 256)
    polar_K            = cfg.get("polar_K", 128)
    construction_method = cfg.get("construction_method", 0)
    design_snr_db      = cfg.get("design_snr_db", 0.0)
    sigma              = cfg.get("sigma", 0.9)
    crc_size           = cfg.get("crc_size", 0)
    alpha              = cfg.get("alpha", 0.6)
    iter_num           = cfg.get("iter_num", 5)
    is_interleaver     = cfg.get("is_interleaver", True)
    if ce_method is None:
        ce_method      = cfg.get("ce_method", "mmse")
    ebno_start         = cfg.get("ebno_db_range", [0, 10])[0]
    ebno_stop          = cfg.get("ebno_db_range", [0, 10])[1]
    ebno_step          = cfg.get("ebno_db_step", 1)
    max_bits           = cfg.get("max_bits", int(1e7))
    min_bits           = cfg.get("min_bits", 50000)
    min_errors         = cfg.get("min_errors", 50)

    polar_n = int(np.log2(polar_N))

    EbN0 = np.arange(ebno_start, ebno_stop + 1e-9, ebno_step)

    # ── Polar code initialisation ─────────────────────────────────────────────
    fz_lookup, bit_reversed_indices, f_kron_n = init_pc(
        polar_N, polar_K, polar_n,
        construction_method, design_snr_db, sigma, crc_size,
        data_dir
    )

    # ── Load SCMA codebook ────────────────────────────────────────────────────
    mat_path = os.path.join(data_dir, "codebook_6users_4chips_qpsk.mat")
    mat_data = scipy.io.loadmat(mat_path)
    CB = mat_data["CB"].astype(np.complex128)  # shape (K, M, V)

    K_res = CB.shape[0]   # 4 resources
    M     = CB.shape[1]   # 4 codewords (QPSK)
    V     = CB.shape[2]   # 6 users

    log2M = int(np.log2(M))

    # Number of SCMA symbols per user per frame
    N_sym = polar_N // log2M   # 256 // 2 = 128

    # Effective SNR offset: EbN0 → SNR
    # MATLAB: SNR = EbN0 + 10*log10(polar_K/polar_N * log2(M) * V / K_res)
    offset_db = 10.0 * np.log10(
        (polar_K / polar_N) * log2M * V / K_res
    )
    SNR_db = EbN0 + offset_db
    N0_vec = 1.0 / 10.0 ** (SNR_db / 10.0)

    # ── Per-Eb/N0 results storage ─────────────────────────────────────────────
    results = {}

    for idx_ebn0, ebn0 in enumerate(EbN0):
        n_err  = 0
        n_bits = 0
        t_start = time.time()

        logger.info(f"[jidd_scma] Eb/N0 = {ebn0:.1f} dB")

        while (n_err < min_errors or n_bits < min_bits) and n_bits < max_bits:

            # ── Generate info bits ────────────────────────────────────────────
            infobits = np.random.randint(0, 2, (V, polar_K), dtype=np.int32)

            # ── Polar encode each user ────────────────────────────────────────
            c = np.zeros((V, polar_N), dtype=np.int32)
            for user in range(V):
                c[user, :] = pencode(infobits[user, :], fz_lookup,
                                     crc_size, bit_reversed_indices, f_kron_n)

            # ── Interleaver ───────────────────────────────────────────────────
            if is_interleaver:
                interleaver = np.zeros((V, polar_N), dtype=np.int32)
                interleaved_bits = np.zeros_like(c)
                for ii in range(V):
                    interleaver[ii, :] = np.random.permutation(polar_N)
                    interleaved_bits[ii, :] = c[ii, interleaver[ii, :]]
            else:
                interleaved_bits = c
                interleaver = np.tile(np.arange(polar_N, dtype=np.int32), (V, 1))

            # ── Reshape bits → SCMA symbols ───────────────────────────────────
            # MATLAB:
            #   temp1 = reshape(interleavered_bits', polar_N*V, 1)  → column-major
            #   temp2 = reshape(temp1, log2M, N_sym*V)
            #   x_temp = bi2de(temp2', 'left-msb')    → shape (N_sym*V,)
            #   x = reshape(x_temp, N_sym, V)'         → shape (V, N_sym)
            #
            # MATLAB reshape(A', N, 1) reads A' column-by-column = A row-by-row.
            # For interleaved_bits (V, polar_N): this gives [user0_bits, user1_bits, ...]
            temp1 = interleaved_bits.flatten(order='C')           # shape (polar_N*V,)
            temp2 = temp1.reshape(log2M, N_sym * V, order='F')    # shape (log2M, N_sym*V)
            x_temp = _bi2de_left_msb(temp2.T)                     # shape (N_sym*V,)
            x = x_temp.reshape(N_sym, V, order='F').T             # shape (V, N_sym)

            # ── Channel: Rayleigh flat fading (Paper 1, Table 2) ─────────────
            # h[k, u, n] ~ CN(0,1) — i.i.d. per resource, user, OFDM symbol
            # Translated from generate_channel_matrix_multiuser.m (UL Rayleigh W/ Diversity)
            h = (np.random.randn(K_res, V, N_sym) + 1j * np.random.randn(K_res, V, N_sym)) / np.sqrt(2)

            # ── SCMA encode ───────────────────────────────────────────────────
            s = scmaenc(x, CB, h)  # shape (K_res, N_sym)

            # ── Add pilots ────────────────────────────────────────────────────
            s_pilot, pilot_loc, Xp, data_loc = add_pilot(s)  # shape (640,)

            # ── IFFT ──────────────────────────────────────────────────────────
            s_ifft = np.fft.ifft(s_pilot)

            # ── Add CP ────────────────────────────────────────────────────────
            s_cp = add_cp(s_ifft)  # shape (800,)

            # ── Parallel → serial (already 1-D) ──────────────────────────────

            # ── AWGN channel ──────────────────────────────────────────────────
            y_noisy = _awgn(s_cp, SNR_db[idx_ebn0])

            # ── Remove CP ────────────────────────────────────────────────────
            y_no_cp = remove_cp(y_noisy)  # shape (640,)

            # ── FFT ───────────────────────────────────────────────────────────
            y_freq = np.fft.fft(y_no_cp)  # shape (640,)

            # ── Channel estimation (mmse / ls_linear / ls_spline) ────────────
            if ce_method == "ls_linear":
                H_est = ls_ce(y_freq, Xp, pilot_loc, Nfft=640, Nps=4, int_opt="linear")
            elif ce_method == "ls_spline":
                H_est = ls_ce(y_freq, Xp, pilot_loc, Nfft=640, Nps=4, int_opt="spline")
            else:  # "mmse" (default)
                H_est = mmse_ce(y_freq, Xp, pilot_loc, Nfft=640, Nps=4,
                                snr_db=SNR_db[idx_ebn0])

            # ── Equalization ──────────────────────────────────────────────────
            y_eq = y_freq / (H_est + 1e-15)

            # ── Remove pilots → data subcarriers ─────────────────────────────
            y1 = remove_pilot(y_eq, data_loc)  # shape (4, 128) = (K_res, N_sym)

            # ── JIDD: joint detection + polar decoding ────────────────────────
            mhat_llr = jidd(
                y1, polar_N, polar_K, fz_lookup,
                K_res, V, M, N_sym, CB, N0_vec[idx_ebn0],
                h, iter_num, int(is_interleaver), interleaver, alpha
            )

            # ── Bit error count ───────────────────────────────────────────────
            # MATLAB:
            #   llr = reshape(mhat_llr', 1, V*polar_K)
            #   m_reshape = reshape(infobits', 1, polar_K*V)
            #   m_hat = llr < 0
            #   err = sum(m_hat != m_reshape)
            llr_flat   = mhat_llr.T.flatten(order='C')
            m_flat     = infobits.T.flatten(order='C')
            m_hat      = (llr_flat < 0).astype(np.int32)
            err        = int(np.sum(m_hat != m_flat))

            n_err  += err
            n_bits += len(m_flat)

        ber = n_err / n_bits if n_bits > 0 else 1.0
        runtime = time.time() - t_start

        results[float(ebn0)] = {
            "ber": ber,
            "bit_errors": n_err,
            "total_bits": n_bits,
            "runtime_sec": runtime,
        }

        logger.info(
            f"[jidd_scma] Eb/N0={ebn0:.1f} dB | bits={n_bits} | "
            f"errors={n_err} | BER={ber:.6f} | t={runtime:.1f}s"
        )

    return results
