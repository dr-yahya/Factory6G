"""Analytic complexity of the channel estimators.

Wall-clock runtime is the wrong way to compare these. The measurement is
dominated by eager TensorFlow dispatch overhead, it is unfair to the NumPy-based
estimators, and it varies with the host -- so a claim like "the adaptive
estimator reaches near-LMMSE accuracy at lower complexity" cannot rest on it.

These are complex-multiplication counts per channel estimate, derived from the
algorithms rather than measured, which is what a complexity claim in a paper
needs.
"""

from __future__ import annotations

import math


def ls_complexity(fft_size: int, num_pilot_symbols: int) -> float:
    """Least squares: one division per pilot resource element, plus interpolation."""
    return float(fft_size * num_pilot_symbols)


def dft_complexity(fft_size: int, num_pilot_symbols: int) -> float:
    """DFT truncation: LS, then a forward and inverse FFT per pilot symbol.

    An N-point FFT costs (N/2)log2(N) complex multiplications.
    """
    fft_cost = 0.5 * fft_size * math.log2(max(fft_size, 2))
    return ls_complexity(fft_size, num_pilot_symbols) + 2.0 * fft_cost * num_pilot_symbols


def lmmse_complexity(fft_size: int, num_pilot_symbols: int, *, use_eigen: bool = True) -> float:
    """LMMSE smoothing.

    With the precomputed eigendecomposition this project uses, each estimate
    costs two N x N matrix-vector products (project, shrink, back-project), i.e.
    2*N^2 -- versus N^3 for a direct inverse recomputed per noise level.
    """
    per_symbol = 2.0 * fft_size**2 if use_eigen else float(fft_size**3)
    return ls_complexity(fft_size, num_pilot_symbols) + per_symbol * num_pilot_symbols


def adaptive_complexity(
    fft_size: int,
    num_pilot_symbols: int,
    *,
    lmmse_fraction: float,
    selection_mode: str = "per_user",
) -> float:
    """Adaptive hybrid complexity.

    ``lmmse_fraction`` is the share of estimates that actually need the LMMSE
    branch. In ``scalar`` mode the branch is chosen per batch, so only the
    selected branch is evaluated and the cost interpolates between DFT and
    LMMSE. In ``per_user`` mode both branches are evaluated and blended, so the
    cost is their sum plus the selection statistic -- the price paid for a
    decision that is per user and graph-traceable.
    """
    fraction = min(max(float(lmmse_fraction), 0.0), 1.0)
    dft = dft_complexity(fft_size, num_pilot_symbols)
    lmmse = lmmse_complexity(fft_size, num_pilot_symbols)
    if selection_mode == "scalar":
        return fraction * lmmse + (1.0 - fraction) * dft
    # Selection statistic: one inverse FFT per pilot symbol for the delay profile.
    selection = 0.5 * fft_size * math.log2(max(fft_size, 2)) * num_pilot_symbols
    return dft + lmmse + selection


def estimator_complexity(
    name: str,
    fft_size: int,
    num_pilot_symbols: int,
    **kwargs,
) -> float:
    """Complex multiplications per channel estimate for a named estimator."""
    key = name.lower()
    if key in {"ls", "ls_nn", "ls_lin"}:
        return ls_complexity(fft_size, num_pilot_symbols)
    if key in {"dft", "dft-based"}:
        return dft_complexity(fft_size, num_pilot_symbols)
    if key in {"lmmse", "approx_lmmse"}:
        return lmmse_complexity(fft_size, num_pilot_symbols, **kwargs)
    if key in {"adaptive", "adaptive_hybrid"}:
        kwargs.setdefault("lmmse_fraction", 0.5)
        return adaptive_complexity(fft_size, num_pilot_symbols, **kwargs)
    raise ValueError(f"No complexity model for estimator '{name}'.")
