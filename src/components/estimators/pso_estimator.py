"""PSO-based channel estimator that fits a low-order polynomial across frequency.

This estimator runs a lightweight Particle Swarm Optimization (PSO) per OFDM
symbol to fit a complex-valued polynomial in subcarrier index that best
approximates an initial LS estimate. The idea is to denoise and regularize
the frequency response while being robust to non-convexities in the objective.

Notes:
- Uses LSChannelEstimator to obtain an initial estimate and error variance.
- Per-OFDM-symbol, per (rx, tx, ut, stream), we fit degree-d complex polynomial:
    H_hat[k] ≈ Σ_{m=0..d} c_m k^m,  k ∈ {0, ..., fft_size-1}
- PSO minimizes squared error against the initial LS estimate across all
  subcarriers for the symbol (acts like a smooth regression).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from typing import Tuple

from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid

from ..config import SystemConfig


def _poly_eval(k: np.ndarray, coeffs_real_imag: np.ndarray) -> np.ndarray:
    """Evaluate complex polynomial with real-imag coefficients on k.
    coeffs_real_imag has shape [(degree+1)*2] arranged as:
        [Re(c0), Im(c0), Re(c1), Im(c1), ..., Re(cd), Im(cd)]
    Returns complex array of shape [len(k)].
    """
    degree = coeffs_real_imag.shape[0] // 2 - 1
    real = coeffs_real_imag[0::2]
    imag = coeffs_real_imag[1::2]
    coeffs = real + 1j * imag  # shape [degree+1]
    # Horner's method
    y = np.zeros_like(k, dtype=np.complex64)
    for c in coeffs[::-1]:
        y = y * k + c
    return y


def _pso_optimize(
    target: np.ndarray,
    k: np.ndarray,
    degree: int,
    swarm_size: int,
    iters: int,
    w_start: float,
    w_end: float,
    c1: float,
    c2: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run PSO to fit complex polynomial to target (complex) vs k (float).
    Returns best coeffs_real_imag as a flat array of length (degree+1)*2.
    """
    dim = (degree + 1) * 2
    # Initialize swarm within bounds based on target magnitude statistics
    mag = np.maximum(1e-6, np.median(np.abs(target)))
    pos = rng.uniform(low=-mag, high=mag, size=(swarm_size, dim)).astype(np.float32)
    vel = np.zeros_like(pos, dtype=np.float32)

    def fitness(p):
        pred = _poly_eval(k, p)
        # MSE on complex values
        err = target - pred
        return np.mean((err.real ** 2 + err.imag ** 2).astype(np.float32))

    fvals = np.array([fitness(p) for p in pos], dtype=np.float32)
    pbest = pos.copy()
    pbest_val = fvals.copy()
    g_idx = int(np.argmin(pbest_val))
    gbest = pbest[g_idx].copy()
    gbest_val = pbest_val[g_idx]

    for t in range(iters):
        w = w_start + (w_end - w_start) * (t / max(1, iters - 1))
        r1 = rng.random(size=(swarm_size, dim), dtype=np.float32)
        r2 = rng.random(size=(swarm_size, dim), dtype=np.float32)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = pos + vel

        fvals = np.array([fitness(p) for p in pos], dtype=np.float32)
        improved = fvals < pbest_val
        if np.any(improved):
            pbest[improved] = pos[improved]
            pbest_val[improved] = fvals[improved]
            g_idx = int(np.argmin(pbest_val))
            if pbest_val[g_idx] < gbest_val:
                gbest = pbest[g_idx].copy()
                gbest_val = pbest_val[g_idx]
    return gbest


class PSOChannelEstimator(Block):
    """PSO-regularized estimator that smooths LS estimates across frequency."""

    def __init__(
        self,
        config: SystemConfig,
        resource_grid: ResourceGrid,
        degree: int = 3,
        swarm_size: int = 32,
        iters: int = 60,
        inertia_start: float = 0.7,
        inertia_end: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._base = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self._rg = resource_grid
        self.degree = int(degree)
        self.swarm_size = int(swarm_size)
        self.iters = int(iters)
        self.inertia_start = float(inertia_start)
        self.inertia_end = float(inertia_end)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self._rng = np.random.default_rng(seed)

        # Precompute k index normalized to [-1, 1] for numerical stability
        n_sc = int(self._rg.fft_size)
        k = np.linspace(-1.0, 1.0, num=n_sc, dtype=np.float32)
        self._k = k

    def call(self, y: tf.Tensor, noise_variance: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Estimate channel with PSO smoothing.
        Args:
            y: Received resource grid (complex tensor), shape
               [B, rx, stream, n_sym, n_sc]
            noise_variance: Noise variance tensor (unused here beyond LS).
        Returns:
            (h_hat, err_var): Smoothed channel and LS-derived error variance.
        """
        # Initial LS estimate (already interpolated to all REs)
        h_ls, err_var = self._base(y, noise_variance)

        # Expect shape: [B, rx, tx, ut, stream, n_sym, n_sc]
        shape = tf.shape(h_ls)
        b = int(h_ls.shape[0])
        rx = int(h_ls.shape[1])
        tx = int(h_ls.shape[2])
        ut = int(h_ls.shape[3])
        stream = int(h_ls.shape[4])
        n_sym = int(h_ls.shape[5])
        n_sc = int(h_ls.shape[6])

        # Fallback to dynamic shape if not statically known
        if any(v is None for v in [b, rx, tx, ut, stream, n_sym, n_sc]):
            b = int(shape[0])
            rx = int(shape[1])
            tx = int(shape[2])
            ut = int(shape[3])
            stream = int(shape[4])
            n_sym = int(shape[5])
            n_sc = int(shape[6])

        k = self._k  # [-1, 1], length n_sc
        assert n_sc == k.shape[0], "Resource grid FFT size mismatch."

        h_np = h_ls.numpy()
        h_out = np.empty_like(h_np)

        # Loop: per batch, rx, tx, ut, stream, per symbol -> PSO over coefficients
        for ib in range(b):
            for ir in range(rx):
                for it in range(tx):
                    for iu in range(ut):
                        for is_ in range(stream):
                            for isym in range(n_sym):
                                target = h_np[ib, ir, it, iu, is_, isym, :]
                                best = _pso_optimize(
                                    target=target,
                                    k=k,
                                    degree=self.degree,
                                    swarm_size=self.swarm_size,
                                    iters=self.iters,
                                    w_start=self.inertia_start,
                                    w_end=self.inertia_end,
                                    c1=self.c1,
                                    c2=self.c2,
                                    rng=self._rng,
                                )
                                pred = _poly_eval(k, best).astype(target.dtype)
                                h_out[ib, ir, it, iu, is_, isym, :] = pred

        h_pred = tf.convert_to_tensor(h_out)
        return h_pred, err_var


