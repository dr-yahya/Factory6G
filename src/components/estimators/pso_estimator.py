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


def _poly_eval_vec(k: np.ndarray, coeffs_real_imag: np.ndarray) -> np.ndarray:
    """Evaluate complex polynomial with real-imag coefficients on k (vectorized).
    
    Args:
        k: Subcarrier indices [SC]
        coeffs_real_imag: Coefficients [N, Swarm, (degree+1)*2]
        
    Returns:
        y: Evaluated polynomial [N, Swarm, SC]
    """
    # coeffs_real_imag shape: [N, Swarm, Dim]
    # k shape: [SC]
    
    dim = coeffs_real_imag.shape[-1]
    degree = dim // 2 - 1
    
    # Extract real and imag parts
    # coeffs_real_imag is [..., 2*(degree+1)]
    # real parts at 0, 2, 4...
    real = coeffs_real_imag[..., 0::2] # [N, Swarm, degree+1]
    imag = coeffs_real_imag[..., 1::2] # [N, Swarm, degree+1]
    
    coeffs = real + 1j * imag  # [N, Swarm, degree+1]
    
    # Horner's method vectorized
    # We want output [N, Swarm, SC]
    # Initialize y with zeros
    shape = coeffs.shape[:-1] + (k.shape[0],) # [N, Swarm, SC]
    y = np.zeros(shape, dtype=np.complex64)
    
    # Reshape k for broadcasting: [1, 1, SC]
    k_reshaped = k.reshape(1, 1, -1)
    
    # Iterate through coefficients from highest degree
    for i in range(degree, -1, -1):
        c = coeffs[..., i] # [N, Swarm]
        c = c[..., np.newaxis] # [N, Swarm, 1]
        y = y * k_reshaped + c
        
    return y


def _pso_optimize_vec(
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
    """Run PSO to fit complex polynomial to target (complex) vs k (float) - Vectorized.
    
    Args:
        target: Target values [N, SC]
        k: Subcarrier indices [SC]
        
    Returns:
        best_coeffs: [N, Dim]
    """
    num_problems = target.shape[0]
    dim = (degree + 1) * 2
    
    # Initialize swarm
    # [N, Swarm, Dim]
    # Initialize within bounds based on target magnitude
    # Compute median magnitude per problem: [N, 1, 1]
    mag = np.maximum(1e-6, np.median(np.abs(target), axis=1))
    mag = mag.reshape(num_problems, 1, 1)
    
    pos = rng.uniform(low=-1.0, high=1.0, size=(num_problems, swarm_size, dim)).astype(np.float32)
    pos = pos * mag # Scale by magnitude
    
    vel = np.zeros_like(pos, dtype=np.float32)
    
    # Target reshaping for broadcasting: [N, 1, SC]
    target_expanded = target.reshape(num_problems, 1, -1)
    
    def evaluate_fitness(p):
        # p: [N, Swarm, Dim]
        # Returns: [N, Swarm]
        pred = _poly_eval_vec(k, p) # [N, Swarm, SC]
        err = target_expanded - pred
        # Mean squared error over subcarriers
        mse = np.mean(err.real**2 + err.imag**2, axis=-1)
        return mse.astype(np.float32)

    # Initial evaluation
    fvals = evaluate_fitness(pos) # [N, Swarm]
    
    pbest = pos.copy()
    pbest_val = fvals.copy()
    
    # Find global best per problem
    # argmin over swarm dimension
    g_idx = np.argmin(pbest_val, axis=1) # [N]
    
    # Extract gbest: [N, Dim]
    # We need advanced indexing
    row_indices = np.arange(num_problems)
    gbest = pbest[row_indices, g_idx, :].copy() # [N, Dim]
    gbest_val = pbest_val[row_indices, g_idx].copy() # [N]
    
    # Reshape gbest for broadcasting: [N, 1, Dim]
    gbest_expanded = gbest.reshape(num_problems, 1, dim)

    for t in range(iters):
        w = w_start + (w_end - w_start) * (t / max(1, iters - 1))
        
        r1 = rng.random(size=(num_problems, swarm_size, dim), dtype=np.float32)
        r2 = rng.random(size=(num_problems, swarm_size, dim), dtype=np.float32)
        
        # Update velocity
        # gbest_expanded: [N, 1, Dim] broadcasts to [N, Swarm, Dim]
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest_expanded - pos)
        pos = pos + vel
        
        # Evaluate
        fvals = evaluate_fitness(pos)
        
        # Update pbest
        improved = fvals < pbest_val # [N, Swarm]
        pbest[improved] = pos[improved]
        pbest_val[improved] = fvals[improved]
        
        # Update gbest
        # Find best in current pbest
        current_best_idx = np.argmin(pbest_val, axis=1) # [N]
        current_best_val = pbest_val[row_indices, current_best_idx] # [N]
        
        improved_g = current_best_val < gbest_val # [N]
        
        if np.any(improved_g):
            gbest[improved_g] = pbest[improved_g, current_best_idx[improved_g], :]
            gbest_val[improved_g] = current_best_val[improved_g]
            gbest_expanded = gbest.reshape(num_problems, 1, dim)
            
    return gbest


class PSOChannelEstimator(Block):
    """PSO-regularized estimator that smooths LS estimates across frequency."""

    def __init__(
        self,
        config: dict,
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
        # We need concrete values for reshaping
        
        h_np = h_ls.numpy()
        orig_shape = h_np.shape
        n_sc = orig_shape[-1]
        
        # Flatten all dimensions except subcarriers
        # [N_problems, SC]
        h_flat = h_np.reshape(-1, n_sc)
        
        k = self._k
        assert n_sc == k.shape[0], "Resource grid FFT size mismatch."

        # Run vectorized PSO
        best_coeffs = _pso_optimize_vec(
            target=h_flat,
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
        
        # Evaluate polynomial with best coefficients
        # best_coeffs: [N_problems, Dim]
        # We need to reshape for _poly_eval_vec which expects [N, Swarm, Dim]
        # Here we just have 1 "particle" (the best one) per problem
        best_coeffs_expanded = best_coeffs[:, np.newaxis, :] # [N_problems, 1, Dim]
        
        pred = _poly_eval_vec(k, best_coeffs_expanded) # [N_problems, 1, SC]
        pred = pred.squeeze(axis=1) # [N_problems, SC]
        
        # Reshape back to original
        h_out = pred.reshape(orig_shape)

        h_pred = tf.convert_to_tensor(h_out)
        return h_pred, err_var
