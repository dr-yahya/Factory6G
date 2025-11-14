"""Temporal (EMA) LS-based channel estimator across OFDM symbols.

This module implements a channel estimator that uses exponential moving average
(EMA) to track slowly-varying channels over time. The estimator applies LS
estimation per OFDM symbol and then smooths the estimates temporally using EMA.

Theory:
    Exponential Moving Average (EMA):
    
    1. Motivation:
       - Channels vary slowly over time (coherence time T_c)
       - Previous channel estimates contain useful information
       - EMA combines current and past estimates: Ĥ_EMA[t] = α·Ĥ_EMA[t-1] + (1-α)·Ĥ_LS[t]
       
    2. EMA Filter:
       - Update equation: y[t] = α·y[t-1] + (1-α)·x[t]
       - Smoothing factor: α ∈ [0, 1]
       - Larger α: More smoothing, slower response to changes
       - Smaller α: Less smoothing, faster response to changes
       
    3. Frequency Response:
       - EMA is a first-order IIR lowpass filter
       - Transfer function: H(z) = (1-α) / (1 - α·z^(-1))
       - 3-dB cutoff frequency: f_c = (1-α) / (2π·T_sym)
       - High-frequency noise is attenuated
       
    4. Optimal Smoothing Factor:
       - Depends on channel coherence time and Doppler spread
       - For slow fading: α ≈ 0.7-0.9 (heavy smoothing)
       - For fast fading: α ≈ 0.3-0.5 (light smoothing)
       - Adaptive α based on channel variation rate
       
    5. Channel Tracking:
       - EMA tracks channel variations over time
       - Useful for channels with significant temporal correlation
       - Performance improves when channel changes slowly
       - May lag behind rapid channel changes

References:
    - Proakis & Salehi, "Digital Communications" (Channel tracking)
    - Tse & Viswanath, "Fundamentals of Wireless Communication" (Channel estimation)
"""

from __future__ import annotations

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid

from ..config import SystemConfig


class TemporalEstimator(Block):
    """
    Apply exponential moving average over the OFDM symbol dimension to track
    slowly-varying channels. Uses LS per symbol then smooths temporally.
    
    This estimator applies LS estimation per OFDM symbol and then uses
    exponential moving average (EMA) to smooth the estimates over time.
    The EMA exploits temporal correlation to reduce noise and track channel
    variations.
    
    Theory:
        The estimation process:
        
        1. Per-Symbol LS Estimation:
           Ĥ_LS[t] = LS_Estimator(Y[t], X_pilot[t])
           
        2. Temporal EMA Smoothing:
           Ĥ_EMA[t] = α·Ĥ_EMA[t-1] + (1-α)·Ĥ_LS[t]
           
        where α is the smoothing factor.
        
        The EMA acts as a first-order IIR lowpass filter:
        - Attenuates high-frequency noise
        - Tracks slow channel variations
        - Trade-off: Noise reduction vs. tracking speed
        
        Optimal α depends on:
        - Channel coherence time: T_c ≈ 1/(2·f_d_max)
        - OFDM symbol duration: T_sym
        - Ratio: T_c / T_sym determines optimal α
    """

    def __init__(self, config: SystemConfig, resource_grid: ResourceGrid,
                 alpha: float = 0.7) -> None:
        """
        Initialize temporal EMA channel estimator.
        
        Creates a channel estimator that applies exponential moving average
        (EMA) to LS estimates over the OFDM symbol dimension. The EMA smooths
        the estimates temporally to reduce noise and track channel variations.
        
        Theory:
            EMA smoothing factor α:
            - α ∈ [0, 1]: Smoothing factor
            - α → 1: Heavy smoothing, slow response (good for slow fading)
            - α → 0: Light smoothing, fast response (good for fast fading)
            - Optimal α depends on channel coherence time and symbol duration
            
            For channels with coherence time T_c and symbol duration T_sym:
            - Slow fading (T_c >> T_sym): Use α ≈ 0.7-0.9
            - Fast fading (T_c ≈ T_sym): Use α ≈ 0.3-0.5
            
            The EMA acts as a lowpass filter with cutoff frequency:
            f_c = (1-α) / (2π·T_sym)
            
        Args:
            config: System configuration parameters.
            resource_grid: OFDM resource grid defining time-frequency structure.
            alpha: EMA smoothing factor. Default: 0.7 (good for moderate
                channel variations). Range: [0, 1]. Larger values provide
                more smoothing but slower response to channel changes.
        """
        super().__init__()
        self._base = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def _ema_time(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply EMA along the OFDM symbol dimension using tf.scan.
        
        Performs exponential moving average over the OFDM symbol dimension
        (temporal smoothing). Uses tf.scan for graph-safe sequential processing.
        
        Theory:
            EMA update equation:
            y[t] = α·y[t-1] + (1-α)·x[t]
            
            where:
            - y[t]: EMA output at time t
            - x[t]: Input at time t
            - α: Smoothing factor
            
            The EMA is computed sequentially over the symbol dimension:
            - Initial value: y[0] = x[0] (no previous value)
            - Subsequent values: y[t] = α·y[t-1] + (1-α)·x[t]
            
            tf.scan applies the update function sequentially:
            - scan_fn(prev, cur) = α·prev + (1-α)·cur
            - Processes symbols in order: t = 0, 1, 2, ..., T-1
        
        Args:
            x: Input tensor
                Shape: [batch, rx, tx, ut, stream, n_sym, n_sc]
                Last two dimensions are (OFDM symbols, subcarriers)
                
        Returns:
            EMA-smoothed tensor
            Same shape as input x
        """
        # Move symbol axis to first for scan: (..., n_sym, n_sc) -> (n_sym, ... , n_sc)
        # This allows tf.scan to process symbols sequentially
        x_t = tf.transpose(x, perm=[-2, 0, 1, 2, 3, 4, -1])
        
        def scan_fn(prev, cur):
            """EMA update function: y[t] = α·y[t-1] + (1-α)·x[t]"""
            return self.alpha * prev + (1.0 - self.alpha) * cur
        
        # Apply EMA sequentially over symbol dimension
        y_t = tf.scan(scan_fn, x_t)
        
        # Transpose back to original shape
        y = tf.transpose(y_t, perm=[1, 2, 3, 4, 5, 0, 6])
        return y

    def call(self, y: tf.Tensor, noise_variance: tf.Tensor):
        """
        Estimate channel using temporal EMA estimator.
        
        Performs LS estimation per OFDM symbol and then applies exponential
        moving average (EMA) to smooth the estimates over time. The EMA
        exploits temporal correlation to reduce noise and track channel variations.
        
        Theory:
            The estimation process:
            
            1. Per-Symbol LS Estimation:
               Ĥ_LS[t] = LS_Estimator(Y[t], X_pilot[t]) for each symbol t
               
            2. Temporal EMA Smoothing:
               Ĥ_EMA[t] = α·Ĥ_EMA[t-1] + (1-α)·Ĥ_LS[t]
               
            The EMA reduces noise variance approximately by:
            σ²_EMA ≈ σ²_LS · (1-α) / (1+α)
            
            However, EMA also introduces delay in tracking rapid channel changes.
            The trade-off depends on channel coherence time and smoothing factor α.
        
        Args:
            y: Received resource grid
                Shape: [batch_size, num_rx, num_streams, num_ofdm_symbols, fft_size]
            noise_variance: Noise variance per resource element
                Used by LS estimator to compute error variance
                
        Returns:
            Tuple of:
            - h_ema: EMA-smoothed channel estimate
                Shape: [batch_size, num_rx, num_tx, num_streams, num_ofdm_symbols, fft_size]
            - err_var: Channel estimation error variance
                Same as LS estimator (EMA may reduce actual error, but
                error variance computation is not updated)
        """
        # Stage 1: Per-symbol LS estimation
        h_ls, err_var = self._base(y, noise_variance)
        
        # Stage 2: Apply EMA separately to real and imaginary parts
        real = self._ema_time(tf.math.real(h_ls))
        imag = self._ema_time(tf.math.imag(h_ls))
        
        # Stage 3: Recombine into complex channel estimate
        h_ema = tf.complex(real, imag)
        
        return h_ema, err_var
