"""Smoothed LS-based channel estimator using 2D convolutional smoothing.

This module implements a channel estimator that applies 2D smoothing filters
to least squares (LS) channel estimates. The smoothing reduces noise by
exploiting channel correlation in time (OFDM symbols) and frequency (subcarriers).

Theory:
    Channel Smoothing:
    
    1. Motivation:
       - LS estimates are noisy: Ĥ_LS = H + ε, where ε ~ CN(0, σ²)
       - Channel is correlated in time and frequency due to:
         * Coherence time: Channel changes slowly over time
         * Coherence bandwidth: Channel varies slowly over frequency
       - Smoothing exploits correlation to reduce noise
    
    2. Separable 2D Filtering:
       - Time smoothing: Average over adjacent OFDM symbols
       - Frequency smoothing: Average over adjacent subcarriers
       - Separable filter: h(t,f) = h_t(t) · h_f(f)
       - Efficient implementation using 1D convolutions
       
    3. Box Filter (Moving Average):
       - Kernel: h[n] = 1/N for n = 0, ..., N-1
       - Frequency response: H(ω) = sin(Nω/2) / (N·sin(ω/2)) · e^(-jω(N-1)/2)
       - Reduces noise variance by factor 1/N
       - Trade-off: Noise reduction vs. resolution loss
       
    4. Optimal Smoothing:
       - Wiener filter: Optimal linear filter for stationary signals
       - Requires knowledge of signal and noise statistics
       - Box filter is suboptimal but simple and effective
       
    5. Channel Correlation:
       - Time correlation: R_t(Δt) = E[H(t)·H*(t+Δt)] = J₀(2πf_d·Δt)
         where J₀ is Bessel function, f_d is Doppler spread
       - Frequency correlation: R_f(Δf) = E[H(f)·H*(f+Δf)] = sinc(2πτ_rms·Δf)
         where τ_rms is RMS delay spread

References:
    - Proakis & Salehi, "Digital Communications" (Wiener filtering)
    - Tse & Viswanath, "Fundamentals of Wireless Communication" (Channel correlation)
"""

from __future__ import annotations

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid

from ..config import SystemConfig


class SmoothedLSEstimator(Block):
    """
    Apply a separable 2D smoothing filter to the LS estimate across
    (ofdm_symbol, subcarrier) dimensions.
    
    This estimator applies 2D smoothing to LS channel estimates to reduce
    noise. The smoothing is performed separately in time (OFDM symbols) and
    frequency (subcarriers) dimensions using box filters (moving average).
    
    Theory:
        The smoothing operation:
        
        1. LS Estimation:
           Ĥ_LS = LS_Estimator(Y, X_pilot)
           
        2. Time Smoothing:
           Ĥ_smooth_time[t,f] = (1/K_t) · Σ_{k=0}^{K_t-1} Ĥ_LS[t+k-K_t/2, f]
           
        3. Frequency Smoothing:
           Ĥ_smooth[t,f] = (1/K_f) · Σ_{k=0}^{K_f-1} Ĥ_smooth_time[t, f+k-K_f/2]
        
        The separable filter reduces noise variance by approximately:
        σ²_smooth ≈ σ²_LS / (K_t · K_f)
        
        Trade-offs:
        - Larger kernels: More noise reduction, but more resolution loss
        - Smaller kernels: Less noise reduction, but better resolution
        - Optimal kernel size depends on channel coherence time/bandwidth
    """

    def __init__(self, config: SystemConfig, resource_grid: ResourceGrid,
                 kernel_time: int = 3, kernel_freq: int = 5) -> None:
        """
        Initialize smoothed LS channel estimator.
        
        Creates a channel estimator that applies 2D smoothing to LS estimates.
        The smoothing is performed using separable box filters in time and
        frequency dimensions.
        
        Theory:
            Box filter (moving average) characteristics:
            - Kernel size K: Number of samples to average
            - Normalization: 1/K ensures unity gain
            - Noise reduction: Variance reduced by factor 1/K
            - Resolution: Spatial resolution reduced by factor K
            
            Optimal kernel sizes depend on:
            - Coherence time: T_c ≈ 1/(2·f_d_max)
            - Coherence bandwidth: B_c ≈ 1/(5·τ_rms)
            - Pilot spacing: Should be smaller than coherence time/bandwidth
            
        Args:
            config: System configuration parameters.
            resource_grid: OFDM resource grid defining time-frequency structure.
            kernel_time: Time smoothing kernel size (OFDM symbols).
                Larger values provide more noise reduction but reduce time resolution.
                Should be odd for symmetric filtering. Default: 3.
            kernel_freq: Frequency smoothing kernel size (subcarriers).
                Larger values provide more noise reduction but reduce frequency resolution.
                Should be odd for symmetric filtering. Default: 5.
        """
        super().__init__()
        self._base = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self.kernel_time = kernel_time
        self.kernel_freq = kernel_freq
        
        # Build normalized box filters for separable 2D convolution
        # Time filter: [kernel_time, 1, 1, 1] for vertical (time) smoothing
        kt = tf.ones([kernel_time, 1, 1, 1], dtype=tf.float32) / float(kernel_time)
        # Frequency filter: [1, kernel_freq, 1, 1] for horizontal (frequency) smoothing
        kf = tf.ones([1, kernel_freq, 1, 1], dtype=tf.float32) / float(kernel_freq)
        self.kt = kt
        self.kf = kf

    def _smooth(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply separable smoothing on the last two dimensions: (symbols, subcarriers).
        
        Performs 2D smoothing by applying 1D filters separately in time and
        frequency dimensions. The smoothing is applied independently to real
        and imaginary parts of the channel estimate.
        
        Theory:
            Separable 2D convolution:
            - 2D filter h[t,f] = h_t[t] · h_f[f] can be applied as:
              1. Convolve with h_t along time dimension
              2. Convolve with h_f along frequency dimension
            - Computational complexity: O(N·(K_t + K_f)) vs O(N·K_t·K_f) for 2D
            - Equivalent result for separable filters
            
            The smoothing operation:
            - Separates real and imaginary parts
            - Applies time smoothing (vertical convolution)
            - Applies frequency smoothing (horizontal convolution)
            - Recombines into complex channel estimate
        
        Args:
            x: Channel estimate to smooth
                Shape: [batch, rx, tx, ut, stream, n_sym, n_sc]
                Last two dimensions are (OFDM symbols, subcarriers)
                
        Returns:
            Smoothed channel estimate
            Same shape as input x
        """
        # Expect shape: [B, rx, tx, ut, stream, n_sym, n_sc]
        # Move (sym, sc) to spatial dims for conv2d: NHWC format
        real = tf.math.real(x)
        imag = tf.math.imag(x)
        
        def smooth_real_imag(t):
            """Apply 2D smoothing to real or imaginary part."""
            # Reshape to [batch*other_dims, n_sym, n_sc, 1] for conv2d
            t4 = tf.reshape(t, [-1, tf.shape(t)[-2], tf.shape(t)[-1], 1])
            # Time smoothing (vertical convolution along OFDM symbols)
            y = tf.nn.conv2d(t4, self.kt, strides=1, padding='SAME')
            # Frequency smoothing (horizontal convolution along subcarriers)
            y = tf.nn.conv2d(y, self.kf, strides=1, padding='SAME')
            # Reshape back to original shape
            y = tf.reshape(y, tf.shape(t))
            return y
        
        # Apply smoothing to real and imaginary parts separately
        real_s = smooth_real_imag(real)
        imag_s = smooth_real_imag(imag)
        
        # Recombine into complex channel estimate
        return tf.complex(real_s, imag_s)

    def call(self, y: tf.Tensor, noise_variance: tf.Tensor):
        """
        Estimate channel using smoothed LS estimator.
        
        Performs LS estimation followed by 2D smoothing to reduce noise.
        The smoothing exploits channel correlation in time and frequency.
        
        Theory:
            The estimation process:
            
            1. LS Estimation:
               Ĥ_LS = LS_Estimator(Y, X_pilot)
               
            2. 2D Smoothing:
               Ĥ_smooth = Smooth_2D(Ĥ_LS)
               
            The smoothing reduces noise variance approximately by:
            σ²_smooth ≈ σ²_LS / (K_t · K_f)
            
            However, smoothing also reduces resolution and may blur rapid
            channel variations. The trade-off depends on channel coherence
            time and bandwidth.
        
        Args:
            y: Received resource grid
                Shape: [batch_size, num_rx, num_streams, num_ofdm_symbols, fft_size]
            noise_variance: Noise variance per resource element
                Used by LS estimator to compute error variance
                
        Returns:
            Tuple of:
            - h_sm: Smoothed channel estimate
                Shape: [batch_size, num_rx, num_tx, num_streams, num_ofdm_symbols, fft_size]
            - err_var: Channel estimation error variance
                Same as LS estimator (smoothing may reduce actual error, but
                error variance computation is not updated)
        """
        # Stage 1: LS estimation
        h_ls, err_var = self._base(y, noise_variance)
        
        # Stage 2: 2D smoothing
        h_sm = self._smooth(h_ls)
        
        return h_sm, err_var
