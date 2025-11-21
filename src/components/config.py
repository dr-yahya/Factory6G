"""
System configuration parameters for 6G smart factory physical layer.

This module defines the SystemConfig dataclass that encapsulates all system
parameters for OFDM-MIMO wireless communication systems. The configuration
follows 5G/6G standards and 3GPP TR 38.901 channel modeling specifications.

Theory:
    OFDM (Orthogonal Frequency Division Multiplexing) divides the available
    bandwidth into multiple orthogonal subcarriers. The key parameters are:
    
    - FFT Size: Number of subcarriers (typically 128, 256, 512, 1024, 2048)
      Determines the frequency resolution and computational complexity.
      
    - Subcarrier Spacing (Δf): Spacing between adjacent subcarriers.
      For 5G NR: 15 kHz, 30 kHz, 60 kHz, 120 kHz, 240 kHz
      Formula: Δf = 1 / T_sym, where T_sym is the OFDM symbol duration
      
    - Cyclic Prefix (CP): Guard interval to combat inter-symbol interference
      (ISI) and maintain orthogonality in multipath channels.
      Length must exceed channel delay spread: L_CP > τ_max
      
    - Resource Grid: 2D grid (time × frequency) representing OFDM symbols
      and subcarriers. Each resource element (RE) can carry modulation symbols
      or pilot signals for channel estimation.
      
    MIMO (Multiple Input Multiple Output) systems use multiple antennas to
    increase spectral efficiency and reliability:
    
    - Spatial Multiplexing: Transmit multiple independent data streams
      Capacity scales linearly with min(N_tx, N_rx) in rich scattering
      
    - Diversity: Transmit same signal from multiple antennas to improve
      reliability through spatial diversity
      
    - Beamforming: Use phase/amplitude control to focus energy toward
      intended receivers
      
    Channel coding provides error correction capability:
    
    - Code Rate (R): Ratio of information bits to coded bits (R = k/n)
      Lower code rate = more redundancy = better error correction but lower
      spectral efficiency
      
    - LDPC (Low-Density Parity-Check) codes: Modern channel codes used in
      5G/6G with near-Shannon-limit performance
      
    Modulation maps bits to complex symbols:
    
    - QPSK: 2 bits per symbol, 4 constellation points
    - 16-QAM: 4 bits per symbol, 16 constellation points
    - 64-QAM: 6 bits per symbol, 64 constellation points
    - Higher order = higher data rate but requires better SNR

References:
    - 3GPP TS 38.211: Physical channels and modulation
    - 3GPP TS 38.212: Multiplexing and channel coding
    - 3GPP TR 38.901: Study on channel model for frequencies from 0.5 to 100 GHz
    - Proakis & Salehi, "Digital Communications", 5th Edition
    - Tse & Viswanath, "Fundamentals of Wireless Communication"
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SystemConfig:
    """
    Configuration parameters for the 6G smart factory system.
    
    This dataclass encapsulates all physical layer parameters for OFDM-MIMO
    systems, including RF parameters, frame structure, MIMO configuration,
    modulation/coding schemes, and channel model settings.
    
    Attributes:
        carrier_frequency: Carrier frequency in Hz. Default 3.5 GHz (mid-band 5G).
            For 6G, sub-6 GHz (6.425-7.125 GHz) and mmWave (24-100 GHz) are
            being considered. Higher frequencies enable larger bandwidths but
            suffer from higher path loss.
            
        fft_size: FFT/IFFT size, determining the number of subcarriers.
            Must be power of 2. Larger values provide better frequency resolution
            but increase computational complexity. Typical values: 128, 256, 512.
            
        subcarrier_spacing: Spacing between adjacent subcarriers in Hz.
            For 5G NR: 15 kHz (normal CP), 30 kHz, 60 kHz, 120 kHz, 240 kHz.
            Larger spacing reduces sensitivity to Doppler shift but increases
            overhead from cyclic prefix. Formula: Δf = 1 / T_sym.
            
        num_ofdm_symbols: Number of OFDM symbols per slot/frame.
            Default 14 symbols per slot (normal CP in 5G). Each symbol duration
            is T_sym = 1/Δf + T_CP, where T_CP is cyclic prefix duration.
            
        cyclic_prefix_length: Length of cyclic prefix in samples.
            Must exceed maximum channel delay spread to prevent inter-symbol
            interference (ISI). Typical: 20-25% of FFT size for normal CP.
            Formula: L_CP = τ_max * f_s, where f_s is sampling rate.
            
        pilot_ofdm_symbol_indices: OFDM symbol indices containing pilot signals.
            Pilots are known reference signals used for channel estimation.
            Typically placed at regular intervals (e.g., symbols 2 and 11).
            More pilots = better channel estimation but lower data rate.
            
        num_bs_ant: Number of base station (receiver) antennas.
            For MIMO systems, more antennas enable spatial multiplexing and
            beamforming. Massive MIMO uses hundreds of antennas for improved
            spectral efficiency and interference suppression.
            
        num_ut: Number of user terminals (transmitters in uplink).
            Multiple users share the same time-frequency resources through
            spatial multiplexing. Limited by min(N_BS, N_UT_total).
            
        num_ut_ant: Number of antennas per user terminal.
            Typically 1-4 antennas for mobile devices. More antennas enable
            MIMO transmission but increase device complexity.
            
        num_bits_per_symbol: Number of bits per modulation symbol.
            2 = QPSK, 4 = 16-QAM, 6 = 64-QAM, 8 = 256-QAM.
            Higher order modulations provide higher data rates but require
            better signal-to-noise ratio (SNR).
            
        coderate: Channel code rate (k/n), ratio of information bits to coded bits.
            Range: (0, 1]. Lower rate = more redundancy = better error correction.
            Typical values: 0.33, 0.5, 0.67, 0.75. Selected based on channel
            conditions through link adaptation.
            
        scenario: Channel scenario string ("umi", "uma", "rma").
            - UMi: Urban Microcell (dense urban, <100m cell radius)
            - UMa: Urban Macrocell (urban, >1km cell radius)
            - RMa: Rural Macrocell (rural, >5km cell radius)
            Each scenario has different path loss, shadowing, and multipath
            characteristics as defined in 3GPP TR 38.901.
            
        direction: Transmission direction ("uplink" or "downlink").
            Uplink: User terminals transmit to base station
            Downlink: Base station transmits to user terminals
            Channel characteristics may differ due to different power levels
            and interference environments.
            
        o2i_model: Outdoor-to-indoor penetration model ("low", "high").
            Models signal attenuation when penetrating buildings. Important for
            smart factory scenarios where base stations may be outdoor while
            devices are indoor.
            
        enable_pathloss: Whether to enable large-scale path loss modeling.
            Path loss models large-scale signal attenuation with distance.
            Formula: PL(d) = PL_0 + 10α log10(d/d_0) + X_σ, where α is path
            loss exponent and X_σ is shadow fading.
            
        enable_shadow_fading: Whether to enable shadow fading modeling.
            Shadow fading models slow variations due to obstacles (buildings,
            trees). Typically modeled as log-normal distribution with standard
            deviation 3-10 dB.
            
        active_ut_mask: Binary mask indicating which UTs are scheduled.
            Length = num_ut, values 0 or 1. Used for dynamic scheduling and
            interference management. 1 = active/scheduled, 0 = muted.
            
        per_ut_power: Per-UT power scaling factors (linear scale).
            Length = num_ut. Used for power control to balance received signal
            strengths and manage interference. Values typically in range [0, 1].
            Applied as: x_scaled = x * sqrt(power_factor).
            
        pilot_reuse_factor: Pilot reuse factor for interference management.
            Reusing pilots across cells causes pilot contamination. Higher reuse
            reduces contamination but requires more orthogonal pilot sequences.
            Factor of 1 means no reuse (all cells use same pilots).
            
        target_bler: Target block error rate for link adaptation.
            BLER = 1 - (1 - BER)^n, where n is block length.
            Used to select appropriate modulation and coding scheme (MCS).
            Typical target: 1e-3 to 1e-2 for data channels.
    """
    
    # RF Parameters
    carrier_frequency: float = 3.5e9  # Hz (3.5 GHz)
    fft_size: int = 512  # Enforce minimum per 6G params (never less)
    subcarrier_spacing: float = 30e3  # Hz (30 kHz)
    
    # OFDM Frame Structure
    num_ofdm_symbols: int = 14  # Updated to max params: 3GPP standard
    cyclic_prefix_length: int = 20
    pilot_ofdm_symbol_indices: List[int] = None
    
    # MIMO Configuration
    num_bs_ant: int = 32  # Enforce minimum per 6G params (never less)
    num_ut: int = 8  # Enforce minimum per 6G params (never less)
    num_ut_ant: int = 2  # Enforce minimum per 6G params (never less)
    
    # Modulation and Coding
    num_bits_per_symbol: int = 2  # QPSK - stable for error-free baseline
    coderate: float = 0.5  # Moderate code rate for stability
    
    # Channel Model
    scenario: str = "umi"  # UMi, UMa, RMa
    direction: str = "uplink"  # "uplink" or "downlink"
    o2i_model: str = "low"  # Outdoor-to-indoor model
    enable_pathloss: bool = False
    enable_shadow_fading: bool = False
    
    # Resource Management (hooks)
    # Active UT mask (length = num_ut); 1 = scheduled, 0 = muted
    active_ut_mask: Optional[List[int]] = None
    # Per-UT linear power scaling (length = num_ut), applied at TX symbols
    per_ut_power: Optional[List[float]] = None
    # Pilot reuse factor placeholder (e.g., 1 = no reuse)
    pilot_reuse_factor: int = 1
    # Target BLER for link adaptation decisions
    target_bler: float = 1e-3
    
    # Channel Model Type
    channel_model_type: str = "tr38901"  # "tr38901" or "rayleigh"

    # Mobility
    min_ut_velocity: float = 0.0  # m/s
    max_ut_velocity: float = 0.0  # m/s
    
    def __post_init__(self):
        """
        Initialize default values after dataclass initialization.
        
        Sets default values for optional parameters that depend on other
        configuration parameters. This method is automatically called by the
        dataclass after __init__.
        """
        if self.pilot_ofdm_symbol_indices is None:
            # Default pilot placement: symbols 2 and 11 (typical for 14-symbol slot)
            # Provides good time-domain sampling for channel tracking
            self.pilot_ofdm_symbol_indices = [2, 11]
        if self.active_ut_mask is None:
            # Default: all UTs are active (scheduled)
            self.active_ut_mask = [1] * self.num_ut
        if self.per_ut_power is None:
            # Default: full power for all UTs (no power control)
            self.per_ut_power = [1.0] * self.num_ut
    
    @property
    def num_tx(self) -> int:
        """
        Number of transmitters (UTs for uplink).
        
        In uplink MIMO systems, each user terminal is a transmitter.
        This property provides a convenient alias for num_ut in the uplink context.
        
        Returns:
            Number of transmitters (equal to num_ut for uplink)
        """
        return self.num_ut
    
    @property
    def num_streams_per_tx(self) -> int:
        """
        Number of data streams per transmitter.
        
        In MIMO systems, each transmitter can send multiple independent data
        streams (spatial multiplexing). The number of streams is limited by
        the number of transmit antennas and channel rank.
        
        Returns:
            Number of streams per transmitter (equal to num_ut_ant)
        """
        return self.num_ut_ant
    
    def get_rx_tx_association(self) -> np.ndarray:
        """
        Create RX-TX association matrix for MIMO stream management.
        
        The association matrix defines which receivers (base station antennas)
        receive signals from which transmitters (user terminals). This is used
        for MIMO stream management and interference coordination.
        
        Theory:
            In MIMO systems, the association matrix A has shape [num_rx, num_tx],
            where A[i,j] = 1 indicates that receiver i receives at least one
            stream from transmitter j. For single-cell uplink, all UTs are
            associated with the single BS, so the matrix is all ones.
            
            The association matrix is used to:
            1. Group streams for equalization
            2. Manage interference between streams
            3. Support multi-cell scenarios with coordinated multipoint (CoMP)
        
        Returns:
            Association matrix of shape [1, num_ut] where all elements are 1.
            The first dimension represents the single base station, and the
            second dimension represents user terminals. All UTs are associated
            with the BS in this single-cell configuration.
        """
        bs_ut_association = np.zeros([1, self.num_ut])
        bs_ut_association[0, :] = 1
        return bs_ut_association

