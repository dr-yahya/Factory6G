"""
Channel model components for 6G smart factory systems.

This module implements wireless channel models following 3GPP TR 38.901
specifications. The channel model captures large-scale fading (path loss,
shadowing) and small-scale fading (multipath, Doppler) effects in realistic
wireless propagation environments.

Theory:
    Wireless channels are characterized by multiple propagation effects:
    
    1. Path Loss:
       - Large-scale signal attenuation with distance
       - Free-space path loss: PL_fs = (4πd/λ)²
       - Empirical models: PL(d) = PL_0 + 10α·log10(d/d_0) + X_σ
       - Path loss exponent α: 2 (free space), 2-4 (urban), 3-5 (indoor)
       - Reference distance d_0: Typically 1m or 10m
       
    2. Shadow Fading:
       - Slow variations due to obstacles (buildings, trees)
       - Log-normal distribution: X_σ ~ N(0, σ²)
       - Standard deviation σ: 3-10 dB (depends on environment)
       - Spatial correlation: e^(-d/d_cor), where d_cor is correlation distance
       
    3. Multipath Fading:
       - Fast variations due to multiple signal paths
       - Small-scale fading: Rayleigh (NLOS) or Rician (LOS)
       - Delay spread: τ_rms = √(Σ(τ_i - τ_mean)² · P_i)
       - Coherence bandwidth: B_c ≈ 1/(5·τ_rms)
       - Frequency-selective fading when B > B_c
       
    4. Doppler Effect:
       - Frequency shift due to mobility: f_d = (v/c)·f_c·cos(θ)
       - Doppler spread: f_d_max = v·f_c/c
       - Coherence time: T_c ≈ 1/(2·f_d_max)
       - Fast fading when T_sym < T_c
       
    5. MIMO Channel:
       - Channel matrix H: y = H·x + n
       - H[i,j]: Channel from TX antenna j to RX antenna i
       - Spatial correlation: R = E[H·H^H]
       - Channel rank: rank(H) ≤ min(N_tx, N_rx)
       - Capacity: C = log2(det(I + (ρ/N_tx)·H·H^H)) bits/s/Hz
       
    6. 3GPP TR 38.901 Channel Model:
       - Clustered delay line (CDL) model
       - Multiple clusters with different delays and angles
       - Each cluster has multiple rays (subpaths)
       - Power delay profile (PDP): P(τ) = Σ P_cluster · δ(τ - τ_cluster)
       - Angular spread: AS = √(Σ(θ_i - θ_mean)² · P_i)
       
    Scenarios:
        - UMi (Urban Microcell): Dense urban, <100m cell radius
          * High building density, street-level deployment
          * High multipath, moderate delay spread
          
        - UMa (Urban Macrocell): Urban, >1km cell radius
          * Rooftop deployment, larger coverage
          * Moderate multipath, larger delay spread
          
        - RMa (Rural Macrocell): Rural, >5km cell radius
          * Open areas, few obstacles
          * Lower multipath, larger delay spread
        
    OFDM Channel:
        The channel is applied in frequency domain per subcarrier:
        Y[k] = H[k]·X[k] + N[k]
        
        where:
        - Y[k]: Received signal at subcarrier k
        - H[k]: Channel frequency response at subcarrier k
        - X[k]: Transmitted symbol at subcarrier k
        - N[k]: AWGN noise at subcarrier k
        
        Channel frequency response:
        H[k] = Σ h[n]·exp(-j2πkn/N)
        
        where h[n] is the channel impulse response and N is FFT size.

References:
    - 3GPP TR 38.901: Study on channel model for frequencies from 0.5 to 100 GHz
    - Molisch, "Wireless Communications", 2nd Edition
    - Tse & Viswanath, "Fundamentals of Wireless Communication"
    - Rappaport, "Wireless Communications: Principles and Practice"
"""

import tensorflow as tf
from sionna.phy.channel.tr38901 import UMi, UMa, RMa
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel import OFDMChannel
from sionna.phy.ofdm import ResourceGrid
from .config import SystemConfig
from .antenna import AntennaConfig


class ChannelModel:
    """
    Channel model for 6G smart factory environments.
    
    This class implements wireless channel models following 3GPP TR 38.901
    specifications. It supports multiple scenarios (UMi, UMa, RMa) and applies
    the channel in the frequency domain for OFDM systems.
    
    Theory:
        The channel model captures:
        1. Large-scale fading: Path loss, shadow fading
        2. Small-scale fading: Multipath, Doppler, spatial correlation
        3. MIMO effects: Spatial multiplexing, antenna correlation
        4. Frequency selectivity: Delay spread, coherence bandwidth
        
        The channel is applied per OFDM subcarrier:
        Y[k] = H[k]·X[k] + N[k]
        
        where H[k] is the frequency-domain channel response computed from
        the time-domain channel impulse response via FFT.
    """
    
    def __init__(self, config: SystemConfig, antenna_config: AntennaConfig, 
                 resource_grid: ResourceGrid):
        """
        Initialize channel model.
        
        Creates a 3GPP TR 38.901 channel model for the specified scenario
        and wraps it with an OFDM channel that applies the channel in the
        frequency domain and adds AWGN noise.
        
        Theory:
            The channel model configuration includes:
            - Carrier frequency: Affects path loss and Doppler
            - Antenna arrays: Determines spatial correlation and MIMO gains
            - Scenario: Defines propagation environment (UMi/UMa/RMa)
            - Direction: Uplink vs downlink (different power levels)
            - Path loss and shadow fading: Large-scale effects
            
            The OFDM channel performs:
            1. Converts time-domain channel to frequency domain
            2. Applies channel per subcarrier: Y[k] = H[k]·X[k]
            3. Adds AWGN: Y[k] = H[k]·X[k] + N[k]
            4. Normalizes channel: E[|H[k]|²] = 1 (optional)
            
        Args:
            config: System configuration parameters including scenario,
                carrier frequency, and channel model options.
            antenna_config: Antenna configuration for BS and UTs, affecting
                spatial correlation and MIMO channel characteristics.
            resource_grid: OFDM resource grid defining the time-frequency
                structure for channel application.
        """
        super().__init__()
        self.config = config
        self.antenna_config = antenna_config
        self.resource_grid = resource_grid
        
        # Create channel model based on scenario
        self._channel_model = self._create_channel_model()
        
        # Create OFDM channel that applies channel in frequency domain
        self._ofdm_channel = OFDMChannel(
            self._channel_model,
            resource_grid,
            add_awgn=True,          # Add additive white Gaussian noise
            normalize_channel=True, # Normalize channel power to 1
            return_channel=True     # Return channel response for receiver
        )
    
    def _create_channel_model(self):
        """
        Create 3GPP TR 38.901 channel model based on scenario.
        
        Creates the appropriate channel model (UMi, UMa, or RMa) according
        to the system configuration. Each scenario has different propagation
        characteristics:
        
        - UMi (Urban Microcell): Dense urban, high multipath, moderate delay spread
        - UMa (Urban Macrocell): Urban, moderate multipath, larger delay spread
        - RMa (Rural Macrocell): Rural, lower multipath, larger delay spread
        
        Theory:
            The 3GPP TR 38.901 channel model uses a clustered delay line (CDL)
            approach:
            - Multiple clusters with different delays and angles
            - Each cluster contains multiple rays (subpaths)
            - Power delay profile: P(τ) = Σ P_cluster · δ(τ - τ_cluster)
            - Angular spread determines spatial correlation
            - Delay spread determines frequency selectivity
            
        Returns:
            Channel model instance (UMi, UMa, or RMa) configured with the
            specified parameters.
            
        Raises:
            ValueError: If the scenario is not supported.
        """
        channel_params = {
            'carrier_frequency': self.config.carrier_frequency,
            'o2i_model': self.config.o2i_model,
            'ut_array': self.antenna_config.get_ut_array(),
            'bs_array': self.antenna_config.get_bs_array(),
            'direction': self.config.direction,
            'enable_pathloss': self.config.enable_pathloss,
            'enable_shadow_fading': self.config.enable_shadow_fading
        }
        
        scenario_lower = self.config.scenario.lower()
        if scenario_lower == "umi":
            return UMi(**channel_params)
        elif scenario_lower == "uma":
            return UMa(**channel_params)
        elif scenario_lower == "rma":
            return RMa(**channel_params)
        else:
            raise ValueError(f"Unknown scenario: {self.config.scenario}. "
                           f"Supported: 'umi', 'uma', 'rma'")
    
    def set_topology(self, batch_size: int):
        """
        Generate and set new topology for the channel.
        
        Generates random user terminal positions and orientations for each
        batch. The topology determines the large-scale channel parameters
        (path loss, shadowing) and small-scale parameters (multipath, angles).
        
        Theory:
            The topology defines:
            - UT positions: (x, y, z) coordinates relative to BS
            - UT orientations: Azimuth and elevation angles
            - Distance: Determines path loss (PL ∝ d^α)
            - Angles: Determine antenna gains and spatial correlation
            
            For each batch, new topologies are generated to simulate
            different channel realizations and user positions.
            
        Args:
            batch_size: Number of channel realizations (UT positions) to
                generate. Each realization corresponds to a different user
                terminal configuration.
        """
        topology = gen_topology(
            batch_size,
            self.config.num_ut,
            self.config.scenario,
            min_ut_velocity=0.0,  # Static UTs (no mobility)
            max_ut_velocity=0.0   # Static UTs (no mobility)
        )
        self._channel_model.set_topology(*topology)
    
    def call(self, x_rg: tf.Tensor, noise_var: tf.Tensor) -> tuple:
        """
        Apply channel to input signal.
        
        Applies the wireless channel to the transmitted signal in the frequency
        domain and adds AWGN noise. The channel is applied per OFDM subcarrier.
        
        Theory:
            The channel application can be described as:
            
            Y[k] = H[k]·X[k] + N[k]
            
            where:
            - Y[k]: Received signal at subcarrier k
            - H[k]: Channel frequency response at subcarrier k
            - X[k]: Transmitted symbol at subcarrier k
            - N[k]: AWGN noise: N[k] ~ CN(0, σ²)
            
            Channel frequency response:
            H[k] = FFT{h[n]} = Σ h[n]·exp(-j2πkn/N)
            
            where h[n] is the time-domain channel impulse response.
            
            For MIMO systems:
            Y[k] = H[k]·X[k] + N[k]
            
            where:
            - Y[k]: [num_rx_ant, 1] received vector
            - H[k]: [num_rx_ant, num_tx_ant] channel matrix
            - X[k]: [num_tx_ant, 1] transmitted vector
            - N[k]: [num_rx_ant, 1] noise vector
            
        Args:
            x_rg: Input signal in resource grid format
                Shape: [batch_size, num_tx, num_streams, num_ofdm_symbols, fft_size]
            noise_var: Noise variance (power) per resource element
                Can be scalar or tensor with same shape as resource grid
                Related to Eb/No: σ² = N₀ = (Eb/No)^(-1) · R · E[|x|²]
                
        Returns:
            Tuple of:
            - y: Received signal in resource grid format
                Shape: [batch_size, num_rx, num_streams, num_ofdm_symbols, fft_size]
            - h: Channel frequency response
                Shape: [batch_size, num_rx, num_tx, num_streams, num_ofdm_symbols, fft_size]
                Used for perfect CSI simulations or channel analysis
        """
        y, h = self._ofdm_channel(x_rg, noise_var)
        return y, h
    
    def __call__(self, x_rg: tf.Tensor, noise_var: tf.Tensor) -> tuple:
        """
        Alias for call method for convenience.
        
        Allows the channel to be called as a function:
        y, h = channel(x_rg, noise_var)
        
        Args:
            x_rg: Input signal in resource grid format
            noise_var: Noise variance
            
        Returns:
            Tuple of (received signal, channel response)
        """
        return self.call(x_rg, noise_var)
    
    def get_channel_model(self):
        """
        Get the underlying channel model.
        
        Returns the 3GPP TR 38.901 channel model instance. Useful for
        accessing channel parameters, statistics, or advanced features.
        
        Returns:
            Channel model instance (UMi, UMa, or RMa)
        """
        return self._channel_model

