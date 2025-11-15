"""
Main 6G smart factory physical layer model using component-based architecture.

This module implements the complete end-to-end physical layer system model,
composing all components (transmitter, channel, receiver) into a unified
system for simulation and evaluation. The model supports various channel
scenarios, channel estimators, and resource management strategies.

Theory:
    End-to-End System Model:
    
    The complete transmission chain can be described as:
    
    1. Transmitter:
       b → [LDPC Encoder] → c → [QAM Mapper] → x → [Resource Grid] → x_rg
       
    2. Channel:
       x_rg → [OFDM Channel] → y = H·x_rg + n
       
    3. Receiver:
       y → [Channel Estimation] → Ĥ, σ²_ε
       y, Ĥ, σ²_ε → [Equalization] → x̂, σ²_eff
       x̂, σ²_eff → [Demapping] → LLR
       LLR → [LDPC Decoder] → b̂
       
    Performance Metrics:
    - Bit Error Rate (BER): P(b̂ ≠ b) = E[I(b̂ ≠ b)]
    - Block Error Rate (BLER): P(∃ i: b̂[i] ≠ b[i]) = 1 - (1 - BER)^n
    - Throughput: R = R_code · log2(M) · (1 - BLER) bits/s/Hz
    - Spectral Efficiency: η = R / B Hz^-1
    
    System Capacity:
    - Shannon capacity: C = log2(1 + SNR) bits/s/Hz
    - MIMO capacity: C = log2(det(I + (ρ/N_tx)·H·H^H)) bits/s/Hz
    - With imperfect CSI: C_imperfect < C_perfect (performance gap)
    
    Resource Management:
    - Scheduling: Select which UTs to serve (active_ut_mask)
    - Power Control: Adjust transmit power per UT (per_ut_power)
    - Link Adaptation: Adjust MCS based on channel conditions
    - Pilot Reuse: Manage pilot contamination in multi-cell systems

References:
    - 3GPP TS 38.211, 38.212: Physical layer specifications
    - Tse & Viswanath, "Fundamentals of Wireless Communication"
    - Proakis & Salehi, "Digital Communications"
"""

import tensorflow as tf
import numpy as np
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.utils import ebnodb2no

from ..components.config import SystemConfig
from ..components.antenna import AntennaConfig
from ..components.transmitter import Transmitter
from ..components.channel import ChannelModel
from ..components.receiver import Receiver
from ..components.estimators import NeuralChannelEstimator, SmoothedLSEstimator, TemporalEstimator
from .resource_manager import ResourceManager, StaticResourceManager, ResourceDirectives


class Model:
    """
    Complete 6G smart factory physical layer model.
    
    This model composes transmitter, channel, and receiver components
    to simulate OFDM MIMO transmissions over 3GPP TR 38.901 channel models.
    The model supports various configurations, channel estimators, and resource
    management strategies for comprehensive system evaluation.
    
    Theory:
        The model implements a complete communication system:
        
        Transmitter Chain:
        - Binary source → LDPC encoder → QAM mapper → Resource grid mapper
        - Supports dynamic scheduling and power control
        
        Channel:
        - 3GPP TR 38.901 channel models (UMi, UMa, RMa)
        - OFDM channel with AWGN noise
        - MIMO spatial multiplexing
        
        Receiver Chain:
        - Channel estimation (LS, neural, smoothed, temporal)
        - LMMSE equalization
        - APP demapping
        - LDPC decoding
        
        The model can be used for:
        - BER/BLER simulation
        - Performance evaluation
        - System optimization
        - Resource management studies
    """
    
    def __init__(
        self,
        scenario: str = "umi",
        perfect_csi: bool = False,
        config: SystemConfig | None = None,
        estimator_type: str = "ls",
        estimator_weights: str | None = None,
        estimator_kwargs: dict | None = None,
        resource_manager: ResourceManager | None = None,
    ):
        """
        Initialize the complete system model.
        
        Args:
            scenario: Channel scenario ("umi", "uma", "rma")
            perfect_csi: Whether to use perfect channel state information
            config: Optional custom system configuration. If None, uses defaults.
        """
        super().__init__()
        
        # Initialize configuration
        if config is None:
            self.config = SystemConfig(scenario=scenario)
        else:
            self.config = config
            self.config.scenario = scenario
        
        self.perfect_csi = perfect_csi
        self.estimator_type = estimator_type
        self._resource_manager = resource_manager
        if self._resource_manager is not None:
            # Allow resource manager to mutate config before building components
            self._resource_manager.apply_pre_build(self.config)
        
        # Setup resource grid
        rx_tx_association = self.config.get_rx_tx_association()
        self._rg = ResourceGrid(
            num_ofdm_symbols=self.config.num_ofdm_symbols,
            fft_size=self.config.fft_size,
            subcarrier_spacing=self.config.subcarrier_spacing,
            num_tx=self.config.num_tx,
            num_streams_per_tx=self.config.num_streams_per_tx,
            cyclic_prefix_length=self.config.cyclic_prefix_length,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=self.config.pilot_ofdm_symbol_indices
        )
        
        # Setup stream management
        self._sm = StreamManagement(rx_tx_association, self.config.num_streams_per_tx)
        
        # Initialize components
        self._antenna_config = AntennaConfig(self.config)
        self._transmitter = Transmitter(self.config, self._rg)
        self._channel = ChannelModel(self.config, self._antenna_config, self._rg)

        # Prepare optional channel estimator
        channel_estimator = None
        if not perfect_csi:
            estimator_kwargs = estimator_kwargs or {}
            et = estimator_type.lower()
            if et in ("ls", "ls_nn", "ls-nn"):
                # Use default LS with 'nn' interpolation inside Receiver by passing None,
                # or explicitly construct with 'nn' to be explicit
                from sionna.phy.ofdm import LSChannelEstimator
                channel_estimator = LSChannelEstimator(self._rg, interpolation_type="nn")
            elif et in ("ls_lin", "ls-lin", "ls_linear"):
                from sionna.phy.ofdm import LSChannelEstimator
                channel_estimator = LSChannelEstimator(self._rg, interpolation_type="lin")
            elif et == "neural":
                channel_estimator = NeuralChannelEstimator(
                    self.config,
                    self._rg,
                    weights_path=estimator_weights,
                    **estimator_kwargs,
                )
            elif et == "ls_smooth":
                channel_estimator = SmoothedLSEstimator(
                    self.config,
                    self._rg,
                    **estimator_kwargs,
                )
            elif et == "ls_temporal":
                channel_estimator = TemporalEstimator(
                    self.config,
                    self._rg,
                    **estimator_kwargs,
                )
            else:
                raise ValueError(
                    f"Unsupported estimator_type '{estimator_type}'. "
                    "Supported: 'ls', 'ls_nn', 'ls_lin', 'neural', 'ls_smooth', 'ls_temporal'."
                )

        # Receiver needs encoder reference for LDPC decoder
        encoder = self._transmitter._encoder
        self._receiver = Receiver(
            self.config,
            self._rg,
            self._sm,
            encoder,
            perfect_csi=perfect_csi,
            channel_estimator=channel_estimator,
        )
    
    def new_topology(self, batch_size: int):
        """
        Generate and set new topology for the channel.
        
        Args:
            batch_size: Batch size for topology generation
        """
        self._channel.set_topology(batch_size)
    
    @tf.function
    def call(self, batch_size: int, ebno_db: float) -> tuple:
        """
        Simulate transmission through the complete system.
        
        Args:
            batch_size: Batch size for simulation
            ebno_db: Energy per bit to noise ratio in dB
            
        Returns:
            Tuple of (transmitted bits, received bits)
        """
        # Generate new topology
        self.new_topology(batch_size)
        
        # Query resource manager for per-batch directives
        if self._resource_manager is not None:
            directives: ResourceDirectives = self._resource_manager.get_runtime_directives(
                self.config, ebno_db, feedback=None
            )
            # Apply runtime directives to config for this batch
            if directives.active_ut_mask is not None:
                self.config.active_ut_mask = list(directives.active_ut_mask)
            if directives.per_ut_power is not None:
                self.config.per_ut_power = list(directives.per_ut_power)
            if directives.pilot_reuse_factor is not None:
                self.config.pilot_reuse_factor = int(directives.pilot_reuse_factor)
        
        # Calculate noise variance
        no = ebnodb2no(
            ebno_db,
            self.config.num_bits_per_symbol,
            self.config.coderate,
            self._rg
        )
        
        # Transmitter: Generate bits and map to resource grid
        x_rg, b = self._transmitter.call(batch_size)
        
        # Channel: Apply channel and noise
        y, h = self._channel(x_rg, no)
        
        # Receiver: Estimate channel, equalize, demap, and decode
        if self.perfect_csi:
            b_hat = self._receiver.process_with_perfect_csi(y, h, no)
        else:
            h_hat, err_var = self._receiver.estimate_channel(y, no)
            b_hat = self._receiver(y, h_hat, err_var, no)
        
        return b, b_hat
    
    def get_config(self) -> SystemConfig:
        """Get system configuration"""
        return self.config
    
    def get_transmitter(self) -> Transmitter:
        """Get transmitter component"""
        return self._transmitter
    
    def get_channel(self) -> ChannelModel:
        """Get channel model component"""
        return self._channel
    
    def get_receiver(self) -> Receiver:
        """Get receiver component"""
        return self._receiver

    def __call__(self, batch_size: int, ebno_db: float) -> tuple:
        """Alias to support Sionna sim_ber(mc_fun, ...) expectations."""
        return self.call(batch_size, ebno_db)
    
    def run_batch(self, batch_size: int, ebno_db: float, include_details: bool = True) -> dict:
        """
        Run a batch simulation and return detailed results.
        
        Args:
            batch_size: Batch size for simulation
            ebno_db: Energy per bit to noise ratio in dB
            include_details: If True, return detailed metrics for analysis
            
        Returns:
            Dictionary with simulation results including:
            - bits: Transmitted bits [batch, num_tx, num_streams, num_bits]
            - bits_hat: Received/decoded bits [batch, num_tx, num_streams, num_bits]
            - decoder_iterations: Number of LDPC decoder iterations [batch, num_tx, num_streams]
            - channel: True channel [batch, num_rx, num_tx, num_streams, num_ofdm, fft_size]
            - channel_hat: Estimated channel [batch, num_rx, num_tx, num_streams, num_ofdm, fft_size]
            - qam: Transmitted QAM symbols [batch, num_tx, num_streams, num_symbols]
            - qam_hat: Equalized QAM symbols [batch, num_tx, num_streams, num_symbols]
            - no_eff: Effective noise variance [batch, num_tx, num_streams, num_symbols]
        """
        import tensorflow as tf
        import numpy as np
        
        # Generate new topology
        self.new_topology(batch_size)
        
        # Query resource manager for per-batch directives
        if self._resource_manager is not None:
            from ..models.resource_manager import ResourceDirectives
            directives: ResourceDirectives = self._resource_manager.get_runtime_directives(
                self.config, ebno_db, feedback=None
            )
            if directives.active_ut_mask is not None:
                self.config.active_ut_mask = list(directives.active_ut_mask)
            if directives.per_ut_power is not None:
                self.config.per_ut_power = list(directives.per_ut_power)
            if directives.pilot_reuse_factor is not None:
                self.config.pilot_reuse_factor = int(directives.pilot_reuse_factor)
        
        # Calculate noise variance
        no = ebnodb2no(
            ebno_db,
            self.config.num_bits_per_symbol,
            self.config.coderate,
            self._rg
        )
        
        # Transmitter: Generate bits and map to resource grid
        x_rg, b, x_qam = self._transmitter.call(batch_size)
        
        # Channel: Apply channel and noise
        y, h = self._channel(x_rg, no)
        
        # Receiver processing
        if self.perfect_csi:
            b_hat = self._receiver.process_with_perfect_csi(y, h, no)
            h_hat = h
            err_var = tf.zeros_like(h)
        else:
            h_hat, err_var = self._receiver.estimate_channel(y, no)
            b_hat = self._receiver(y, h_hat, err_var, no)
        
        if include_details:
            import time
            
            # Measure latency: start timing
            t_start = time.time()
            
            # Get detailed information for metrics
            # Equalize to get QAM symbols and effective noise
            x_hat, no_eff = self._receiver.equalize(y, h_hat, err_var, no)
            
            # Get decoder iterations - decode again to get iterations
            llr = self._receiver.demap(x_hat, no_eff)
            _, decoder_iter = self._receiver.decode(llr)
            
            # Measure latency: end timing (encoding + transmission + decoding)
            # Note: Encoding time is negligible, transmission time is OFDM symbol duration
            # Decoding time is proportional to iterations
            t_end = time.time()
            latency_sec = t_end - t_start
            
            # Estimate OFDM symbol transmission time
            subcarrier_spacing = self.config.subcarrier_spacing  # Hz
            symbol_duration = 1.0 / subcarrier_spacing  # seconds
            cyclic_prefix_ratio = self.config.cyclic_prefix_length / self.config.fft_size
            total_symbol_duration = symbol_duration * (1 + cyclic_prefix_ratio)
            frame_transmission_time = total_symbol_duration * self.config.num_ofdm_symbols
            
            # Add frame transmission time to latency
            latency_sec += frame_transmission_time
            
            # Estimate energy consumption (physical layer)
            # Energy = Power × Time
            # Power estimates (typical values for 6G systems):
            # - Encoding: ~10 mW per Mbps
            # - RF Transmission: ~100-500 mW (depends on power control)
            # - RF Reception: ~50-200 mW
            # - Decoding: ~50 mW per Mbps (depends on iterations)
            
            # Estimate throughput for energy calculation
            # Get num_info_bits from transmitter
            num_info_bits = self._transmitter.num_info_bits
            
            # Encoding energy (baseband processing)
            encoding_power_watts = 10e-3 * (num_info_bits / latency_sec) / 1e6  # 10 mW per Mbps
            encoding_energy = encoding_power_watts * latency_sec * 0.1  # 10% of latency is encoding
            
            # RF transmission energy
            tx_power_watts = 0.2  # 200 mW typical for smart factory devices
            tx_energy = tx_power_watts * frame_transmission_time
            
            # RF reception energy
            rx_power_watts = 0.1  # 100 mW typical
            rx_energy = rx_power_watts * frame_transmission_time
            
            # Decoding energy (proportional to iterations)
            avg_iterations = float(tf.reduce_mean(decoder_iter))
            decoding_power_watts = 50e-3 * (num_info_bits / latency_sec) / 1e6 * (1 + avg_iterations / 10)
            decoding_energy = decoding_power_watts * latency_sec * 0.3  # 30% of latency is decoding
            
            total_energy_joules = encoding_energy + tx_energy + rx_energy + decoding_energy
            
            # Get noise power for SNR calculation (without interference)
            # For SNR, we use the pure noise variance (no interference)
            # Convert to numpy scalar if tensor
            if hasattr(no, 'numpy'):
                noise_power = float(no.numpy()) if no.shape == () else no.numpy()
            else:
                noise_power = float(no) if np.isscalar(no) else np.array(no)
            # Ensure it's a scalar for SNR calculation
            if isinstance(noise_power, np.ndarray) and noise_power.size == 1:
                noise_power = float(noise_power.item())
            elif isinstance(noise_power, np.ndarray):
                # If it's an array, use mean for SNR calculation
                noise_power = float(np.mean(noise_power))
            
            # Use QAM symbols from transmitter (x_qam) - these are the transmitted symbols
            # Convert to numpy for metrics accumulator
            result = {
                "bits": b.numpy(),
                "bits_hat": b_hat.numpy(),
                "decoder_iterations": decoder_iter.numpy(),
                "channel": h.numpy(),
                "channel_hat": h_hat.numpy(),
                "qam": x_qam.numpy(),
                "qam_hat": x_hat.numpy(),
                "no_eff": no_eff.numpy(),
                "noise_power": noise_power,
                "latency_sec": latency_sec,
                "energy_joules": total_energy_joules,
            }
        else:
            result = {
                "bits": b.numpy(),
                "bits_hat": b_hat.numpy(),
            }
        
        return result
