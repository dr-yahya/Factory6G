"""
Transmitter components for 6G smart factory physical layer.

This module implements the transmitter chain for OFDM-MIMO systems, including
binary source generation, LDPC channel coding, QAM modulation, and resource
grid mapping. The transmitter follows 5G/6G standards for channel coding and
modulation schemes.

Theory:
    The transmitter chain performs the following operations:
    
    1. Binary Source:
       - Generates random information bits b ∈ {0, 1}
       - Typically uniformly distributed for capacity-achieving codes
       - Shape: [batch_size, num_tx, num_streams, num_info_bits]
       
    2. Channel Coding (LDPC):
       - Encodes information bits to add redundancy for error correction
       - 5G LDPC codes: Quasi-cyclic (QC) LDPC codes with structured design
       - Code rate R = k/n, where k = info bits, n = coded bits
       - Encoding: c = G * b (mod 2), where G is generator matrix
       - LDPC codes have sparse parity-check matrix H: H * c^T = 0 (mod 2)
       - Near-Shannon-limit performance with iterative decoding
       
    3. Modulation (QAM):
       - Maps coded bits to complex modulation symbols
       - QPSK: 2 bits → 1 symbol, 4 constellation points
       - 16-QAM: 4 bits → 1 symbol, 16 constellation points
       - 64-QAM: 6 bits → 1 symbol, 64 constellation points
       - Gray coding: Adjacent symbols differ by 1 bit (minimizes BER)
       - Constellation: x = (2m - M - 1) + j(2n - M - 1), where M = √M_total
       
    4. Resource Grid Mapping:
       - Maps modulation symbols to OFDM resource grid
       - Resource grid: 2D array [OFDM symbols × subcarriers]
       - Pilot symbols: Known reference signals for channel estimation
       - Data symbols: User data mapped to available resource elements
       - Null subcarriers: Guard bands and DC subcarrier
       - Pattern: Determined by pilot pattern (e.g., Kronecker pattern)
       
    5. OFDM Modulation (performed by channel model):
       - IFFT: Converts frequency-domain symbols to time-domain
       - Cyclic prefix: Copies tail to beginning to prevent ISI
       - Parallel-to-serial: Converts to time-domain waveform
       
    Mathematical Formulation:
        b → [LDPC Encoder] → c → [QAM Mapper] → x → [Resource Grid] → x_rg
        
        where:
        - b: Information bits [batch, num_tx, num_streams, k]
        - c: Coded bits [batch, num_tx, num_streams, n]
        - x: Modulation symbols [batch, num_tx, num_streams, num_symbols]
        - x_rg: Resource grid [batch, num_tx, num_streams, num_ofdm_sym, fft_size]
        
    Resource Management:
        The transmitter supports dynamic resource allocation:
        - Active UT mask: Enable/disable specific UTs (scheduling)
        - Per-UT power scaling: Adjust transmit power per UT
        - Applied before transmission: x_rg_scaled = mask * sqrt(power) * x_rg

References:
    - 3GPP TS 38.211: Physical channels and modulation
    - 3GPP TS 38.212: Multiplexing and channel coding (LDPC)
    - Richardson & Urbanke, "Modern Coding Theory" (LDPC codes)
    - Proakis & Salehi, "Digital Communications" (Modulation)
"""

import tensorflow as tf
from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.fec.ldpc import LDPC5GEncoder
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper
from .config import SystemConfig
import tensorflow as tf_ops


class Transmitter:
    """
    Transmitter chain: Binary Source -> LDPC Encoder -> QAM Mapper -> Resource Grid Mapper.
    
    This class implements the complete transmitter processing chain for OFDM-MIMO
    systems. It generates information bits, applies channel coding (LDPC), maps
    bits to modulation symbols (QAM), and maps symbols to the OFDM resource grid.
    
    Theory:
        The transmitter performs the following signal processing steps:
        
        1. Binary Source Generation:
           - Generates uniformly distributed random bits
           - Represents information to be transmitted
           
        2. LDPC Channel Coding:
           - Adds redundancy to enable error correction at receiver
           - 5G LDPC codes use quasi-cyclic structure for efficient encoding
           - Code rate R = k/n determines coding overhead
           
        3. QAM Modulation:
           - Maps coded bits to complex modulation symbols
           - Constellation size: M = 2^num_bits_per_symbol
           - Gray coding minimizes bit errors
           
        4. Resource Grid Mapping:
           - Maps symbols to time-frequency resource grid
           - Allocates pilots for channel estimation
           - Handles null subcarriers (DC, guard bands)
           
    The transmitter supports resource management features:
    - Dynamic scheduling (active UT mask)
    - Power control (per-UT power scaling)
    """
    
    def __init__(self, config: SystemConfig, resource_grid: ResourceGrid):
        """
        Initialize transmitter components.
        
        Creates and configures all transmitter chain components including
        binary source, LDPC encoder, QAM mapper, and resource grid mapper.
        Code parameters are calculated based on the resource grid dimensions
        and code rate.
        
        Theory:
            Code parameters calculation:
            - Number of data symbols: num_data_symbols = total REs - pilot REs - null REs
            - Coded bits: n = num_data_symbols × num_bits_per_symbol
            - Information bits: k = n × coderate (rounded to valid LDPC block length)
            
            The LDPC encoder uses 5G base graph selection:
            - Base graph 1: For larger blocks (k > 3840) or high code rates
            - Base graph 2: For smaller blocks (k ≤ 3840) or low code rates
            
        Args:
            config: System configuration parameters containing modulation order,
                code rate, and other transmitter settings.
            resource_grid: OFDM resource grid defining the time-frequency
                structure, including pilot pattern and null subcarriers.
        """
        super().__init__()
        self.config = config
        self.resource_grid = resource_grid
        
        # Calculate code parameters
        # n = number of coded bits = data symbols × bits per symbol
        self._n = int(resource_grid.num_data_symbols * config.num_bits_per_symbol)
        # k = number of information bits = n × code rate
        self._k = int(self._n * config.coderate)
        
        # Initialize components
        self._binary_source = BinarySource()  # Random bit generator
        self._encoder = LDPC5GEncoder(self._k, self._n)  # 5G LDPC encoder
        self._mapper = Mapper("qam", config.num_bits_per_symbol)  # QAM modulator
        self._rg_mapper = ResourceGridMapper(resource_grid)  # Resource grid mapper
    
    @property
    def num_info_bits(self) -> int:
        """
        Number of information bits (K) per transmission block.
        
        This is the number of uncoded information bits that can be transmitted
        in one resource grid. The actual value depends on:
        - Resource grid size (number of data symbols)
        - Modulation order (bits per symbol)
        - Code rate (k = n × R)
        
        Returns:
            Number of information bits per block
        """
        return self._k
    
    @property
    def num_coded_bits(self) -> int:
        """
        Number of coded bits (N) per transmission block.
        
        This is the total number of bits after channel coding, which includes
        both information bits and parity bits. The code rate R = k/n determines
        the ratio of information to coded bits.
        
        Returns:
            Number of coded bits per block
        """
        return self._n
    
    def call(self, batch_size: int) -> tuple:
        """
        Transmit signal through the transmitter chain.
        
        Processes information bits through the complete transmitter chain:
        1. Generate random information bits
        2. Encode with LDPC encoder
        3. Map to QAM symbols
        4. Map to resource grid
        5. Apply resource management (scheduling, power control)
        
        Theory:
            The transmission process can be mathematically described as:
            
            b ~ Uniform({0,1})  # Information bits
            c = Encoder(b)      # LDPC encoding: c = G·b (mod 2)
            x = Mapper(c)       # QAM mapping: x ∈ {constellation points}
            x_rg = RG_Mapper(x) # Resource grid mapping
            x_tx = Mask · √P · x_rg  # Resource management
            
            where:
            - b: Information bits [batch, num_tx, num_streams, k]
            - c: Coded bits [batch, num_tx, num_streams, n]
            - x: Modulation symbols [batch, num_tx, num_streams, num_symbols]
            - x_rg: Resource grid [batch, num_tx, num_streams, num_ofdm_sym, fft_size]
            - Mask: Active UT mask (0 or 1)
            - P: Per-UT power scaling factor
            
            Resource Management:
            - Active UT mask: x_rg[masked_UT] = 0 (scheduling)
            - Power scaling: x_rg[UT_i] = √P_i · x_rg[UT_i] (power control)
            - Power is applied in linear scale: P_linear = (P_dB/10)^10
            
        Args:
            batch_size: Number of channel realizations to process in parallel.
                Larger batch sizes improve GPU utilization but require more memory.
                
        Returns:
            Tuple of:
            - x_rg: Transmitted symbols in resource grid format
                Shape: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
            - b: Original information bits (for BER/BLER calculation)
                Shape: [batch_size, num_tx, num_streams_per_tx, num_info_bits]
        """
        # Generate information bits
        # Shape: [batch_size, num_tx, num_streams_per_tx, num_info_bits]
        b = self._binary_source([
            batch_size,
            self.config.num_tx,
            self.config.num_streams_per_tx,
            self._k
        ])
        
        # Encode bits with LDPC encoder
        # Shape: [batch_size, num_tx, num_streams_per_tx, num_coded_bits]
        c = self._encoder(b)
        
        # Map coded bits to QAM symbols
        # Shape: [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
        x = self._mapper(c)
        
        # Map symbols to resource grid (time-frequency allocation)
        # Shape: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        x_rg = self._rg_mapper(x)
        
        # Resource management hooks:
        # - Apply per-UT activation mask (scheduling)
        # - Apply per-UT power scaling (power control)
        # Shapes:
        #   x_rg: [batch, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        # Build masks with broadcasting-friendly shapes for element-wise multiplication
        if self.config.active_ut_mask is not None:
            # Active UT mask: 1 = scheduled, 0 = muted
            # Broadcasting: [1, num_tx, 1, 1, 1] allows element-wise multiplication
            ut_mask = tf_ops.constant(self.config.active_ut_mask, dtype=x_rg.dtype)  # [num_tx]
            ut_mask = tf_ops.reshape(ut_mask, [1, self.config.num_tx, 1, 1, 1])
            x_rg = x_rg * ut_mask
        if self.config.per_ut_power is not None:
            # Per-UT power scaling: applied in linear scale
            # Power scaling: x_scaled = x * sqrt(P), where P is linear power factor
            # This maintains the same average power per symbol while scaling amplitude
            ut_power = tf_ops.constant(self.config.per_ut_power, dtype=x_rg.dtype)  # [num_tx]
            ut_power = tf_ops.reshape(ut_power, [1, self.config.num_tx, 1, 1, 1])
            x_rg = x_rg * tf_ops.sqrt(ut_power)
        
        return x_rg, b  # Return both resource grid and original bits
    
    def __call__(self, batch_size: int) -> tuple:
        """
        Alias for call method for convenience.
        
        Allows the transmitter to be called as a function:
        x_rg, b = transmitter(batch_size)
        
        Args:
            batch_size: Number of channel realizations
            
        Returns:
            Tuple of (resource grid, information bits)
        """
        return self.call(batch_size)
    
    def encode_and_map(self, bits: tf.Tensor) -> tf.Tensor:
        """
        Encode and map custom input bits to resource grid.
        
        This method allows using custom bit sequences instead of random bits.
        Useful for testing with known bit patterns or when bits come from
        higher layers.
        
        Theory:
            The encoding and mapping process is:
            c = Encoder(bits)  # LDPC encoding
            x = Mapper(c)      # QAM modulation
            x_rg = RG_Mapper(x) # Resource grid mapping
            
            Note: Resource management (masking, power control) is NOT applied
            in this method. Use call() for complete transmission with resource
            management.
        
        Args:
            bits: Input information bits tensor
                Shape: [batch_size, num_tx, num_streams_per_tx, num_info_bits]
                Must match the expected number of information bits (self._k)
                
        Returns:
            Mapped symbols in resource grid format
            Shape: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]
        """
        c = self._encoder(bits)  # LDPC encoding
        x = self._mapper(c)      # QAM modulation
        x_rg = self._rg_mapper(x)  # Resource grid mapping
        return x_rg

