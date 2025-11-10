"""
Transmitter components for 6G smart factory physical layer
"""

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.fec.ldpc import LDPC5GEncoder
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper
from .config import SystemConfig


class Transmitter(Block):
    """
    Transmitter chain: Binary Source -> LDPC Encoder -> QAM Mapper -> Resource Grid Mapper
    """
    
    def __init__(self, config: SystemConfig, resource_grid: ResourceGrid):
        """
        Initialize transmitter components.
        
        Args:
            config: System configuration parameters
            resource_grid: OFDM resource grid
        """
        super().__init__()
        self.config = config
        self.resource_grid = resource_grid
        
        # Calculate code parameters
        self._n = int(resource_grid.num_data_symbols * config.num_bits_per_symbol)  # Coded bits
        self._k = int(self._n * config.coderate)  # Information bits
        
        # Initialize components
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._k, self._n)
        self._mapper = Mapper("qam", config.num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(resource_grid)
    
    @property
    def num_info_bits(self) -> int:
        """Number of information bits (K)"""
        return self._k
    
    @property
    def num_coded_bits(self) -> int:
        """Number of coded bits (N)"""
        return self._n
    
    def call(self, batch_size: int) -> tuple:
        """
        Transmit signal through the transmitter chain.
        
        Args:
            batch_size: Batch size for simulation
            
        Returns:
            Tuple of (mapped symbols in resource grid format, original bits)
        """
        # Generate information bits
        b = self._binary_source([
            batch_size,
            self.config.num_tx,
            self.config.num_streams_per_tx,
            self._k
        ])
        
        # Encode bits
        c = self._encoder(b)
        
        # Map to QAM symbols
        x = self._mapper(c)
        
        # Map to resource grid
        x_rg = self._rg_mapper(x)
        
        return x_rg, b  # Return both resource grid and original bits
    
    def __call__(self, batch_size: int) -> tuple:
        """Alias for call method for convenience"""
        return self.call(batch_size)
    
    def encode_and_map(self, bits: tf.Tensor) -> tf.Tensor:
        """
        Encode and map bits to resource grid (for custom bit input).
        
        Args:
            bits: Input bits tensor
            
        Returns:
            Mapped symbols in resource grid format
        """
        c = self._encoder(bits)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)
        return x_rg

