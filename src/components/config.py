"""
System configuration parameters for 6G smart factory physical layer
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SystemConfig:
    """Configuration parameters for the 6G smart factory system"""
    
    # RF Parameters
    carrier_frequency: float = 3.5e9  # Hz (3.5 GHz)
    fft_size: int = 128
    subcarrier_spacing: float = 30e3  # Hz (30 kHz)
    
    # OFDM Frame Structure
    num_ofdm_symbols: int = 14
    cyclic_prefix_length: int = 20
    pilot_ofdm_symbol_indices: List[int] = None
    
    # MIMO Configuration
    num_bs_ant: int = 8  # Base station antennas
    num_ut: int = 4  # Number of user terminals
    num_ut_ant: int = 1  # Antennas per user terminal
    
    # Modulation and Coding
    num_bits_per_symbol: int = 2  # QPSK
    coderate: float = 0.5
    
    # Channel Model
    scenario: str = "umi"  # UMi, UMa, RMa
    direction: str = "uplink"  # "uplink" or "downlink"
    o2i_model: str = "low"  # Outdoor-to-indoor model
    enable_pathloss: bool = False
    enable_shadow_fading: bool = False
    
    def __post_init__(self):
        """Initialize default values after dataclass initialization"""
        if self.pilot_ofdm_symbol_indices is None:
            self.pilot_ofdm_symbol_indices = [2, 11]
    
    @property
    def num_tx(self) -> int:
        """Number of transmitters (UTs for uplink)"""
        return self.num_ut
    
    @property
    def num_streams_per_tx(self) -> int:
        """Number of streams per transmitter"""
        return self.num_ut_ant
    
    def get_rx_tx_association(self) -> np.ndarray:
        """
        Create RX-TX association matrix.
        rx_tx_association[i,j]=1 means receiver i gets at least one stream from transmitter j.
        """
        bs_ut_association = np.zeros([1, self.num_ut])
        bs_ut_association[0, :] = 1
        return bs_ut_association

