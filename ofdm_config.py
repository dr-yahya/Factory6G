import numpy as np
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.mimo import StreamManagement
from config import Config6G

class OFDMConfig:
    """OFDM and Stream Management Configuration"""
    
    def __init__(self):
        self.config = Config6G()
        self._setup_stream_management()
        self._create_resource_grid()
    
    def _setup_stream_management(self):
        """Setup RX-TX association and stream management"""
        self.rx_tx_association = np.ones([1, self.config.NUM_UT])
        self.stream_management = StreamManagement(
            self.rx_tx_association,
            self.config.NUM_UT_ANT
        )
    
    def _create_resource_grid(self):
        """Create 6G OFDM resource grid"""
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=self.config.NUM_OFDM_SYMBOLS,
            fft_size=self.config.FFT_SIZE,
            subcarrier_spacing=self.config.SUBCARRIER_SPACING,
            num_tx=self.config.NUM_UT,
            num_streams_per_tx=self.config.NUM_UT_ANT,
            cyclic_prefix_length=self.config.CYCLIC_PREFIX_LENGTH,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=self.config.PILOT_SYMBOL_INDICES
        )
    
    @property
    def rg(self):
        return self.resource_grid
    
    @property
    def sm(self):
        return self.stream_management