from sionna.phy.channel.tr38901 import Antenna, AntennaArray
from config import Config6G

class AntennaConfig:
    """Configure antenna arrays for 6G system"""
    
    def __init__(self):
        self.config = Config6G()
    
    def create_ut_antenna(self):
        """Create user terminal antenna"""
        return Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.config.CARRIER_FREQUENCY
        )
    
    def create_bs_antenna(self):
        """Create base station antenna array for massive MIMO"""
        return AntennaArray(
            num_rows=self.config.NUM_BS_ANT_ROWS,
            num_cols=self.config.NUM_BS_ANT_COLS,
            polarization="dual",
            polarization_type="VH",
            antenna_pattern="38.901",
            carrier_frequency=self.config.CARRIER_FREQUENCY
        )