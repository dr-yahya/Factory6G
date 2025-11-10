"""
Antenna array configuration for 6G smart factory systems
"""

from sionna.phy.channel.tr38901 import AntennaArray
from .config import SystemConfig


class AntennaConfig:
    """Manages antenna array configurations for BS and UTs"""
    
    def __init__(self, config: SystemConfig):
        """
        Initialize antenna arrays for base station and user terminals.
        
        Args:
            config: System configuration parameters
        """
        self.config = config
        self.ut_array = self._create_ut_array()
        self.bs_array = self._create_bs_array()
    
    def _create_ut_array(self) -> AntennaArray:
        """
        Create user terminal antenna array.
        Typically single antenna with omni-directional pattern.
        """
        return AntennaArray(
            num_rows=1,
            num_cols=1,
            polarization="single",
            polarization_type="V",
            antenna_pattern="omni",
            carrier_frequency=self.config.carrier_frequency
        )
    
    def _create_bs_array(self) -> AntennaArray:
        """
        Create base station antenna array.
        Typically dual-polarized array with 3GPP 38.901 pattern.
        """
        return AntennaArray(
            num_rows=1,
            num_cols=int(self.config.num_bs_ant / 2),  # Dual polarization
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=self.config.carrier_frequency
        )
    
    def get_ut_array(self) -> AntennaArray:
        """Get user terminal antenna array"""
        return self.ut_array
    
    def get_bs_array(self) -> AntennaArray:
        """Get base station antenna array"""
        return self.bs_array

