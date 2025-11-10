"""
Channel model components for 6G smart factory systems
"""

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.channel.tr38901 import UMi, UMa, RMa
from sionna.phy.channel import gen_single_sector_topology as gen_topology
from sionna.phy.channel import OFDMChannel
from sionna.phy.ofdm import ResourceGrid
from .config import SystemConfig
from .antenna import AntennaConfig


class ChannelModel(Block):
    """
    Channel model for 6G smart factory environments.
    Supports 3GPP TR 38.901 channel models (UMi, UMa, RMa).
    """
    
    def __init__(self, config: SystemConfig, antenna_config: AntennaConfig, 
                 resource_grid: ResourceGrid):
        """
        Initialize channel model.
        
        Args:
            config: System configuration parameters
            antenna_config: Antenna configuration
            resource_grid: OFDM resource grid
        """
        super().__init__()
        self.config = config
        self.antenna_config = antenna_config
        self.resource_grid = resource_grid
        
        # Create channel model based on scenario
        self._channel_model = self._create_channel_model()
        
        # Create OFDM channel
        self._ofdm_channel = OFDMChannel(
            self._channel_model,
            resource_grid,
            add_awgn=True,
            normalize_channel=True,
            return_channel=True
        )
    
    def _create_channel_model(self):
        """Create 3GPP TR 38.901 channel model based on scenario"""
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
        
        Args:
            batch_size: Batch size for topology generation
        """
        topology = gen_topology(
            batch_size,
            self.config.num_ut,
            self.config.scenario,
            min_ut_velocity=0.0,
            max_ut_velocity=0.0
        )
        self._channel_model.set_topology(*topology)
    
    def call(self, x_rg: tf.Tensor, noise_var: tf.Tensor) -> tuple:
        """
        Apply channel to input signal.
        
        Args:
            x_rg: Input signal in resource grid format
            noise_var: Noise variance
            
        Returns:
            Tuple of (received signal, channel response)
        """
        y, h = self._ofdm_channel(x_rg, noise_var)
        return y, h
    
    def __call__(self, x_rg: tf.Tensor, noise_var: tf.Tensor) -> tuple:
        """Alias for call method for convenience"""
        return self.call(x_rg, noise_var)
    
    def get_channel_model(self):
        """Get the underlying channel model"""
        return self._channel_model

