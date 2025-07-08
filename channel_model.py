from sionna.phy.channel.tr38901 import UMi
from sionna.phy.channel import gen_single_sector_topology, OFDMChannel
from config import Config6G

class ChannelModel6G:
    """6G Channel Model Configuration"""
    
    def __init__(self, ut_array, bs_array, resource_grid):
        self.config = Config6G()
        self.ut_array = ut_array
        self.bs_array = bs_array
        self.resource_grid = resource_grid
        
        # Create channel model
        self.channel_model = UMi(
            carrier_frequency=self.config.CARRIER_FREQUENCY,
            o2i_model="low",
            ut_array=self.ut_array,
            bs_array=self.bs_array,
            direction=self.config.DIRECTION,
            enable_pathloss=False,
            enable_shadow_fading=False
        )
        
        # Create OFDM channel
        self.ofdm_channel = OFDMChannel(
            self.channel_model,
            self.resource_grid,
            add_awgn=True,
            normalize_channel=True
        )
    
    def set_topology(self, batch_size=None):
        """Generate and set channel topology"""
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
            
        topology = gen_single_sector_topology(
            batch_size,
            self.config.NUM_UT,
            self.config.SCENARIO
        )
        self.channel_model.set_topology(*topology)
    
    def apply(self, x_rg, no):
        """Apply channel to resource grid"""
        return self.ofdm_channel(x_rg, no)