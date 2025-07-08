from sionna.phy.ofdm import LSChannelEstimator, LMMSEEqualizer
from sionna.phy.mapping import Demapper
from sionna.phy.fec.ldpc import LDPC5GDecoder
from config import Config6G

class Receiver6G:
    """6G Receiver Chain"""
    
    def __init__(self, resource_grid, stream_management, encoder):
        self.config = Config6G()
        self.resource_grid = resource_grid
        self.stream_management = stream_management
        
        # Initialize components
        self.channel_estimator = LSChannelEstimator(
            resource_grid,
            interpolation_type="nn"
        )
        self.equalizer = LMMSEEqualizer(
            resource_grid,
            stream_management
        )
        self.demapper = Demapper(
            "app",
            "qam",
            self.config.BITS_PER_SYMBOL
        )
        self.decoder = LDPC5GDecoder(encoder, hard_out=True)
    
    def receive(self, y, no):
        """Process received signal"""
        # Channel estimation
        h_hat, err_var = self.channel_estimator(y, no)
        
        # Equalization
        x_hat, no_eff = self.equalizer(y, h_hat, err_var, no)
        
        # Demapping
        llr = self.demapper(x_hat, no_eff)
        
        # Decoding
        b_hat = self.decoder(llr)
        
        return b_hat