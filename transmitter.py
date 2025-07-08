from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.fec.ldpc import LDPC5GEncoder
from sionna.phy.ofdm import ResourceGridMapper
from config import Config6G

class Transmitter6G:
    """6G Transmitter Chain"""
    
    def __init__(self, resource_grid):
        self.config = Config6G()
        self.resource_grid = resource_grid
        
        # Calculate coding parameters
        self.n = int(resource_grid.num_data_symbols * self.config.BITS_PER_SYMBOL)
        self.k = int(self.n * self.config.CODE_RATE)
        
        # Initialize components
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.mapper = Mapper("qam", self.config.BITS_PER_SYMBOL)
        self.rg_mapper = ResourceGridMapper(resource_grid)
    
    def transmit(self, batch_size=None):
        """Generate and transmit data"""
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        # Generate bits
        b = self.binary_source([
            batch_size,
            self.config.NUM_UT,
            self.config.NUM_UT_ANT,
            self.k
        ])
        
        # Encode
        c = self.encoder(b)
        
        # Map to symbols
        x = self.mapper(c)
        
        # Map to resource grid
        x_rg = self.rg_mapper(x)
        
        return b, x_rg