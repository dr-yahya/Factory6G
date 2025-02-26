# physical_layer/mapping.py
import sionna
from sionna.mapping import Constellation, Mapper, Demapper
import tensorflow as tf

class MapperDemapper:
    def __init__(self, num_bits_per_symbol, dtype=tf.complex64):
        self.num_bits_per_symbol = num_bits_per_symbol
        self.dtype = dtype
        self.constellation = Constellation(constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, dtype=dtype)
        self.mapper = Mapper(constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, dtype=dtype)
        self.demapper = Demapper(demapping_method="app", constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol, hard_out=False, dtype=dtype)

    def map(self, bits):
        return self.mapper(bits)

    def demap(self, symbols, noise_variance):
        return self.demapper((symbols, noise_variance))