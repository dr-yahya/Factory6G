# physical_layer/resource_grid.py
import sionna
from sionna.ofdm import ResourceGrid
import tensorflow as tf

'''
TODO 
3. Resource grid; replace OFDM with [RSCMA], or NOMA
4. Siona what resource allocation they use? is it simple (give best channel to best user) [greedy, or water filling]
'''

class ResourceGridConfig:
    def __init__(self, num_ofdm_symbols, fft_size, subcarrier_spacing, num_tx, num_tx_ant, dtype=tf.complex64):
        self.num_ofdm_symbols = num_ofdm_symbols
        self.fft_size = fft_size
        self.subcarrier_spacing = subcarrier_spacing
        self.num_tx = num_tx
        self.num_tx_ant = num_tx_ant
        self.dtype = dtype

        self.resource_grid = self._create_resource_grid()

    def _create_resource_grid(self):
        return ResourceGrid(
            num_ofdm_symbols=self.num_ofdm_symbols,
            fft_size=self.fft_size,
            subcarrier_spacing=self.subcarrier_spacing,
            num_tx=self.num_tx,
            num_streams_per_tx=self.num_tx_ant,
            cyclic_prefix_length=0,
            num_guard_carriers=(0, 0),
            dc_null=False,
            pilot_pattern="empty",
            pilot_ofdm_symbol_indices=None,
            dtype=self.dtype
        )

    def get_resource_grid(self):
        return self.resource_grid