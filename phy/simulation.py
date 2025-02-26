# physical_layer/simulation.py
import sionna
from sionna.utils import sim_ber, ebnodb2no
from .channel import ChannelInterface
from .encoding import EncoderInterface, DecoderInterface
from .mapping import MapperDemapper
import tensorflow as tf

class Simulator:
    def __init__(self, channel: ChannelInterface, encoder: EncoderInterface, decoder: DecoderInterface, mapper_demapper: MapperDemapper, resource_grid, ebno_dbs, batch_size, max_mc_iter):
        self.channel = channel
        self.encoder = encoder
        self.decoder = decoder
        self.mapper_demapper = mapper_demapper
        self.resource_grid = resource_grid
        self.ebno_dbs = ebno_dbs
        self.batch_size = batch_size
        self.max_mc_iter = max_mc_iter

    def simulate(self):
        def mc_fun(batch_size, ebno_db):
            # Generate random information bits
            info_bits = tf.random.uniform((batch_size, self.encoder.k), maxval=2, dtype=tf.float32)

            # Encode
            encoded_bits = self.encoder.encode(info_bits)

            # Map
            symbols = self.mapper_demapper.map(encoded_bits)

            # Reshape symbols for OFDM channel
            symbols_reshaped = tf.reshape(symbols, [batch_size, self.channel.num_tx, self.channel.num_tx_ant, self.resource_grid.num_ofdm_symbols, self.resource_grid.fft_size])

            # Apply channel
            noisy_output, _ = self.channel.apply_channel(symbols_reshaped, ebnodb2no(ebno_db, self.mapper_demapper.num_bits_per_symbol, self.encoder.k / self.encoder.n, self.resource_grid))

            # Demap
            llrs = self.mapper_demapper.demap(noisy_output, ebnodb2no(ebno_db, self.mapper_demapper.num_bits_per_symbol, self.encoder.k / self.encoder.n, self.resource_grid))

            # Reshape LLRs for decoding: [batch_size, n]
            llrs_reshaped = tf.reshape(llrs[:, 0, 0, 0, :self.encoder.n], [batch_size, self.encoder.n])

            # Decode
            decoded_bits = self.decoder.decode(llrs_reshaped)

            return info_bits, decoded_bits

        return sim_ber(
            mc_fun=mc_fun,
            ebno_dbs=self.ebno_dbs,
            batch_size=self.batch_size,
            max_mc_iter=self.max_mc_iter,
            soft_estimates=False,
            num_target_bit_errors=1000,
            num_target_block_errors=100,
            target_ber=1e-6,
            target_bler=1e-5,
            early_stop=True,
            graph_mode="graph",
            distribute=None,
            verbose=True
        )