import sionna
from sionna.channel import OFDMChannel, AWGN, RayleighBlockFading, SpatialCorrelation, KroneckerModel
from sionna.utils import ebnodb2no, hard_decisions, complex_normal, sim_ber
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.ofdm import ResourceGrid
from sionna.utils.plotting import PlotBER, plot_ber
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# Abstract Base Classes
class ChannelInterface(ABC):
    @abstractmethod
    def apply_channel(self, symbols, noise_variance):
        pass

    @abstractmethod
    def get_channel_response(self):
        pass

class DecoderInterface(ABC):
    @abstractmethod
    def decode(self, llrs):
        pass

# Channel Model
class ChannelModel(ChannelInterface):
    def __init__(self, num_tx, num_rx, num_tx_ant, num_rx_ant, resource_grid, correlation_distance):
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        self.resource_grid = resource_grid
        self.correlation_distance = correlation_distance

        # Initialize channel components
        self.rayleigh_channel = RayleighBlockFading(
            num_rx=num_rx, num_rx_ant=num_rx_ant, num_tx=num_tx, num_tx_ant=num_tx_ant, dtype=tf.complex64
        )
        self.spatial_corr = self._create_spatial_correlation()
        self.ofdm_channel = OFDMChannel(
            channel_model=self.rayleigh_channel,
            resource_grid=resource_grid,
            add_awgn=False,
            normalize_channel=False,
            return_channel=True,
            dtype=tf.complex64
        )
        self.awgn = AWGN(dtype=tf.complex64)

    def _create_spatial_correlation(self):
        def manual_exp_corr_mat(size, rho):
            indices = tf.range(size, dtype=tf.float32)
            i, j = tf.meshgrid(indices, indices)
            distances = tf.abs(i - j)
            corr_matrix = tf.pow(rho, distances)
            return tf.cast(corr_matrix, tf.complex64)

        r_tx = manual_exp_corr_mat(self.num_tx_ant, self.correlation_distance)
        r_rx = manual_exp_corr_mat(self.num_rx_ant, self.correlation_distance)
        return KroneckerModel(r_tx=r_tx, r_rx=r_rx)

    def apply_channel(self, symbols, noise_variance):
        channel_output, h_freq = self.ofdm_channel(symbols)
        noise = complex_normal(tf.shape(channel_output), var=noise_variance, dtype=tf.complex64)
        noisy_output = channel_output + noise
        return noisy_output, h_freq

    def get_channel_response(self):
        return self.rayleigh_channel(), self.spatial_corr

# Resource Grid Configuration
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

# Encoder Class
class Encoder:
    def __init__(self, k, n, num_bits_per_symbol, dtype=tf.float32):
        self.k = k
        self.n = n
        self.num_bits_per_symbol = num_bits_per_symbol
        self.dtype = dtype
        self.encoder = LDPC5GEncoder(k=k, n=n, num_bits_per_symbol=num_bits_per_symbol, dtype=dtype)

    def encode(self, info_bits):
        return self.encoder(info_bits)

# Decoder Class
class LDPC5GDecoderWrapper(DecoderInterface):
    def __init__(self, encoder):
        if not isinstance(encoder, LDPC5GEncoder):
            raise AssertionError("Encoder must be an instance of LDPC5GEncoder.")

        self.decoder = LDPC5GDecoder(
            encoder=encoder,
            trainable=False,
            cn_type='boxplus-phi',
            hard_out=True,
            track_exit=False,
            return_infobits=True,
            prune_pcm=True,
            num_iter=20,
            stateful=False,
            output_dtype=tf.float32
        )

    def decode(self, llrs):
        return self.decoder(llrs)

# Mapper/Demapper Class
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

# Simulator Class
class Simulator:
    def __init__(self, channel, encoder, decoder, mapper_demapper, resource_grid, ebno_dbs, batch_size, max_mc_iter):
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
            # Ensure we only take n LLRs (codeword length)
            llrs_sliced = llrs[:, 0, 0, 0, :self.encoder.n]  # Take first tx, stream, symbol, and truncate to n
            llrs_reshaped = tf.reshape(llrs_sliced, [batch_size, self.encoder.n])

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

# Plotter Class
class Plotter:
    def __init__(self, title="Bit/Block Error Rate for 6G Smart Factory"):
        self.plotter = PlotBER(title=title)

    def plot_results(self, ebno_dbs, ber, bler, legend="6G Simulation"):
        self.plotter.simulate(
            mc_fun=lambda batch_size, ebno_db: self.simulate_step(batch_size, ebno_db),
            ebno_dbs=ebno_dbs,
            batch_size=100,
            max_mc_iter=1000,
            legend=legend,
            add_ber=True,
            add_bler=True,
            show_fig=True,
            verbose=True
        )

    def simulate_step(self, batch_size, ebno_db):
        # This is a placeholder; actual simulation should be handled by Simulator
        info_bits, decoded_bits = Simulator(None, None, None, None, None, None, batch_size, None).simulate()
        return info_bits, decoded_bits

# Main Execution
def main():
    # Configuration
    num_ofdm_symbols = 14
    fft_size = 512
    subcarrier_spacing = 60e3
    num_tx = 1
    num_rx = 1
    num_tx_ant = 64
    num_rx_ant = 32
    correlation_distance = 0.5
    k = 100  # Number of information bits
    n = 200  # Desired codeword length
    num_bits_per_symbol = 4
    ebno_dbs = tf.linspace(0.0, 20.0, 21)
    batch_size = 100
    max_mc_iter = 1000

    # Initialize components
    resource_grid_config = ResourceGridConfig(num_ofdm_symbols, fft_size, subcarrier_spacing, num_tx, num_tx_ant)
    channel = ChannelModel(num_tx, num_rx, num_tx_ant, num_rx_ant, resource_grid_config.get_resource_grid(), correlation_distance)
    encoder = Encoder(k, n, num_bits_per_symbol)
    decoder = LDPC5GDecoderWrapper(encoder)
    mapper_demapper = MapperDemapper(num_bits_per_symbol)
    simulator = Simulator(channel, encoder, decoder, mapper_demapper, resource_grid_config.get_resource_grid(), ebno_dbs, batch_size, max_mc_iter)
    plotter = Plotter()

    # Run simulation
    ber, bler = simulator.simulate()

    # Plot results
    plotter.plot_results(ebno_dbs, ber, bler, "6G Simulation")

if __name__ == "__main__":
    main()