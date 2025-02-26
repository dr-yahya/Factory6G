# main.py
from phy.channel import ChannelModel, ChannelInterface
from phy.resource_grid import ResourceGridConfig
from phy.encoding import Encoder, LDPC5GDecoderWrapper, EncoderInterface, DecoderInterface
from phy.mapping import MapperDemapper
from phy.simulation import Simulator
from phy.plotting import Plotter
import tensorflow as tf

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
    k = 100
    n = 200
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