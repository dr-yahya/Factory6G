import os
import tensorflow as tf
import sionna

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Import components
from config import Config6G
from antenna_config import AntennaConfig
from channel_model import ChannelModel6G
from ofdm_config import OFDMConfig
from transmitter import Transmitter6G
from receiver import Receiver6G
from simulator import Simulator6G

def main():
    """Main simulation function"""
    
    # Set random seed
    config = Config6G()
    sionna.phy.config.seed = config.SEED
    
    print("=== 6G Multiuser MIMO OFDM Simulation ===")
    print(f"Carrier Frequency: {config.CARRIER_FREQUENCY/1e9:.0f} GHz")
    print(f"Number of Users: {config.NUM_UT}")
    print(f"BS Antennas: {config.NUM_BS_ANT_ROWS * config.NUM_BS_ANT_COLS}")
    print(f"Modulation: {2**config.BITS_PER_SYMBOL}-QAM")
    print(f"FFT Size: {config.FFT_SIZE}")
    print("-" * 40)
    
    # 1. Configure antennas
    antenna_config = AntennaConfig()
    ut_antenna = antenna_config.create_ut_antenna()
    bs_antenna = antenna_config.create_bs_antenna()
    
    # 2. Configure OFDM and streams
    ofdm_config = OFDMConfig()
    resource_grid = ofdm_config.rg
    stream_management = ofdm_config.sm
    
    # 3. Configure channel
    channel = ChannelModel6G(ut_antenna, bs_antenna, resource_grid)
    
    # 4. Configure transmitter
    transmitter = Transmitter6G(resource_grid)
    
    # 5. Configure receiver
    receiver = Receiver6G(resource_grid, stream_management, transmitter.encoder)
    
    # 6. Create simulator
    simulator = Simulator6G(transmitter, receiver, channel, resource_grid)
    
    # 7. Run simulation
    print("\nRunning simulation...")
    ebno_range = range(-5, 21, 5)
    ebno_values, ber_results = simulator.run_simulation(ebno_range)
    
    # 8. Plot results
    simulator.plot_results(ebno_values, ber_results)
    
    print("\nSimulation completed!")

if __name__ == "__main__":
    main()