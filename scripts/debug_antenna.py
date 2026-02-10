
import sys
import os
import tensorflow as tf
import numpy as np

# Add project root
sys.path.insert(0, os.getcwd())

from src.components.config import SystemConfig
from src.components.antenna import AntennaConfig
from src.sim.env import configure_env

def test_antenna_init():
    configure_env(force_cpu=True, gpu_num=0)
    print("TF Devices:", tf.config.list_physical_devices())
    
    config = SystemConfig()
    print(f"Config: num_bs_ant={config.num_bs_ant}, freq={config.carrier_frequency}")
    
    print("Creating AntennaConfig...")
    try:
        ant_config = AntennaConfig(config)
        print("AntennaConfig created successfully.")
        print(f"BS Array: {ant_config.bs_array}")
    except Exception as e:
        print(f"Failed to create AntennaConfig: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_antenna_init()
