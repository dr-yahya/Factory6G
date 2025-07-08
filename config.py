class Config6G:
    """6G System Configuration Parameters"""
    
    # Frequency parameters
    CARRIER_FREQUENCY = 100e9      # 100 GHz (THz band for 6G)
    SUBCARRIER_SPACING = 120e3     # 120 kHz
    
    # OFDM parameters
    NUM_OFDM_SYMBOLS = 14
    FFT_SIZE = 256
    CYCLIC_PREFIX_LENGTH = 32
    PILOT_SYMBOL_INDICES = [2, 11]
    
    # MIMO parameters
    NUM_UT = 8                     # Number of user terminals
    NUM_UT_ANT = 1                 # Antennas per UT
    NUM_BS_ANT_ROWS = 4
    NUM_BS_ANT_COLS = 8            # Total: 32 BS antennas
    
    # Modulation and coding
    BITS_PER_SYMBOL = 4            # 16-QAM
    CODE_RATE = 0.5
    
    # Simulation parameters
    BATCH_SIZE = 32
    SCENARIO = "umi"
    DIRECTION = "uplink"
    
    # Random seed
    SEED = 42