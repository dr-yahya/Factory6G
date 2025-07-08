# 6G Multiuser MIMO OFDM Simulation

A modular implementation of 6G multiuser MIMO OFDM uplink simulation using Sionna.

## Project Structure

```
6g-mimo-simulation/
├── config.py           # System configuration parameters
├── antenna_config.py   # Antenna array configurations
├── channel_model.py    # 6G channel model
├── ofdm_config.py      # OFDM resource grid and stream management
├── transmitter.py      # Transmitter chain components
├── receiver.py         # Receiver chain components
├── simulator.py        # Simulation engine
├── main.py            # Main integration and runner
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Features

- **Modular Design**: Each component in a separate file for easy modification
- **6G Parameters**: 
  - 100 GHz carrier frequency (THz band)
  - 256 FFT size with 120 kHz subcarrier spacing
  - 8 simultaneous users
  - 32 BS antennas (4x8 massive MIMO)
  - 16-QAM modulation (extensible)
- **Realistic Channel**: 3GPP 38.901 UMi model adapted for 6G
- **Advanced Processing**: LDPC coding, LMMSE equalization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Customization

### Change System Parameters
Edit `config.py` to modify:
- Carrier frequency
- Number of users
- Antenna configurations
- Modulation schemes
- OFDM parameters

### Add New Components
1. Create a new file for your component
2. Import it in `main.py`
3. Integrate into the simulation chain

### Example: Change to 256-QAM
```python
# In config.py
BITS_PER_SYMBOL = 8  # 256-QAM
```

## Output

The simulation outputs:
1. BER values for each Eb/No point
2. BER vs Eb/No plot
3. System configuration summary

## Extension Ideas

- Add different channel models (UMa, RMa)
- Implement perfect CSI comparison
- Add MIMO precoding
- Include channel coding comparison
- Add real-time visualization