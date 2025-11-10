# Factory6G

A research project for 6G wireless communication systems in smart factory environments, focusing on channel estimation and MIMO-OFDM systems.

## Project Structure

```
Factory6G/
├── src/                    # Source code
│   ├── components/        # Modular physical layer components
│   │   ├── __init__.py
│   │   ├── config.py     # System configuration parameters
│   │   ├── antenna.py    # Antenna array configuration
│   │   ├── transmitter.py # Transmitter chain (encoder, mapper, etc.)
│   │   ├── channel.py    # Channel models (3GPP TR 38.901)
│   │   ├── receiver.py   # Receiver chain (estimator, equalizer, decoder)
│   │   ├── estimators/   # Channel estimator implementations
│   │   │   ├── __init__.py
│   │   │   └── neural_estimator.py  # Neural refinement on top of LS
│   │   └── README.md     # Component documentation
│   └── models/            # Complete system models
│       ├── __init__.py
│       ├── model.py      # Main 5G/6G OFDM MIMO system model
│       └── e2e_channel_estimation.py  # End-to-end neural channel estimation
├── notebooks/             # Jupyter notebooks for experiments
│   ├── main..ipynb       # Main BER/BLER simulation notebook
│   ├── notes.ipynb       # Research notes on 6G parameters
│   └── results_5G.ipynb  # 5G simulation results
├── references/           # Research papers and documentation
│   ├── Channel Estimation/
│   └── Sionna/
├── results/              # Output results and plots
├── docs/                # Additional documentation
└── README.md            # This file
```

## Overview

This project implements and evaluates wireless communication systems for smart factory applications using:

- **Sionna Framework**: NVIDIA's GPU-accelerated wireless simulation framework
- **3GPP TR 38.901 Channel Models**: Standardized channel models (UMi, UMa, RMa)
- **MIMO-OFDM Systems**: 4×8 MIMO uplink configuration
- **Channel Estimation**: Traditional LS estimator and neural network refinement
- **LDPC Coding**: 5G LDPC encoder/decoder

## Key Features

### 1. Component-Based Architecture

The physical layer is divided into modular, reusable components:

- **Configuration** (`src/components/config.py`): Centralized system parameters
- **Antenna** (`src/components/antenna.py`): Antenna array setup for BS and UTs
- **Transmitter** (`src/components/transmitter.py`): Encoding, mapping, resource grid mapping
- **Channel** (`src/components/channel.py`): 3GPP TR 38.901 channel models
- **Receiver** (`src/components/receiver.py`): Channel estimation, equalization, demapping, decoding
- **Estimators** (`src/components/estimators/`): LS and neural channel estimators

This modular design enables:
- Easy component replacement and extension
- Independent testing of each component
- Flexible system composition
- Better code maintainability

### 2. Traditional Signal Processing Model (`src/models/model.py`)

- Composes all components into a complete system
- OFDM-based MIMO system with 3GPP TR 38.901 channel models
- Supports perfect and imperfect CSI scenarios
- LDPC 5G encoding/decoding
- LMMSE equalization
- Configurable for different scenarios (UMi, UMa, RMa)

**System Parameters:**
- Carrier Frequency: 3.5 GHz
- FFT Size: 128
- Subcarrier Spacing: 30 kHz
- Antennas: 4 UTs × 8 BS antennas
- Modulation: QPSK (2 bits/symbol)
- Code Rate: 0.5

### 2. End-to-End Neural Channel Estimation (`src/models/e2e_channel_estimation.py`)

- Differentiable end-to-end model
- Neural network-based channel estimator
- Joint optimization of encoder, channel estimator, and decoder
- Trained with Binary Cross-Entropy loss

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- TensorFlow 2.x
- Sionna

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Factory6G
```

2. Install dependencies:
```bash
pip install tensorflow sionna numpy matplotlib
```

For GPU support, ensure you have CUDA installed and use:
```bash
pip install tensorflow-gpu
```

## Usage

### Running Simulations

1. **Main Simulation Script** (`main.py`):
   - Command-line interface for running complete system simulations
   - Automatically generates BER/BLER results and plots
   - Supports multiple scenarios, channel estimators, and configurations
   
   ```bash
   # Run default simulation (UMi, perfect and imperfect CSI)
   python main.py
   
   # Run with specific scenario
   python main.py --scenario uma
   
   # Run only perfect CSI
   python main.py --perfect-csi-only
   
   # Custom Eb/No range
   python main.py --ebno-min -5 --ebno-max 15 --ebno-step 2
   
   # Run with neural estimator (requires trained weights)
   python main.py \
       --estimator neural \
       --neural-weights artifacts/neural_channel_estimator.weights.h5

   # Compare LS vs neural estimator
   python main.py \
       --imperfect-csi-only \
       --estimator ls neural \
       --neural-weights artifacts/neural_channel_estimator.weights.h5

   # See all options
   python main.py --help
   ```

2. **Jupyter Notebook** (`notebooks/main..ipynb`):
   - Interactive simulation notebook
   - Useful for experimentation and analysis
   - Runs BER/BLER simulations across different Eb/No values
   - Compares perfect vs imperfect CSI performance

3. **Neural Estimator Training** (`train_neural_estimator.py`):
   ```bash
   # Quick training run (CPU friendly)
   python train_neural_estimator.py --num-batches 30 --batch-size 8 --epochs 2

   # Custom output path (ensures .weights.h5 suffix)
   python train_neural_estimator.py --output artifacts/my_estimator.weights.h5
   ```

4. **End-to-End Training** (`src/models/e2e_channel_estimation.py`):
   ```bash
   python src/models/e2e_channel_estimation.py
   ```

### Example: Using the Model Class

```python
from src.models.model import Model
from sionna.phy.utils import sim_ber

# Create model with imperfect CSI
model = Model(scenario="umi", perfect_csi=False)

# Run BER simulation
ebno_db = [-5, -3, -1, 1, 3, 5]
ber, bler = sim_ber(
    model,
    ebno_db,
    batch_size=128,
    max_mc_iter=1000,
    num_target_block_errors=1000
)
```

### Example: Using Individual Components

```python
from src.components.config import SystemConfig
from src.components.transmitter import Transmitter
from src.components.channel import ChannelModel
from src.components.receiver import Receiver
from sionna.phy.ofdm import ResourceGrid

# Create custom configuration
config = SystemConfig(
    scenario="umi",
    carrier_frequency=3.5e9,
    num_bs_ant=8,
    num_ut=4
)

# Initialize components independently
transmitter = Transmitter(config, resource_grid)
channel = ChannelModel(config, antenna_config, resource_grid)
receiver = Receiver(config, resource_grid, stream_mgmt, encoder)

# Use components as needed
x_rg, bits = transmitter(batch_size=128)
y, h = channel(x_rg, noise_var)
bits_hat = receiver(y, h_hat, err_var, noise_var)
```

### Example: Loading the Neural Channel Estimator

```python
from src.models.model import Model

model = Model(
    scenario="umi",
    perfect_csi=False,
    estimator_type="neural",
    estimator_weights="artifacts/neural_channel_estimator.weights.h5",
    estimator_kwargs={"hidden_units": [32, 32]},
)
```

## AI-Based Channel Estimation Results

A quick comparison using `python main.py --imperfect-csi-only --estimator ls neural --neural-weights artifacts/neural_channel_estimator.weights.h5 --ebno-min 0 --ebno-max 5 --ebno-step 5 --batch-size 32 --max-iter 20 --target-block-errors 20 --no-plot` yields:

| Estimator | Eb/No (dB) | BER           | BLER          |
|-----------|------------|---------------|---------------|
| LS        | 0          | 1.79e-2       | 1.45e-1       |
| LS        | 5          | 5.77e-4       | 3.52e-3       |
| Neural    | 0          | 2.29e-2       | 2.11e-1       |
| Neural    | 5          | 1.32e-3       | 6.25e-3       |

The neural estimator provides a learnable refinement on top of the LS baseline, but performance depends heavily on training data volume and network capacity. The provided training script gives a lightweight CPU-friendly example; for improved accuracy, increase the number of batches, batch size, and epochs.

## Research Context

This project is part of research on 6G wireless systems for smart factories, focusing on:

- **6G Frequency Bands**: Sub-6GHz (6.425-7.125 GHz), mmWave (24-100 GHz), sub-THz (100-300 GHz)
- **Smart Factory Requirements**: Ultra-low latency (<1ms), ultra-high reliability (99.9999%)
- **Massive IoT**: Target of 10M devices/km² (10x improvement over 5G)
- **Channel Models**: 3GPP TR 38.901 Indoor Factory (InF) models

See `notebooks/notes.ipynb` for detailed research notes on 6G parameters and channel models.

## Results

Simulation results comparing perfect vs imperfect CSI are available in `notebooks/results_5G.ipynb`. Key findings:

- **Perfect CSI**: Achieves BLER < 1e-3 at Eb/No ≈ 1 dB
- **Imperfect CSI**: Requires higher Eb/No (≈ 7-9 dB) for similar performance
- Performance gap highlights the importance of accurate channel estimation

See `docs/Channel_Esimation_Enhancment_Result_Comparision.md` for the latest LS vs Neural channel estimation comparison (plots and tables).

## References

Research papers and documentation are stored in the `