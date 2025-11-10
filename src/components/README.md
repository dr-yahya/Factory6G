# Physical Layer Components

This directory contains modular components for the 6G smart factory physical layer system.

## Component Architecture

The physical layer is divided into the following components:

### 1. Configuration (`config.py`)
- **SystemConfig**: Centralized configuration dataclass
- Contains all system parameters (RF, OFDM, MIMO, modulation, coding)
- Easy to modify and extend for different scenarios

### 2. Antenna Configuration (`antenna.py`)
- **AntennaConfig**: Manages antenna arrays for BS and UTs
- Creates and configures antenna arrays based on system parameters
- Supports different antenna patterns (omni, 3GPP 38.901)

### 3. Transmitter (`transmitter.py`)
- **Transmitter**: Complete transmitter chain
- Components:
  - Binary source
  - LDPC encoder
  - QAM mapper
  - Resource grid mapper
- Processes: Bits → Encoded bits → QAM symbols → Resource grid

### 4. Channel Model (`channel.py`)
- **ChannelModel**: 3GPP TR 38.901 channel models
- Supports scenarios: UMi, UMa, RMa
- Handles topology generation and OFDM channel effects
- Applies fading, noise, and multipath propagation

### 5. Receiver (`receiver.py`)
- **Receiver**: Complete receiver chain
- Components:
  - Channel estimator (LS-based, or perfect CSI)
  - LMMSE equalizer
  - QAM demapper
  - LDPC decoder
- Processes: Received signal → Channel estimate → Equalization → LLRs → Decoded bits

## Usage Example

```python
from src.components.config import SystemConfig
from src.components.antenna import AntennaConfig
from src.components.transmitter import Transmitter
from src.components.channel import ChannelModel
from src.components.receiver import Receiver
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.mimo import StreamManagement

# Create configuration
config = SystemConfig(
    scenario="umi",
    carrier_frequency=3.5e9,
    num_bs_ant=8,
    num_ut=4
)

# Setup resource grid
rg = ResourceGrid(...)
sm = StreamManagement(...)

# Initialize components
antenna_config = AntennaConfig(config)
transmitter = Transmitter(config, rg)
channel = ChannelModel(config, antenna_config, rg)
receiver = Receiver(config, rg, sm, transmitter._encoder, perfect_csi=False)

# Use components independently or compose them
```

## Benefits of Component-Based Architecture

1. **Modularity**: Each component has a single, well-defined responsibility
2. **Reusability**: Components can be used independently or in different combinations
3. **Testability**: Each component can be tested in isolation
4. **Extensibility**: Easy to add new components (e.g., different channel estimators, equalizers)
5. **Maintainability**: Changes to one component don't affect others
6. **Flexibility**: Easy to swap components (e.g., different channel models, estimators)

## Extending Components

To add new functionality:

1. **New Channel Estimator**: Create a new estimator class and modify `Receiver` to support it
2. **New Channel Model**: Extend `ChannelModel` to support additional 3GPP scenarios
3. **New Modulation**: Modify `Transmitter` and `Receiver` to support different modulations
4. **New Coding Scheme**: Create new encoder/decoder components and integrate with `Transmitter`/`Receiver`

