# Component Architecture Overview

## Physical Layer Component Breakdown

The 6G smart factory physical layer has been divided into modular components for better organization, maintainability, and extensibility.

## Component Hierarchy

```
Model (Complete System)
├── SystemConfig (Configuration)
├── AntennaConfig (Antenna Arrays)
├── Transmitter (TX Chain)
│   ├── BinarySource
│   ├── LDPC5GEncoder
│   ├── QAM Mapper
│   └── ResourceGridMapper
├── ChannelModel (Channel)
│   ├── 3GPP TR 38.901 Model (UMi/UMa/RMa)
│   └── OFDMChannel
└── Receiver (RX Chain)
    ├── Channel Estimator (LS or Perfect CSI)
    ├── LMMSE Equalizer
    ├── QAM Demapper
    └── LDPC5GDecoder
├── Estimators
    └── NeuralChannelEstimator (LS baseline + neural refinement)
```

## Component Details

### 1. SystemConfig (`src/components/config.py`)

**Purpose**: Centralized configuration management

**Key Parameters**:
- RF: Carrier frequency, FFT size, subcarrier spacing
- OFDM: Number of symbols, cyclic prefix, pilot indices
- MIMO: Number of BS/UT antennas, streams
- Modulation: Bits per symbol, code rate
- Channel: Scenario (UMi/UMa/RMa), direction, pathloss settings

**Benefits**:
- Single source of truth for all parameters
- Easy to modify and extend
- Type-safe with dataclass
- Can be serialized/deserialized

### 2. AntennaConfig (`src/components/antenna.py`)

**Purpose**: Antenna array configuration

**Components**:
- UT Array: Single antenna, omni-directional
- BS Array: Dual-polarized, 3GPP 38.901 pattern

**Benefits**:
- Separates antenna concerns from system logic
- Easy to modify antenna patterns
- Reusable across different scenarios

### 3. Transmitter (`src/components/transmitter.py`)

**Purpose**: Complete transmitter chain

**Processing Flow**:
```
Bits → LDPC Encoder → QAM Mapper → Resource Grid Mapper → Resource Grid
```

**Components**:
- `BinarySource`: Generates information bits
- `LDPC5GEncoder`: Encodes bits with LDPC
- `Mapper`: Maps coded bits to QAM symbols
- `ResourceGridMapper`: Maps symbols to OFDM resource grid

**Methods**:
- `__call__()`: Complete TX chain (generates bits internally)
- `encode_and_map()`: For custom bit input

**Benefits**:
- Encapsulates all TX processing
- Can be used independently
- Easy to test and modify

### 4. ChannelModel (`src/components/channel.py`)

**Purpose**: Channel modeling and propagation

**Features**:
- Supports 3GPP TR 38.901 scenarios (UMi, UMa, RMa)
- Handles topology generation
- Applies OFDM channel effects (fading, noise, multipath)

**Methods**:
- `set_topology()`: Generate new channel topology
- `__call__()`: Apply channel to input signal

**Benefits**:
- Easy to swap channel models
- Supports multiple scenarios
- Handles complex channel effects

### 5. Receiver (`src/components/receiver.py`)

**Purpose**: Complete receiver chain

**Processing Flow**:
```
Received Signal → Channel Estimation → Equalization → Demapping → Decoding → Bits
```

**Components**:
- `LSChannelEstimator`: Least-squares channel estimation (or perfect CSI)
- `LMMSEEqualizer`: Linear MMSE equalization
- `Demapper`: QAM demapping to LLRs
- `LDPC5GDecoder`: LDPC decoding

**Methods**:
- `estimate_channel()`: Channel estimation
- `equalize()`: Signal equalization
- `demap()`: Symbol demapping
- `decode()`: Bit decoding
- `__call__()`: Complete RX chain
- `process_with_perfect_csi()`: RX with perfect channel knowledge

**Benefits**:
- Modular receiver processing
- Easy to swap estimators/equalizers
- Supports both perfect and imperfect CSI

### 6. Estimators (`src/estimators/neural_estimator.py`)

**Purpose**: Provide learnable refinements for the LS estimator.

**Components**:
- `NeuralChannelEstimator`: Wraps the LS estimator and applies a point-wise neural network to the real/imaginary parts of the estimate.

**Training**:
- Use `scripts/train_neural_estimator.py` to generate supervised pairs `(h_ls, h_true)` and fit the neural network.
- Saved weights can be loaded in `Model(estimator_type="neural", estimator_weights=...)`.

**Benefits**:
- Differentiable drop-in replacement for LS
- Lightweight architecture operating per resource element
- Allows experimentation with data-driven channel estimation

## Usage Patterns

### Pattern 1: Using Complete Model

```python
from src.models.model import Model

# Create complete system
model = Model(scenario="umi", perfect_csi=False)

# Simulate
bits_tx, bits_rx = model(batch_size=128, ebno_db=5.0)
```

### Pattern 2: Using Individual Components

```python
from src.components import SystemConfig, Transmitter, ChannelModel, Receiver

# Create configuration
config = SystemConfig(scenario="umi")

# Initialize components
transmitter = Transmitter(config, resource_grid)
channel = ChannelModel(config, antenna_config, resource_grid)
receiver = Receiver(config, resource_grid, stream_mgmt, encoder)

# Use independently
x_rg, bits = transmitter(batch_size=128)
y, h = channel(x_rg, noise_var)
bits_hat = receiver(y, h_hat, err_var, noise_var)
```

### Pattern 3: Custom Component Replacement

```python
# Replace channel estimator with custom neural network
class NeuralChannelEstimator:
    def __call__(self, y, noise_var):
        # Custom estimation logic
        return h_hat, err_var

# Use in receiver
receiver._channel_estimator = NeuralChannelEstimator()
```

## Extension Points

### Adding New Channel Models

1. Extend `ChannelModel._create_channel_model()`
2. Add new scenario to `SystemConfig`
3. Implement channel model class

### Adding New Channel Estimators

1. Create estimator class with `__call__(y, noise_var)` method
2. Modify `Receiver.__init__()` to accept estimator type
3. Update `Receiver.estimate_channel()`

### Adding New Modulation Schemes

1. Modify `Transmitter` to support different mappers
2. Update `Receiver` to support corresponding demappers
3. Add parameters to `SystemConfig`

### Adding New Coding Schemes

1. Create encoder/decoder classes
2. Modify `Transmitter` and `Receiver` to use them
3. Update code rate calculations in `SystemConfig`

## Benefits of This Architecture

1. **Modularity**: Each component has a single responsibility
2. **Reusability**: Components can be used in different combinations
3. **Testability**: Each component can be tested independently
4. **Extensibility**: Easy to add new components or modify existing ones
5. **Maintainability**: Changes are localized to specific components
6. **Flexibility**: Easy to swap components for different algorithms
7. **Readability**: Clear separation of concerns makes code easier to understand

## File Structure

```
src/
├── components/
│   ├── __init__.py          # Component exports
│   ├── config.py            # System configuration
│   ├── antenna.py           # Antenna arrays
│   ├── transmitter.py       # TX chain
│   ├── channel.py           # Channel models
│   ├── receiver.py          # RX chain
│   └── README.md            # Component documentation
└── models/
    ├── __init__.py
    ├── model.py             # Complete system (uses components)
    └── e2e_channel_estimation.py  # Neural E2E model
```

## Migration Notes

The original monolithic `Model` class has been refactored to use components. The public API remains the same, so existing code should continue to work:

```python
# Old code (still works)
model = Model(scenario="umi", perfect_csi=False)
bits_tx, bits_rx = model(batch_size=128, ebno_db=5.0)

# New code (more flexible)
config = SystemConfig(scenario="umi", carrier_frequency=6.0e9)
model = Model(config=config, perfect_csi=False)
```

