# System Architecture

## Overview

Factory6G implements a modular physical layer simulation system with clear separation of concerns.

## Component Hierarchy

```
Model (End-to-End System)
├── Transmitter
│   ├── LDPC Encoder
│   ├── QAM Mapper
│   └── Resource Grid Mapper
├── Channel
│   ├── Antenna Configuration
│   ├── Channel Model (Rayleigh/TR38.901)
│   └── AWGN Noise
└── Receiver
    ├── Channel Estimator (LS/PSO)
    ├── LMMSE Equalizer
    ├── APP Demapper
    └── LDPC Decoder
```

## Core Components

### 1. Transmitter (`src/components/transmitter.py`)

**Responsibilities:**
- Generate random information bits
- LDPC encoding (rate 0.5)
- QAM modulation (QPSK)
- Resource grid mapping

**Key Methods:**
- `call(batch_size)` → Returns (x_rg, bits, x_qam)

### 2. Channel (`src/components/channel.py`)

**Responsibilities:**
- Apply channel model (Rayleigh or TR38.901)
- Add AWGN noise
- Manage topology

**Key Methods:**
- `__call__(x_rg, no)` → Returns (y, h)
- `set_topology(batch_size)` → Updates channel realization

### 3. Receiver (`src/components/receiver.py`)

**Responsibilities:**
- Channel estimation
- Equalization
- Demapping
- LDPC decoding

**Key Methods:**
- `estimate_channel(y, no)` → Returns (h_hat, err_var)
- `equalize(y, h_hat, err_var, no)` → Returns (x_hat, no_eff)
- `demap(x_hat, no_eff)` → Returns LLR
- `decode(llr)` → Returns (bits_hat, iterations)

### 4. Channel Estimators (`src/components/estimators/`)

#### LS Linear Estimator
- Standard Sionna implementation
- Linear interpolation across frequency
- Fast but noise-sensitive

#### PSO Estimator (`pso_estimator.py`)
- Polynomial fitting via PSO
- Frequency-domain smoothing
- Superior noise robustness

**Key Methods:**
- `call(y, noise_variance)` → Returns (h_hat, err_var)

### 5. Model (`src/models/model.py`)

**Responsibilities:**
- Orchestrate end-to-end transmission
- Manage resource allocation
- Collect metrics

**Key Methods:**
- `call(batch_size, ebno_db)` → Returns (bits, bits_hat)
- `run_batch(...)` → Returns detailed metrics dict

## Data Flow

```
1. Transmitter:
   bits → LDPC → QAM → Resource Grid → x_rg

2. Channel:
   x_rg → Channel Model → + Noise → y
                        ↓
                        h (true channel)

3. Receiver:
   y → Channel Estimation → h_hat
   y, h_hat → Equalization → x_hat
   x_hat → Demapping → LLR
   LLR → LDPC Decoding → bits_hat

4. Metrics:
   Compare bits vs bits_hat → BER, BLER, etc.
```

## Configuration System

### SystemConfig (`src/components/config.py`)

Centralized configuration for all components:

```python
config = SystemConfig(
    carrier_frequency=3.5e9,
    fft_size=512,
    subcarrier_spacing=30e3,
    num_ofdm_symbols=14,
    num_bs_ant=4,
    num_ut=4,
    num_ut_ant=1,
    channel_model_type="rayleigh",
    ...
)
```

### Scenario Specification (`src/sim/scenarios/spec.py`)

Defines complete simulation scenarios:

```python
ScenarioSpec(
    name="6g_pso_enhanced",
    estimators=["pso"],
    batch_size=16,
    max_iter=500,
    ebno_min=-4.0,
    ebno_max=6.0,
    ...
)
```

## Simulation Framework

### Runner (`src/sim/runner.py`)

**Responsibilities:**
- Execute Monte Carlo simulations
- Collect metrics across Eb/No range
- Save results and generate plots

**Workflow:**
```python
for perfect_csi in [True, False]:
    for ebno_db in ebno_range:
        for iteration in range(max_iter):
            results = model.run_batch(batch_size, ebno_db)
            accumulator.update(results)
            if converged:
                break
        metrics = accumulator.finalize()
```

### Metrics (`src/sim/metrics.py`)

**Collected Metrics:**
- BER (Bit Error Rate)
- BLER (Block Error Rate)
- SINR (Signal-to-Interference-plus-Noise Ratio)
- NMSE (Normalized Mean Square Error)
- Throughput
- Latency
- Energy consumption

## Extension Points

### Adding New Estimators

1. Create estimator in `src/components/estimators/`
2. Inherit from `sionna.phy.Block`
3. Implement `call(y, noise_variance)` → `(h_hat, err_var)`
4. Register in `src/components/estimators/__init__.py`
5. Add to scenario configuration

### Adding New Scenarios

1. Create file in `src/sim/scenarios/`
2. Define `SCENARIO = ScenarioSpec(...)`
3. Import in `src/sim/scenarios/__init__.py`
4. Run with `--scenario-profile your_scenario`

### Adding New Metrics

1. Extend `MetricsAccumulator` in `src/sim/metrics.py`
2. Update `run_batch()` in `src/models/model.py`
3. Modify plotting in `src/sim/plotting.py`

## Design Patterns

### 1. Dependency Injection
Components receive dependencies via constructor:
```python
receiver = Receiver(config, rg, sm, encoder, estimator)
```

### 2. Strategy Pattern
Channel estimators are interchangeable:
```python
estimator = LSChannelEstimator(rg)  # or
estimator = PSOChannelEstimator(config, rg)
```

### 3. Builder Pattern
Scenarios build complete configurations:
```python
spec = ScenarioSpec(...)
config = spec.build_config()
```

## Performance Considerations

### Memory Management
- Batch processing for GPU efficiency
- Periodic garbage collection
- TensorFlow session clearing

### Computational Efficiency
- Vectorized operations (NumPy/TensorFlow)
- JIT compilation (@tf.function)
- Parallel batch processing

### Scalability
- Configurable batch sizes
- Adjustable iteration counts
- Modular component design

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock dependencies
- Edge case validation

### Integration Tests
- End-to-end transmission
- Scenario execution
- Metrics validation

### Performance Tests
- Benchmark against baselines
- Convergence verification
- Statistical stability

## Next Steps

- [Channel Estimation](05_Channel_Estimation.md) - PSO deep dive
- [API Reference](09_API_Reference.md) - Detailed API docs
- [Configuration Guide](10_Configuration_Guide.md) - Parameter tuning
