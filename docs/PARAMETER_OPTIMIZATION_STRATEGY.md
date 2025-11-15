# Parameter Optimization Strategy for 6G Channel Estimation Simulation

## Overview

This document describes the strategy and methodology for finding the optimal simulation parameters for channel estimation in 6G smart factory systems. The goal is to maximize simulation parameters (batch size, FFT size, antenna counts, etc.) while maintaining 3GPP 6G compliance and ensuring the simulation runs successfully on the available hardware.

## Problem Statement

Channel estimation simulations for 6G systems require careful parameter selection to balance:

1. **Computational Requirements**: Larger parameters (FFT size, antenna counts, batch size) increase memory usage and computation time
2. **6G 3GPP Compliance**: Parameters must align with 6G standards (e.g., FFT sizes 512-16384, massive MIMO configurations)
3. **Hardware Constraints**: System must not run out of memory or exceed timeout limits
4. **Simulation Accuracy**: Parameters should be large enough to provide meaningful results

The challenge is finding the maximum parameters that satisfy all constraints.

## Strategy

### Approach

We employ an **incremental parameter exploration** strategy with the following phases:

1. **Initial Configuration**: Start with conservative 6G-compliant parameters
2. **Incremental Testing**: Systematically increase parameters while maintaining 6G compliance
3. **Failure Detection**: Identify when the system fails (OOM, timeout, or other errors)
4. **Binary Search Refinement**: Use binary search to find the exact boundary between working and failing configurations
5. **Result Documentation**: Save the last working configuration as the maximum viable parameters

### Key Principles

1. **6G 3GPP Compliance**: All parameters must stay within 6G standard ranges
2. **Conservative Increments**: Use small multipliers to avoid large jumps that might skip viable configurations
3. **Comprehensive Testing**: Test each configuration with actual simulation runs
4. **Graceful Degradation**: If a configuration fails, use binary search to find the maximum working value

## Implementation

### Script: `scripts/find_max_params.py`

The parameter optimization script implements the following features:

#### 1. 6G 3GPP Compliant Starting Parameters

```python
Default Starting Parameters (6G):
- batch_size: 8 (conservative start)
- fft_size: 256 (will increase to 512+ for 6G)
- num_bs_ant: 16 (will increase to 32+ for massive MIMO)
- num_ut: 4 (will increase to 8+ for 6G)
- num_ut_ant: 1 (will increase to 2+ for 6G)
- num_ofdm_symbols: 14 (3GPP standard)
```

#### 2. Parameter Bounds (6G 3GPP Standards)

The script enforces the following bounds during optimization:

- **FFT Size**: 256-16384 (must be power of 2)
  - 6G standard range: 512-16384
  - Automatically rounds to nearest power of 2
  
- **Base Station Antennas**: Up to 4096
  - 6G massive MIMO: 32-4096 antennas
  
- **User Terminals**: Up to 256
  - 6G supports large numbers of simultaneous users
  
- **UT Antennas**: Up to 8
  - 6G devices typically have 2-8 antennas
  
- **OFDM Symbols**: Up to 28
  - 3GPP standard: typically 14, but can extend

#### 3. Increment Strategy

Two strategies are available:

**Balanced Strategy** (default):
- Increases all parameters proportionally
- Uses conservative multipliers (1.25-1.26x)
- Ensures all parameters scale together
- Best for finding overall system limits

**Individual Strategy**:
- Tests one parameter at a time
- Uses larger multipliers (1.414-2.0x)
- Useful for understanding individual parameter limits

#### 4. Binary Search Refinement

When a configuration fails, the script performs binary search for each parameter:

1. Identifies the range between last working and failing values
2. Tests midpoint values
3. Narrows down to find the exact boundary
4. Repeats for each parameter independently

This ensures we find the true maximum, not just a conservative estimate.

#### 5. Memory Estimation

The script estimates memory usage for each configuration:

```
Memory (GB) = (batch_size × fft_size × num_ofdm_symbols × 
               num_bs_ant × num_ut × num_ut_ant × 8 × 2) / (1024³)
```

Where:
- `8` = bytes per float32/complex64
- `2` = real and imaginary parts
- This is a rough estimate; actual usage may vary

## Usage

### Basic Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with default 6G parameters
python scripts/find_max_params.py

# With custom timeout
python scripts/find_max_params.py --timeout 60

# Using individual strategy
python scripts/find_max_params.py --strategy individual
```

### Advanced Options

```bash
# Custom starting parameters
python scripts/find_max_params.py \
    --start-batch-size 16 \
    --start-fft-size 512 \
    --start-num-bs-ant 32 \
    --start-num-ut 8 \
    --start-num-ut-ant 2

# Force CPU execution
python scripts/find_max_params.py --cpu

# Custom output location
python scripts/find_max_params.py --output results/my_max_params.json
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--start-batch-size` | 8 | Starting batch size |
| `--start-fft-size` | 256 | Starting FFT size |
| `--start-num-bs-ant` | 16 | Starting BS antennas |
| `--start-num-ut` | 4 | Starting number of UTs |
| `--start-num-ut-ant` | 1 | Starting UT antennas |
| `--start-num-ofdm-symbols` | 14 | Starting OFDM symbols |
| `--strategy` | balanced | Strategy: balanced or individual |
| `--timeout` | 30 | Timeout per test (seconds) |
| `--gpu` | 0 | GPU device number |
| `--cpu` | False | Force CPU execution |
| `--output` | results/max_params.json | Output file path |

## Output Files

The script generates two output files:

### 1. `results/max_params.json`

Complete test history with all iterations:

```json
{
  "max_params": {
    "batch_size": 71,
    "fft_size": 180,
    "num_bs_ant": 14,
    "num_ut": 6,
    "num_ut_ant": 1,
    "num_ofdm_symbols": 18
  },
  "test_history": [
    {
      "iteration": 1,
      "params": {...},
      "success": true,
      "error": null,
      "memory_estimate_gb": 0.027
    },
    ...
  ],
  "final_memory_gb": 0.288
}
```

### 2. `results/max_params_config.json`

Simple configuration file with just the maximum parameters:

```json
{
  "batch_size": 71,
  "fft_size": 180,
  "num_bs_ant": 14,
  "num_ut": 6,
  "num_ut_ant": 1,
  "num_ofdm_symbols": 18
}
```

## Interpreting Results

### Success Criteria

A configuration is considered successful if:

1. **Model Creation**: SystemConfig and Model objects are created without errors
2. **Batch Execution**: `model.run_batch()` completes successfully
3. **Result Validation**: Returns expected keys (`bits`, `bits_hat`)
4. **No Memory Errors**: No `MemoryError` or `ResourceExhaustedError`
5. **No Timeout**: Completes within the specified timeout

### Failure Modes

Common failure modes and their meanings:

1. **Out of Memory (OOM)**: Parameters too large for available RAM
   - Solution: Reduce batch_size or other parameters
   
2. **TensorFlow Resource Exhausted**: GPU/CPU memory limit reached
   - Solution: Reduce parameters or use CPU with more RAM
   
3. **Timeout**: Simulation takes too long
   - Solution: Increase timeout or reduce parameters
   
4. **Assertion Errors**: Invalid parameter combinations
   - Example: `num_effective_subcarriers` must be integer multiple of `num_tx * num_streams_per_tx`
   - Solution: Adjust FFT size or antenna counts

### Using Results

The maximum parameters found can be used in your simulations:

```python
from src.components.config import SystemConfig

# Load maximum parameters
import json
with open('results/max_params_config.json') as f:
    max_params = json.load(f)

# Create configuration
config = SystemConfig(
    fft_size=max_params['fft_size'],
    num_bs_ant=max_params['num_bs_ant'],
    num_ut=max_params['num_ut'],
    num_ut_ant=max_params['num_ut_ant'],
    num_ofdm_symbols=max_params['num_ofdm_symbols']
)

# Use in simulation
batch_size = max_params['batch_size']
```

## Best Practices

### 1. Start Conservative

Begin with smaller parameters and let the script find the maximum. Starting too high may cause immediate failures.

### 2. Monitor Memory

Watch system memory usage during runs. The script provides estimates, but actual usage may vary.

### 3. Adjust Timeout

For larger parameters, increase the timeout to allow simulations to complete:

```bash
python scripts/find_max_params.py --timeout 120
```

### 4. Use Appropriate Strategy

- **Balanced**: When you want to scale all parameters together
- **Individual**: When you want to understand limits of specific parameters

### 5. Validate Results

After finding maximum parameters, validate with a full simulation run:

```bash
python main.py --batch-size <max_batch_size> \
               --scenario umi \
               --estimator ls
```

## 6G 3GPP Compliance

The script ensures all parameters comply with 6G 3GPP standards:

### FFT Size
- **Range**: 512-16384 (6G standard)
- **Constraint**: Must be power of 2
- **Typical Values**: 512, 1024, 2048, 4096, 8192, 16384

### Subcarrier Spacing
- **5G NR**: 15, 30, 60, 120, 240 kHz
- **6G Extended**: 15-960 kHz, up to 3840 kHz for sub-THz

### Massive MIMO
- **Base Station**: 32-4096 antennas (6G)
- **User Terminals**: 2-8 antennas per device

### OFDM Symbols
- **Standard**: 14 symbols per slot (normal CP)
- **Extended**: Can go up to 28 for special configurations

## Troubleshooting

### Script Runs Too Long

- Reduce starting parameters
- Use shorter timeout
- Use individual strategy instead of balanced

### Out of Memory Errors

- Start with smaller batch_size
- Reduce FFT size
- Reduce antenna counts
- Use CPU instead of GPU (if GPU memory limited)

### Invalid Parameter Combinations

Some parameter combinations may not be valid due to system constraints:

- FFT size must allow integer division for resource allocation
- Number of effective subcarriers must be compatible with MIMO configuration
- Adjust parameters to satisfy these constraints

### Results Seem Too Low

If maximum parameters seem conservative:

1. Check available system memory
2. Verify timeout is sufficient
3. Try individual strategy to test each parameter separately
4. Consider using GPU if available

## Example Workflow

1. **Initial Run**: Find baseline maximum parameters
   ```bash
   python scripts/find_max_params.py
   ```

2. **Review Results**: Check `results/max_params.json` for test history

3. **Refine if Needed**: If results seem low, try:
   ```bash
   python scripts/find_max_params.py \
       --start-batch-size 4 \
       --start-fft-size 128 \
       --timeout 60
   ```

4. **Validate**: Use found parameters in actual simulation
   ```bash
   python main.py --batch-size <found_value> ...
   ```

5. **Document**: Record maximum parameters for your hardware configuration

## Future Enhancements

Potential improvements to the optimization strategy:

1. **Multi-Objective Optimization**: Balance memory, speed, and accuracy
2. **Adaptive Timeouts**: Adjust timeout based on parameter size
3. **Parallel Testing**: Test multiple configurations simultaneously
4. **Hardware Profiling**: Automatically detect hardware capabilities
5. **Parameter Dependencies**: Model relationships between parameters
6. **Machine Learning**: Use ML to predict viable configurations

## References

- 3GPP TS 38.211: Physical channels and modulation
- 3GPP TS 38.212: Multiplexing and channel coding
- 3GPP TR 38.901: Study on channel model for frequencies from 0.5 to 100 GHz
- Sionna Documentation: https://nvlabs.github.io/sionna/

## Conclusion

This parameter optimization strategy provides a systematic approach to finding maximum simulation parameters while maintaining 6G 3GPP compliance. By incrementally testing configurations and using binary search refinement, we can identify the optimal parameters for each hardware setup, ensuring efficient use of computational resources while maintaining simulation accuracy.

The strategy balances multiple constraints:
- **6G Standards Compliance**: Ensures parameters align with 3GPP specifications
- **Hardware Limitations**: Respects available memory and computational resources
- **Simulation Requirements**: Maintains sufficient parameter sizes for meaningful results
- **Automation**: Reduces manual parameter tuning effort

Regular re-running of the optimization script is recommended when:
- Hardware configuration changes
- Software dependencies are updated
- Different simulation scenarios are needed
- System resources are reallocated

