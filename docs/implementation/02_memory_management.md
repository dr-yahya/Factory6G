# Memory Management for 6G Simulations

## Overview

This document describes the memory management system implemented to handle large tensor allocations in 6G simulations and prevent out-of-memory (OOM) errors.

## Problem

When running 6G-compliant simulations with large parameters (32 BS antennas, 8 UTs, etc.), TensorFlow/XLA may issue warnings:

```
W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:84] 
Allocation of 2701131776 exceeds 10% of free system memory.
```

These warnings occur when a single tensor allocation exceeds 10% of available system memory. While these are warnings (not errors), they indicate potential memory pressure.

## Solution

The memory management system provides:

1. **Automatic Memory Configuration**: Configures TensorFlow memory growth and limits
2. **Memory Estimation**: Estimates memory requirements before simulation
3. **Optimal Batch Size Calculation**: Automatically adjusts batch size to fit available memory
4. **Periodic Cache Clearing**: Clears TensorFlow cache between batches
5. **Memory Monitoring**: Tracks memory usage during simulation

## Usage

### Basic Usage

The memory management is automatically integrated into `scripts/run_6g_simulation.py`:

```python
from src.utils.memory_manager import (
    configure_tensorflow_memory,
    get_memory_usage,
    estimate_batch_memory_mb,
    get_optimal_batch_size
)

# Configure TensorFlow memory
configure_tensorflow_memory(memory_growth=True)

# Get system memory info
mem_info = get_memory_usage()

# Estimate memory for a batch
estimated_mb = estimate_batch_memory_mb(
    batch_size=8,
    fft_size=512,
    num_ofdm_symbols=14,
    num_bs_ant=32,
    num_ut=8,
    num_ut_ant=2
)

# Calculate optimal batch size
optimal_batch = get_optimal_batch_size(
    max_memory_mb=mem_info['system_available_gb'] * 1024 * 0.5,
    fft_size=512,
    num_ofdm_symbols=14,
    num_bs_ant=32,
    num_ut=8,
    num_ut_ant=2
)
```

### Memory Monitoring

Use the `MemoryMonitor` context manager to track memory usage:

```python
from src.utils.memory_manager import MemoryMonitor

with MemoryMonitor("Simulation"):
    # Your simulation code here
    result = model.run_batch(batch_size, ebno_db)
```

### Manual Cache Clearing

Clear TensorFlow cache manually when needed:

```python
from src.utils.memory_manager import clear_tensorflow_cache
import gc

# Clear cache
clear_tensorflow_cache()
gc.collect()
```

## Memory Estimation Formula

The memory estimation considers:

1. **Resource Grid**: `batch × num_tx × num_streams × num_ofdm × fft_size × 8 bytes` (complex64)
2. **Channel**: `batch × num_rx × num_tx × num_streams × num_ofdm × fft_size × 8 bytes` (complex64)
3. **Bits**: `batch × num_tx × num_streams × num_info_bits × 4 bytes` (float32)
4. **Overhead**: 2x multiplier for intermediate tensors, gradients, etc.

## Automatic Batch Size Adjustment

The system automatically reduces batch size if:
- Estimated memory > 80% of available memory
- This prevents OOM errors while maximizing throughput

## Periodic Cache Clearing

The simulation loop in `main.py` automatically clears TensorFlow cache every 10 iterations:

```python
if iterations % 10 == 0:
    gc.collect()
    tf.keras.backend.clear_session()
```

This prevents memory buildup during long simulations.

## Configuration Options

### TensorFlow Memory Growth

```python
configure_tensorflow_memory(
    memory_growth=True,  # Enable memory growth (recommended)
    memory_limit_mb=None,  # No hard limit
    suppress_allocation_warnings=True  # Suppress warnings
)
```

### GPU Memory Limit

```python
configure_tensorflow_memory(
    memory_growth=False,
    memory_limit_mb=4096,  # Limit to 4 GB
)
```

## Best Practices

1. **Start Small**: Begin with small batch sizes and increase gradually
2. **Monitor Memory**: Use `MemoryMonitor` to track memory usage
3. **Clear Cache**: Clear cache periodically during long simulations
4. **Use Optimal Batch Size**: Let the system calculate optimal batch size
5. **Close Sessions**: Clear TensorFlow sessions between simulations

## Troubleshooting

### Still Getting Allocation Warnings?

These warnings are informational and don't necessarily indicate a problem. However, if you want to eliminate them:

1. **Reduce Batch Size**: Use smaller batch sizes
2. **Reduce Parameters**: Lower antenna counts or FFT size
3. **Use GPU**: GPU memory is separate from system memory
4. **Increase System Memory**: Add more RAM

### Out of Memory Errors

If you get actual OOM errors (not warnings):

1. **Reduce Batch Size**: Start with batch_size=1 or 2
2. **Enable Memory Growth**: `memory_growth=True`
3. **Clear Cache More Frequently**: Reduce the cache clearing interval
4. **Process in Chunks**: Split large simulations into smaller chunks

## Example Output

```
System Memory Status:
  - System Total: 15.55 GB
  - System Available: 11.72 GB
  - System Used: 24.6%
  - Process RSS: 697.9 MB

Memory Estimation:
  - Requested batch size: 8
  - Estimated memory per batch: 935.2 MB
  - Available memory: 12002.1 MB

✓ Batch size 8 is within memory limits
```

## Implementation Details

The memory management system is implemented in:
- `src/utils/memory_manager.py`: Core memory management functions
- `scripts/run_6g_simulation.py`: Integration with simulation script
- `main.py`: Periodic cache clearing in simulation loop

## References

- TensorFlow Memory Management: https://www.tensorflow.org/guide/gpu
- XLA Memory Allocator: TensorFlow XLA documentation
- Python psutil: https://psutil.readthedocs.io/

