# WSL GPU Troubleshooting Guide

## Current Status

After installing `tensorflow[and-cuda]`, GPU is still not detected in WSL. This is a known issue with WSL CUDA bridge configuration.

## Symptoms

- `nvidia-smi` works and shows GPU
- TensorFlow is built with CUDA support
- NVIDIA libraries are installed in `.venv/lib/python3.10/site-packages/nvidia/`
- TensorFlow cannot load GPU libraries: `Cannot dlopen some GPU libraries`

## Root Cause

WSL requires specific CUDA bridge setup between Windows host and WSL2. The libraries are installed, but TensorFlow cannot access them due to:

1. **WSL CUDA Bridge**: WSL2 needs proper CUDA toolkit installation on Windows host
2. **Library Path Resolution**: TensorFlow may not find libraries in the expected locations
3. **Version Compatibility**: TensorFlow 2.20.0 may require specific CUDA/cuDNN versions

## Solutions

### Option 1: Use CPU Mode (Current Workaround)

The simulation **works on CPU** but is slower. This is the current working solution:

```bash
# The simulation will automatically fall back to CPU
source .venv/bin/activate
python scripts/run_6g_simulation.py
```

### Option 2: Fix WSL CUDA Setup (Recommended for Performance)

1. **Install CUDA Toolkit on Windows Host**:
   - Download CUDA Toolkit 12.x from NVIDIA
   - Install on Windows (not in WSL)
   - Ensure NVIDIA driver 470+ is installed on Windows

2. **Verify WSL CUDA Support**:
   ```bash
   # In WSL
   nvidia-smi  # Should work
   ```

3. **Check CUDA Version Compatibility**:
   ```bash
   # TensorFlow 2.20.0 requires CUDA 12.x
   # Verify with:
   python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
   ```

4. **Restart WSL**:
   ```powershell
   # In Windows PowerShell (as admin)
   wsl --shutdown
   # Then restart WSL
   ```

5. **Test GPU Detection**:
   ```bash
   source .venv/bin/activate
   python scripts/check_gpu.py
   ```

### Option 3: Use Docker with GPU Support

If WSL GPU setup is problematic, consider using Docker with NVIDIA Container Toolkit:

```bash
# Install NVIDIA Container Toolkit
# Then run simulation in Docker with GPU passthrough
```

## Current Configuration

- **TensorFlow Version**: 2.20.0
- **CUDA Built**: Yes
- **NVIDIA Libraries**: Installed in `.venv/lib/python3.10/site-packages/nvidia/`
- **GPU Detection**: Not working (WSL CUDA bridge issue)
- **Fallback**: CPU mode (functional but slower)

## Library Paths

The following paths are configured in `src/utils/memory_manager.py`:

- `/usr/lib/wsl/lib` - WSL CUDA libraries
- `.venv/lib/python3.10/site-packages/nvidia/*/lib` - TensorFlow CUDA libraries

## Verification Commands

```bash
# Check GPU hardware
nvidia-smi

# Check TensorFlow CUDA support
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"

# Check GPU detection
source .venv/bin/activate
python scripts/check_gpu.py

# Run simulation (will use CPU if GPU unavailable)
python scripts/run_6g_simulation.py
```

## Performance Impact

- **CPU Mode**: ~10-50x slower than GPU (depending on workload)
- **GPU Mode**: Optimal performance (when working)

For the current 6G simulation parameters:
- **FFT Size**: 512
- **BS Antennas**: 32
- **User Terminals**: 8
- **UT Antennas**: 2
- **OFDM Symbols**: 14

CPU mode is acceptable for testing but GPU is recommended for production runs.

## Next Steps

1. **Immediate**: Continue with CPU mode (simulation works)
2. **Short-term**: Fix WSL CUDA bridge setup (see Option 2)
3. **Long-term**: Consider Docker or native Linux for production

## References

- [TensorFlow GPU Setup](https://www.tensorflow.org/install/gpu)
- [WSL CUDA Support](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [NVIDIA WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

