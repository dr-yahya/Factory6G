# GPU Setup Guide for 6G Simulations

## Problem

When running simulations, you may see:
```
W0000 00:00:... gpu_device.cc:2342] Cannot dlopen some GPU libraries.
Skipping registering GPU devices...
```

This means TensorFlow cannot find the required CUDA/cuDNN libraries.

## Solution

### For WSL (Windows Subsystem for Linux)

TensorFlow 2.20.0 requires:
- **CUDA**: 11.8 or 12.x
- **cuDNN**: 8.6 or 8.9

#### Option 1: Install CUDA Toolkit and cuDNN (Recommended)

1. **Install CUDA Toolkit 12.x**:
   ```bash
   # Download from NVIDIA website or use package manager
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-4
   ```

2. **Install cuDNN**:
   ```bash
   # Download from NVIDIA (requires account)
   # Or use conda:
   conda install -c conda-forge cudnn
   ```

3. **Set environment variables**:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   export PATH=/usr/local/cuda/bin:$PATH
   ```

#### Option 2: Use TensorFlow with Matching CUDA (Easier)

Install TensorFlow that matches your CUDA version:

```bash
# For CUDA 11.8
pip install tensorflow[and-cuda]==2.15.0

# For CUDA 12.x
pip install tensorflow[and-cuda]==2.20.0
```

#### Option 3: Use Pre-built TensorFlow with CUDA (Easiest for WSL)

```bash
# Uninstall current TensorFlow
pip uninstall tensorflow

# Install TensorFlow with CUDA support
pip install tensorflow[and-cuda]
```

This automatically installs matching CUDA and cuDNN versions.

### For Native Linux

1. **Check CUDA installation**:
   ```bash
   nvidia-smi
   nvcc --version
   ```

2. **Install matching versions**:
   - TensorFlow 2.20.0 needs CUDA 11.8 or 12.x
   - Check: https://www.tensorflow.org/install/source#gpu

3. **Set library paths**:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

## Verification

Run the GPU check script:

```bash
python scripts/gpu/check_gpu.py
```

Expected output:
```
✓ GPU is configured and ready to use!
```

## Quick Fix Script

If you just want to try the automatic fix:

```bash
# Install TensorFlow with CUDA support
pip install --upgrade tensorflow[and-cuda]

# Verify
python scripts/gpu/check_gpu.py
```

## Current Status

- **GPU Hardware**: ✓ Available (RTX 3060 detected)
- **NVIDIA Driver**: ✓ Working (nvidia-smi works)
- **TensorFlow CUDA Support**: ✓ Built with CUDA
- **CUDA Libraries**: ✗ Missing or incompatible versions

## Workaround (CPU Mode)

If GPU setup is not possible right now, the simulation will automatically use CPU:

```bash
# The simulation script will detect GPU unavailability
# and continue with CPU (slower but functional)
python scripts/run_6g_simulation.py
```

The memory management system will handle CPU memory allocation.

## Performance Impact

- **GPU**: ~10-100x faster for large batch sizes
- **CPU**: Slower but functional, suitable for testing

For production runs, GPU is highly recommended.

