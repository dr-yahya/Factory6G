# Installation Guide

Detailed installation instructions for Factory6G.

## System Requirements

### Minimum Requirements
- **OS:** Linux (Ubuntu 20.04+), WSL2, or macOS 11+
- **Python:** 3.9 or higher
- **RAM:** 8GB
- **Storage:** 2GB free space
- **CPU:** Multi-core processor (4+ cores recommended)

### Recommended Requirements
- **OS:** Ubuntu 22.04 LTS or WSL2
- **Python:** 3.10
- **RAM:** 16GB+
- **Storage:** 10GB free space
- **CPU:** 8+ cores
- **GPU:** NVIDIA GPU with CUDA support (optional, for acceleration)

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Factory6G.git
cd Factory6G
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Test import
python -c "import sionna; import tensorflow as tf; print('Installation successful!')"
```

## Dependencies

### Core Dependencies
- **TensorFlow:** 2.13+ (Deep learning framework)
- **Sionna:** 0.16+ (Physical layer simulation)
- **NumPy:** 1.24+ (Numerical computing)
- **Matplotlib:** 3.7+ (Plotting)

### Optional Dependencies
- **CUDA:** 11.8+ (GPU acceleration)
- **cuDNN:** 8.6+ (GPU deep learning)

## GPU Support (Optional)

### NVIDIA GPU Setup

1. **Install NVIDIA Drivers**
```bash
# Ubuntu
sudo ubuntu-drivers autoinstall
```

2. **Install CUDA Toolkit**
```bash
# Download from https://developer.nvidia.com/cuda-downloads
# Follow installation instructions
```

3. **Install cuDNN**
```bash
# Download from https://developer.nvidia.com/cudnn
# Follow installation instructions
```

4. **Verify GPU**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### CPU-Only Mode

If you don't have a GPU or encounter GPU errors:

```bash
python main.py --cpu
```

This will disable GPU and suppress CUDA warnings.

## Platform-Specific Instructions

### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-venv

# Install Factory6G
cd Factory6G
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Install Factory6G
cd Factory6G
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (WSL2 Recommended)

**Option 1: WSL2 (Recommended)**
```bash
# Install WSL2
wsl --install

# Follow Ubuntu instructions above
```

**Option 2: Native Windows**
```powershell
# Install Python from python.org
# Open PowerShell in Factory6G directory

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Troubleshooting

### TensorFlow Installation Issues

**Problem:** TensorFlow fails to install
```bash
# Solution: Use specific version
pip install tensorflow==2.13.0
```

**Problem:** CUDA version mismatch
```bash
# Solution: Install CPU-only TensorFlow
pip install tensorflow-cpu==2.13.0
```

### Sionna Installation Issues

**Problem:** Sionna import fails
```bash
# Solution: Install from GitHub
pip install git+https://github.com/NVlabs/sionna.git
```

### Memory Issues

**Problem:** Out of memory during simulation
```bash
# Solution: Reduce batch size in scenario config
# Edit src/sim/scenarios/*.py
batch_size=4  # Reduce from 16
```

### Import Errors

**Problem:** Module not found
```bash
# Solution: Ensure virtual environment is activated
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

## Verification

Run the test suite to verify installation:

```bash
# Quick test (if available)
python -m pytest tests/ -v

# Manual verification
python -c "
import tensorflow as tf
import sionna
import numpy as np
import matplotlib.pyplot as plt
print('✓ All dependencies installed successfully')
print(f'TensorFlow version: {tf.__version__}')
print(f'Sionna version: {sionna.__version__}')
print(f'NumPy version: {np.__version__}')
"
```

## Next Steps

- [Quick Start Guide](01_Quick_Start.md) - Run your first simulation
- [Project Overview](03_Project_Overview.md) - Understand the architecture
- [Configuration Guide](10_Configuration_Guide.md) - Customize parameters

## Support

For installation issues:
- Check [Troubleshooting](#troubleshooting) section
- Search [GitHub Issues](https://github.com/yourusername/Factory6G/issues)
- Create new issue with error details
