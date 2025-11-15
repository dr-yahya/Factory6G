#!/usr/bin/env python3
"""
Check GPU availability and configuration for TensorFlow.

This script helps diagnose GPU setup issues.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup GPU libraries BEFORE importing TensorFlow
from src.utils.memory_manager import setup_gpu_libraries

print("=" * 70)
print("GPU Configuration Check")
print("=" * 70)
print()

# Setup library paths
print("1. Setting up GPU library paths...")
setup_gpu_libraries()
ld_path = os.environ.get("LD_LIBRARY_PATH", "not set")
print(f"   LD_LIBRARY_PATH: {ld_path}")
print()

# Check CUDA_VISIBLE_DEVICES
print("2. Checking CUDA_VISIBLE_DEVICES...")
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
if cuda_visible == "-1":
    print("   ⚠ Warning: GPU is disabled (CUDA_VISIBLE_DEVICES=-1)")
    print("   Setting to '0' to enable GPU...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif cuda_visible == "not set":
    print("   Setting to '0' to enable GPU...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print()

# Check nvidia-smi
print("3. Checking NVIDIA driver...")
import subprocess
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   ✓ nvidia-smi works")
        # Extract GPU info
        lines = result.stdout.split('\n')
        for line in lines:
            if "NVIDIA" in line or "Driver Version" in line or "CUDA Version" in line:
                print(f"   {line.strip()}")
    else:
        print("   ✗ nvidia-smi failed")
except Exception as e:
    print(f"   ✗ nvidia-smi error: {e}")
print()

# Now import TensorFlow
print("4. Importing TensorFlow...")
import tensorflow as tf
print(f"   TensorFlow version: {tf.__version__}")
print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")
print()

# Check GPU devices
print("5. Checking GPU devices...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   ✓ Found {len(gpus)} GPU device(s):")
    for i, gpu in enumerate(gpus):
        print(f"     [{i}] {gpu}")
        try:
            # Try to get GPU details
            details = tf.config.experimental.get_device_details(gpu)
            if details:
                print(f"         Details: {details}")
        except:
            pass
else:
    print("   ✗ No GPU devices found")
    print()
    print("   Troubleshooting:")
    print("   - Check CUDA installation: nvidia-smi should work")
    print("   - Verify TensorFlow GPU support:")
    print("     python -c 'import tensorflow as tf; print(tf.test.is_built_with_cuda())'")
    print("   - Check library paths:")
    print(f"     LD_LIBRARY_PATH={ld_path}")
    print("   - For WSL: Ensure CUDA libraries are in /usr/lib/wsl/lib")
    print("   - Install cuDNN if missing")
print()

# Test GPU computation
if gpus:
    print("6. Testing GPU computation...")
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"   ✓ GPU computation successful")
            print(f"   Result: {c.numpy()}")
    except Exception as e:
        print(f"   ✗ GPU computation failed: {e}")
else:
    print("6. Skipping GPU test (no GPU available)")
print()

print("=" * 70)
if gpus:
    print("✓ GPU is configured and ready to use!")
else:
    print("✗ GPU is not available. Simulation will use CPU (slower).")
print("=" * 70)

