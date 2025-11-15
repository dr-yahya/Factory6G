#!/usr/bin/env python3
"""
Fix GPU detection in WSL environment.

This script creates symlinks and sets up library paths to enable
GPU detection in WSL with TensorFlow.
"""

import os
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("WSL GPU Fix for TensorFlow")
    print("=" * 70)
    print()
    
    project_root = Path(__file__).parent.parent.parent
    venv_path = project_root / ".venv" / "lib" / "python3.10" / "site-packages"
    
    # Check if NVIDIA libraries are installed
    nvidia_path = venv_path / "nvidia"
    if not nvidia_path.exists():
        print("✗ NVIDIA libraries not found!")
        print("  Run: python scripts/gpu/fix_gpu.py --yes")
        return False
    
    print("✓ NVIDIA libraries found")
    print()
    
    # Find key CUDA libraries
    cudart_libs = list(nvidia_path.glob("**/libcudart.so*"))
    cudnn_libs = list(nvidia_path.glob("**/libcudnn.so*"))
    
    print(f"Found {len(cudart_libs)} libcudart.so files")
    print(f"Found {len(cudnn_libs)} libcudnn.so files")
    print()
    
    if not cudart_libs:
        print("⚠ Warning: libcudart.so not found in NVIDIA packages")
        print("  This might be a version compatibility issue")
    
    # Create a wrapper script that sets up environment correctly
    wrapper_script = project_root / "scripts" / "gpu" / "run_with_gpu.sh"
    wrapper_script.parent.mkdir(parents=True, exist_ok=True)
    
    wrapper_content = f"""#!/bin/bash
# GPU-enabled wrapper script for WSL

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu:{venv_path}/nvidia/cuda_nvrtc/lib:{venv_path}/nvidia/cudnn/lib:{venv_path}/nvidia/cublas/lib:{venv_path}/nvidia/cufft/lib:{venv_path}/nvidia/curand/lib:{venv_path}/nvidia/cusolver/lib:{venv_path}/nvidia/cusparse/lib:{venv_path}/nvidia/nccl/lib:$LD_LIBRARY_PATH"

# Activate virtual environment
source {project_root}/.venv/bin/activate

# Run the command
exec "$@"
"""
    
    with open(wrapper_script, 'w') as f:
        f.write(wrapper_content)
    
    os.chmod(wrapper_script, 0o755)
    
    print("✓ Created GPU wrapper script: scripts/gpu/run_with_gpu.sh")
    print()
    print("Usage:")
    print("  bash scripts/gpu/run_with_gpu.sh python scripts/run_6g_simulation.py")
    print()
    
    # Also update the memory manager to use this
    print("=" * 70)
    print("Note: WSL GPU support may require:")
    print("  1. WSL2 with CUDA support enabled")
    print("  2. NVIDIA driver 470+ on Windows host")
    print("  3. Proper CUDA toolkit installation")
    print()
    print("If GPU still doesn't work, the simulation will use CPU.")
    print("CPU mode is functional but slower.")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    main()

