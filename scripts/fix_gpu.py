#!/usr/bin/env python3
"""
Fix GPU setup for TensorFlow.

This script installs TensorFlow with CUDA support to enable GPU acceleration.
"""

import sys
import subprocess
import os

def main():
    import sys
    
    print("=" * 70)
    print("GPU Setup Fix for TensorFlow")
    print("=" * 70)
    print()
    
    print("This script will install TensorFlow with CUDA support.")
    print("This ensures matching CUDA and cuDNN versions are installed.")
    print()
    
    # Check for --yes flag for non-interactive mode
    auto_confirm = '--yes' in sys.argv or '-y' in sys.argv
    
    if not auto_confirm:
        try:
            response = input("Continue? (y/n): ").strip().lower()
            if response != 'y':
                print("Cancelled.")
                return
        except EOFError:
            # Non-interactive mode - auto-confirm
            print("Non-interactive mode detected. Proceeding with installation...")
            print()
    else:
        print("Auto-confirming installation (--yes flag)...")
        print()
    
    print()
    print("Installing TensorFlow with CUDA support...")
    print("This may take a few minutes...")
    print()
    
    try:
        # Install tensorflow[and-cuda] which includes matching CUDA/cuDNN
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "tensorflow[and-cuda]"],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✓ Installation completed successfully!")
        print()
        print("Verifying GPU setup...")
        print()
        
        # Test GPU
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✓ GPU is now available: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"  - {gpu}")
            print()
            print("=" * 70)
            print("✓ GPU setup complete! You can now run simulations with GPU.")
            print("=" * 70)
        else:
            print("⚠ GPU still not detected. Please check:")
            print("  1. NVIDIA driver is installed: nvidia-smi")
            print("  2. CUDA libraries are accessible")
            print("  3. Run: python scripts/check_gpu.py for diagnostics")
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation failed:")
        print(e.stderr)
        print()
        print("Manual installation:")
        print("  pip install --upgrade tensorflow[and-cuda]")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    main()

