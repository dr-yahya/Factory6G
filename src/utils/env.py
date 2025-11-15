"""
Environment configuration utilities for 6G simulations.

This module provides functions to configure the execution environment
before importing TensorFlow/Sionna to avoid CUDA library errors.
"""

import os
from typing import Optional


def configure_env(force_cpu: bool, gpu_num: Optional[int]):
    """
    Configure environment variables BEFORE importing TensorFlow/Sionna.
    This avoids noisy CUDA library errors when GPU runtime is unavailable.
    
    Args:
        force_cpu: If True, disable GPU visibility
        gpu_num: GPU device number to use (None = use default)
    """
    # Reduce TensorFlow log verbosity
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if force_cpu:
        # Fully disable GPU visibility for TF/XLA/JAX
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif gpu_num is not None and os.getenv("CUDA_VISIBLE_DEVICES") is None:
        # Respect explicit GPU selection if provided
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"


def setup_gpu(gpu_num: int = 0):
    """
    Configure GPU settings after TensorFlow is imported.
    
    Args:
        gpu_num: GPU device number to configure
    """
    import tensorflow as tf  # imported late to respect env config

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✓ Using GPU: {gpus[0]}")
        except RuntimeError as e:
            print(f"Warning: {e}")
    else:
        print("⚠ No GPU found, using CPU")

