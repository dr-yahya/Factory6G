"""Runtime environment helpers for CPU/GPU and TensorFlow setup.

This replaces the previous src.utils.env module so that top-level scripts
can configure TensorFlow/Sionna runtime before importing heavy modules.
"""

from __future__ import annotations

import os


def configure_env(force_cpu: bool = False, gpu_num: int = 0) -> None:
    """Configure basic environment variables before importing TensorFlow.

    - Set TF logging verbosity lower
    - Optionally force CPU
    - Otherwise select a single GPU by index
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)


def setup_gpu(gpu_num: int = 0) -> None:
    """Optional post-import TensorFlow GPU configuration (safe on CPU-only hosts)."""
    try:
        import tensorflow as tf  # type: ignore[import]
    except Exception:
        return

    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                # Ignore failures and continue; TF will fall back safely.
                pass
    except Exception:
        return


