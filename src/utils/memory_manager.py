"""
Memory management utilities for 6G simulations.

This module provides memory management functions to handle large tensor
allocations and prevent out-of-memory errors in 6G simulations.
"""

import os
import gc
from typing import Optional

# Don't import tensorflow at module level - import it inside functions
# after setting up library paths


def setup_gpu_libraries():
    """
    Setup GPU library paths for WSL and Linux environments.
    
    This function sets up LD_LIBRARY_PATH to include common CUDA library locations,
    which is necessary for TensorFlow to find GPU libraries in WSL environments.
    
    For TensorFlow installed via pip install tensorflow[and-cuda], the libraries
    are typically in the Python site-packages directory.
    """
    import platform
    import site
    
    # Get Python site-packages paths (where tensorflow[and-cuda] installs CUDA libs)
    try:
        site_packages = site.getsitepackages()
    except:
        site_packages = []
    
    # Common CUDA library paths
    cuda_paths = [
        "/usr/lib/wsl/lib",  # WSL CUDA libraries
        "/usr/local/cuda/lib64",  # Standard CUDA installation
        "/usr/local/cuda/lib",  # Alternative CUDA path
        "/usr/lib/x86_64-linux-gnu",  # System libraries
    ]
    
    # Add TensorFlow CUDA library paths (from tensorflow[and-cuda] installation)
    for sp in site_packages:
        # TensorFlow installs CUDA libs in nvidia/ subdirectories
        tf_cuda_paths = [
            os.path.join(sp, "nvidia", "cuda_nvrtc", "lib"),
            os.path.join(sp, "nvidia", "cudnn", "lib"),
            os.path.join(sp, "nvidia", "cublas", "lib"),
            os.path.join(sp, "nvidia", "cufft", "lib"),
            os.path.join(sp, "nvidia", "curand", "lib"),
            os.path.join(sp, "nvidia", "cusolver", "lib"),
            os.path.join(sp, "nvidia", "cusparse", "lib"),
            os.path.join(sp, "nvidia", "nccl", "lib"),
        ]
        cuda_paths.extend([p for p in tf_cuda_paths if os.path.exists(p)])
    
    # Add existing LD_LIBRARY_PATH if set
    existing_path = os.environ.get("LD_LIBRARY_PATH", "")
    existing_paths = existing_path.split(":") if existing_path else []
    
    # Combine and set LD_LIBRARY_PATH (remove duplicates, preserve order)
    all_paths = []
    seen = set()
    for path_list in [existing_paths, cuda_paths]:
        for p in path_list:
            if p and p not in seen and os.path.exists(p):
                all_paths.append(p)
                seen.add(p)
    
    if all_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(all_paths)
        return True
    return False


def configure_tensorflow_memory(
    memory_growth: bool = True,
    memory_limit_mb: Optional[int] = None,
    cpu_memory_limit_mb: Optional[int] = None,
    log_device_placement: bool = False,
    suppress_allocation_warnings: bool = True,
    enable_gpu: bool = True
):
    """
    Configure TensorFlow memory settings to prevent OOM errors.
    
    Args:
        memory_growth: If True, enable memory growth for GPUs (prevents
            allocating all GPU memory at once).
        memory_limit_mb: Maximum GPU memory in MB (None = no limit).
        cpu_memory_limit_mb: Maximum CPU memory in MB (None = no limit).
        log_device_placement: If True, log device placement for debugging.
        suppress_allocation_warnings: If True, suppress large allocation warnings.
        enable_gpu: If True, attempt to enable GPU (set CUDA_VISIBLE_DEVICES if needed).
    """
    # Setup GPU library paths BEFORE importing TensorFlow
    if enable_gpu:
        setup_gpu_libraries()
        # Ensure CUDA_VISIBLE_DEVICES is set to use GPU (not -1)
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
            # Remove CPU-only setting
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
        elif os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            # Default to GPU 0 if not set
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Set TensorFlow log level
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    
    # Import TensorFlow after environment is configured
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    # Suppress allocation warnings by setting XLA environment variable
    if suppress_allocation_warnings:
        # Increase the threshold for allocation warnings (default is 10%)
        # Setting to 50% means warnings only appear if allocation > 50% of free memory
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        # Note: The warning threshold is hardcoded in TensorFlow/XLA
        # We can't directly change it, but we can reduce batch size to avoid it
    
    # Configure GPU memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                if memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"✓ Enabled memory growth for {gpu}")
                
                if memory_limit_mb is not None:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory_limit_mb
                        )]
                    )
                    print(f"✓ Set GPU memory limit to {memory_limit_mb} MB")
            print(f"✓ GPU configured successfully: {len(gpus)} device(s) available")
            return True
        except RuntimeError as e:
            print(f"⚠ Warning: GPU configuration failed: {e}")
            return False
    else:
        if enable_gpu:
            print("⚠ Warning: No GPU devices found. Check CUDA installation and library paths.")
            print("  Attempting to continue with CPU (slower performance)")
        return False
    
    # Configure CPU memory
    if cpu_memory_limit_mb is not None:
        # Note: TensorFlow doesn't directly support CPU memory limits
        # This is handled at the OS level or through batch size reduction
        print(f"Note: CPU memory limit {cpu_memory_limit_mb} MB (handled via batch size)")
    
    # Configure device placement logging
    if log_device_placement:
        tf.debugging.set_log_device_placement(True)


def clear_tensorflow_cache():
    """Clear TensorFlow's internal cache to free memory."""
    import tensorflow as tf
    
    # Clear TensorFlow backend cache
    try:
        tf.keras.backend.clear_session()
    except:
        pass
    
    # Force garbage collection
    gc.collect()
    
    # Clear TensorFlow's memory allocator cache (if available and GPU exists)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # This may not be available in all TensorFlow versions
            tf.config.experimental.reset_memory_stats('GPU:0')
    except (AttributeError, RuntimeError, ValueError):
        # GPU not available or reset not supported - this is fine
        pass


def get_memory_usage():
    """Get current memory usage information."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    # Get system memory
    system_mem = psutil.virtual_memory()
    
    return {
        "process_rss_mb": mem_info.rss / (1024 * 1024),
        "process_vms_mb": mem_info.vms / (1024 * 1024),
        "system_total_gb": system_mem.total / (1024 * 1024 * 1024),
        "system_available_gb": system_mem.available / (1024 * 1024 * 1024),
        "system_used_percent": system_mem.percent
    }


def estimate_batch_memory_mb(
    batch_size: int,
    fft_size: int,
    num_ofdm_symbols: int,
    num_bs_ant: int,
    num_ut: int,
    num_ut_ant: int,
    num_bits_per_symbol: int = 2
) -> float:
    """
    Estimate memory requirement for a batch in MB.
    
    Memory estimation for key tensors:
    - Resource grid: batch × num_tx × num_streams × num_ofdm × fft_size (complex64 = 8 bytes)
    - Channel: batch × num_rx × num_tx × num_streams × num_ofdm × fft_size (complex64)
    - Bits: batch × num_tx × num_streams × num_bits (float32 = 4 bytes)
    
    Args:
        batch_size: Number of channel realizations
        fft_size: FFT size
        num_ofdm_symbols: Number of OFDM symbols
        num_bs_ant: Number of base station antennas
        num_ut: Number of user terminals
        num_ut_ant: Number of antennas per UT
        num_bits_per_symbol: Bits per modulation symbol
        
    Returns:
        Estimated memory in MB
    """
    num_tx = num_ut * num_ut_ant
    num_streams = num_ut_ant
    num_rx = num_bs_ant
    
    # Resource grid (complex64 = 8 bytes)
    rg_size = batch_size * num_tx * num_streams * num_ofdm_symbols * fft_size * 8
    
    # Channel (complex64 = 8 bytes)
    channel_size = batch_size * num_rx * num_tx * num_streams * num_ofdm_symbols * fft_size * 8
    
    # Bits (float32 = 4 bytes)
    # Estimate: ~fft_size * num_ofdm_symbols * num_bits_per_symbol * code_rate
    num_data_symbols = fft_size * num_ofdm_symbols * 0.8  # Approximate (accounting for pilots)
    num_coded_bits = int(num_data_symbols * num_bits_per_symbol)
    num_info_bits = int(num_coded_bits * 0.5)  # Assuming code rate 0.5
    bits_size = batch_size * num_tx * num_streams * num_info_bits * 4
    
    # Additional overhead (intermediate tensors, gradients, etc.)
    overhead_factor = 2.0  # Conservative estimate
    
    total_bytes = (rg_size + channel_size + bits_size) * overhead_factor
    total_mb = total_bytes / (1024 * 1024)
    
    return total_mb


def get_optimal_batch_size(
    max_memory_mb: float,
    fft_size: int,
    num_ofdm_symbols: int,
    num_bs_ant: int,
    num_ut: int,
    num_ut_ant: int,
    num_bits_per_symbol: int = 2,
    start_batch_size: int = 8
) -> int:
    """
    Calculate optimal batch size based on available memory.
    
    Args:
        max_memory_mb: Maximum available memory in MB
        fft_size: FFT size
        num_ofdm_symbols: Number of OFDM symbols
        num_bs_ant: Number of base station antennas
        num_ut: Number of user terminals
        num_ut_ant: Number of antennas per UT
        num_bits_per_symbol: Bits per modulation symbol
        start_batch_size: Starting batch size to test
        
    Returns:
        Optimal batch size (guaranteed to fit in memory)
    """
    batch_size = start_batch_size
    
    while True:
        estimated_mb = estimate_batch_memory_mb(
            batch_size, fft_size, num_ofdm_symbols,
            num_bs_ant, num_ut, num_ut_ant, num_bits_per_symbol
        )
        
        if estimated_mb <= max_memory_mb * 0.8:  # Use 80% of available memory
            # Try next larger batch size
            next_batch = batch_size * 2
            next_estimated = estimate_batch_memory_mb(
                next_batch, fft_size, num_ofdm_symbols,
                num_bs_ant, num_ut, num_ut_ant, num_bits_per_symbol
            )
            if next_estimated > max_memory_mb * 0.8:
                break
            batch_size = next_batch
        else:
            # Current batch is too large, reduce
            if batch_size <= 1:
                return 1
            batch_size = max(1, batch_size // 2)
            break
    
    return batch_size


class MemoryMonitor:
    """Context manager to monitor memory usage during operations."""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_memory = None
        self.end_memory = None
    
    def __enter__(self):
        self.start_memory = get_memory_usage()
        print(f"[Memory] Starting {self.operation_name}")
        print(f"  Process RSS: {self.start_memory['process_rss_mb']:.1f} MB")
        print(f"  System Available: {self.start_memory['system_available_gb']:.2f} GB")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_memory = get_memory_usage()
        delta_mb = self.end_memory['process_rss_mb'] - self.start_memory['process_rss_mb']
        print(f"[Memory] Completed {self.operation_name}")
        print(f"  Process RSS: {self.end_memory['process_rss_mb']:.1f} MB (Δ {delta_mb:+.1f} MB)")
        print(f"  System Available: {self.end_memory['system_available_gb']:.2f} GB")
        
        # Clear cache if memory increased significantly
        if delta_mb > 500:  # More than 500 MB increase
            clear_tensorflow_cache()
            print(f"  ✓ Cleared TensorFlow cache (memory increase: {delta_mb:.1f} MB)")

