#!/usr/bin/env python3
"""
Run simulation with 6G-compliant parameters.

This script loads the 6G-compliant parameters from min_6g_params_config.json
and runs a full BER/BLER simulation with memory management.
Produces matrices and plots for each metric, organized per run.
"""

import sys
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.components.config import SystemConfig
from src.utils.memory_manager import (
    configure_tensorflow_memory,
    clear_tensorflow_cache,
    get_memory_usage,
    estimate_batch_memory_mb,
    get_optimal_batch_size,
    MemoryMonitor
)
from src.sim.runner import run_simulation
from src.sim.results import load_baseline_results
from src.sim.plotting import save_metric_matrices_and_plots
from src.utils.env import configure_env


def main():
    # Load 6G-compliant parameters
    config_path = project_root / "results" / "min_6g_params_config.json"
    with open(config_path, 'r') as f:
        params = json.load(f)
    
    print("=" * 80)
    print("6G-Compliant Simulation with Memory Management")
    print("=" * 80)
    print(f"Loading parameters from: {config_path}")
    print(json.dumps(params, indent=2))
    print()
    
    # Configure environment BEFORE importing TensorFlow
    # Don't force CPU - we want GPU
    configure_env(force_cpu=False, gpu_num=0)  # Use GPU 0
    
    # Configure TensorFlow memory settings with GPU enabled
    print("Configuring TensorFlow memory settings and GPU...")
    gpu_available = configure_tensorflow_memory(
        memory_growth=True,
        memory_limit_mb=None,  # No hard limit, use growth
        cpu_memory_limit_mb=None,
        enable_gpu=True  # Enable GPU
    )
    print()
    
    if not gpu_available:
        print("⚠ Warning: GPU not available. Simulation will use CPU (slower).")
        print()
        print("  To fix GPU issues, try:")
        print("  1. Install TensorFlow with CUDA support:")
        print("     pip install --upgrade tensorflow[and-cuda]")
        print("  2. Or install matching CUDA/cuDNN versions manually")
        print("  3. Run GPU check: python scripts/gpu/check_gpu.py")
        print("  4. See docs/gpu/01_setup.md for detailed instructions")
        print()
        print("  Continuing with CPU mode (slower but functional)...")
        print()
    
    # Get system memory info
    mem_info = get_memory_usage()
    print("System Memory Status:")
    print(f"  - System Total: {mem_info['system_total_gb']:.2f} GB")
    print(f"  - System Available: {mem_info['system_available_gb']:.2f} GB")
    print(f"  - System Used: {mem_info['system_used_percent']:.1f}%")
    print(f"  - Process RSS: {mem_info['process_rss_mb']:.1f} MB")
    print()
    
    # Create SystemConfig with 6G-compliant parameters
    config = SystemConfig(
        fft_size=params['fft_size'],
        num_bs_ant=params['num_bs_ant'],
        num_ut=params['num_ut'],
        num_ut_ant=params['num_ut_ant'],
        num_ofdm_symbols=params['num_ofdm_symbols'],
        num_bits_per_symbol=2,  # QPSK
        coderate=0.5
    )
    
    print("System Configuration:")
    print(f"  - FFT Size: {config.fft_size}")
    print(f"  - BS Antennas: {config.num_bs_ant}")
    print(f"  - User Terminals: {config.num_ut}")
    print(f"  - UT Antennas: {config.num_ut_ant}")
    print(f"  - OFDM Symbols: {config.num_ofdm_symbols}")
    print(f"  - Total TX Antennas: {config.num_tx}")
    print(f"  - Streams per TX: {config.num_streams_per_tx}")
    print()
    
    # Estimate memory requirements
    requested_batch_size = params['batch_size']
    estimated_memory = estimate_batch_memory_mb(
        requested_batch_size,
        config.fft_size,
        config.num_ofdm_symbols,
        config.num_bs_ant,
        config.num_ut,
        config.num_ut_ant,
        config.num_bits_per_symbol
    )
    
    print("Memory Estimation:")
    print(f"  - Requested batch size: {requested_batch_size}")
    print(f"  - Estimated memory per batch: {estimated_memory:.1f} MB")
    print(f"  - Available memory: {mem_info['system_available_gb'] * 1024:.1f} MB")
    print()
    
    # Calculate optimal batch size
    # Use 50% of available memory as safety margin
    max_memory_mb = mem_info['system_available_gb'] * 1024 * 0.5
    optimal_batch_size = get_optimal_batch_size(
        max_memory_mb,
        config.fft_size,
        config.num_ofdm_symbols,
        config.num_bs_ant,
        config.num_ut,
        config.num_ut_ant,
        config.num_bits_per_symbol,
        start_batch_size=requested_batch_size
    )
    
    if optimal_batch_size < requested_batch_size:
        print(f"⚠ Warning: Reducing batch size from {requested_batch_size} to {optimal_batch_size}")
        print(f"  Reason: Memory constraints (estimated {estimated_memory:.1f} MB > available {max_memory_mb:.1f} MB)")
        batch_size = optimal_batch_size
    else:
        batch_size = requested_batch_size
        print(f"✓ Batch size {batch_size} is within memory limits")
    
    print()
    
    # Clear cache before starting
    clear_tensorflow_cache()
    
    # Create run-specific directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = project_root / "results" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {run_dir}")
    print()
    
    # Run simulation with memory monitoring
    ebno_db_range = np.arange(-5.0, 11.0, 2.0)
    
    with MemoryMonitor("6G Simulation"):
        try:
            results = run_simulation(
                scenario='umi',
                perfect_csi_list=[False, True],  # Both imperfect and perfect CSI
                ebno_db_range=ebno_db_range,
                batch_size=batch_size,
                max_mc_iter=1000,
                num_target_block_errors=1000,
                target_bler=1e-3,
                config=config,
                save_results=True,
                plot_results=True,
                output_dir=str(run_dir),
                estimator_type='ls',
                profile_name='6G_Compliant'
            )
            
            # Save results JSON to run directory
            results_file = run_dir / "simulation_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✓ Results JSON saved to: {results_file}")
            
            # Load baseline results for comparison
            print()
            print("=" * 80)
            print("Loading baseline results for comparison...")
            print("=" * 80)
            baseline_results = load_baseline_results(project_root=project_root)
            print()
            
            # Save matrices and create plots for each metric
            print("=" * 80)
            print("Saving metric matrices and plots...")
            print("=" * 80)
            save_metric_matrices_and_plots(results, run_dir, baseline_results)
            
        finally:
            # Clean up memory after simulation
            clear_tensorflow_cache()
            gc.collect()
    
    print()
    print("=" * 80)
    print("Simulation completed!")
    print(f"All results saved to: {run_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()

