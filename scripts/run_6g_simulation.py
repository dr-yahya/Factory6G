#!/usr/bin/env python3
"""
Run simulation with 6G-compliant parameters.

This script loads the 6G-compliant parameters from max_params_config.json
and runs a full BER/BLER simulation with memory management.
Produces matrices and plots for each metric, organized per run.
"""

import sys
import json
import gc
from pathlib import Path
from datetime import datetime
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
from main import run_simulation, configure_env


def save_metric_matrices_and_plots(results: dict, run_dir: Path):
    """
    Save matrices (numpy arrays) and plots for each metric, organized per run.
    
    Args:
        results: Simulation results dictionary
        run_dir: Directory to save results for this run
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir = run_dir / "matrices"
    plots_dir = run_dir / "plots"
    matrices_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    runs = results.get("runs", [])
    
    # Metrics to process
    per_stream_metrics = ["ber", "bler", "throughput_bits", "decoder_iter_avg", "sinr_db", 
                          "snr_db", "channel_capacity", "outage_probability"]
    overall_metrics = ["ber", "bler", "nmse_db", "evm_percent", "sinr_db", "snr_db",
                       "decoder_iter_avg", "throughput_bits", "spectral_efficiency", "fairness_jain",
                       "channel_capacity", "outage_probability", "air_interface_latency_ms", "energy_per_bit_pj"]
    
    # Collect data for each metric across all runs and Eb/No points
    for run_idx, run in enumerate(runs):
        csi_str = "perfect" if run.get("perfect_csi") else "imperfect"
        metrics_list = run.get("metrics", [])
        
        if not metrics_list:
            continue
        
        # Extract Eb/No values
        ebno_values = [m["ebno_db"] for m in metrics_list]
        
        # Process per-stream metrics (matrices: [num_ebno, num_streams])
        for metric_name in per_stream_metrics:
            metric_data = []
            for metric_entry in metrics_list:
                per_stream = metric_entry.get("per_stream", {})
                data = per_stream.get(metric_name)
                if data is not None:
                    metric_data.append(np.array(data))
                else:
                    # Fill with NaN if missing - find num_streams from first valid entry
                    prev_data = next((m.get("per_stream", {}).get(metric_name) for m in metrics_list if m.get("per_stream", {}).get(metric_name) is not None), None)
                    if prev_data is not None:
                        num_streams = len(prev_data)
                        metric_data.append(np.full(num_streams, np.nan))
                    else:
                        continue
            
            if metric_data:
                # Stack into matrix: [num_ebno, num_streams]
                matrix = np.stack(metric_data)
                matrix_file = matrices_dir / f"{metric_name}_per_stream_{csi_str}_run{run_idx}.npy"
                np.save(matrix_file, matrix)
                print(f"  ✓ Saved matrix: {matrix_file.name} (shape: {matrix.shape})")
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                for stream_idx in range(matrix.shape[1]):
                    ax.plot(ebno_values, matrix[:, stream_idx], 
                           marker='o', label=f'Stream {stream_idx}', linewidth=2)
                ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
                ylabel = metric_name.upper().replace('_', ' ')
                # Add units to ylabel for specific metrics
                if metric_name == "sinr_db" or metric_name == "snr_db":
                    ylabel += " (dB)"
                elif metric_name == "channel_capacity":
                    ylabel += " (bits/s/Hz)"
                elif metric_name == "outage_probability":
                    ylabel += " (probability)"
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_title(f"{ylabel} per Stream - {csi_str.upper()} CSI", fontsize=14)
                ax.legend(ncol=2, fontsize=9)
                ax.grid(True, alpha=0.3)
                if metric_name in ["ber", "bler", "outage_probability"]:
                    ax.set_yscale('log')
                plt.tight_layout()
                plot_file = plots_dir / f"{metric_name}_per_stream_{csi_str}_run{run_idx}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved plot: {plot_file.name}")
        
        # Process overall metrics (vectors: [num_ebno])
        for metric_name in overall_metrics:
            metric_values = []
            for metric_entry in metrics_list:
                overall = metric_entry.get("overall", {})
                value = overall.get(metric_name)
                metric_values.append(value if value is not None else np.nan)
            
            if metric_values:
                vector = np.array(metric_values)
                vector_file = matrices_dir / f"{metric_name}_overall_{csi_str}_run{run_idx}.npy"
                np.save(vector_file, vector)
                print(f"  ✓ Saved vector: {vector_file.name} (shape: {vector.shape})")
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(ebno_values, vector, marker='o', linewidth=2, markersize=8, color='blue')
                ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
                ylabel = metric_name.upper().replace('_', ' ')
                # Add units to ylabel for specific metrics
                if metric_name == "sinr_db" or metric_name == "snr_db":
                    ylabel += " (dB)"
                elif metric_name == "channel_capacity":
                    ylabel += " (bits/s/Hz)"
                elif metric_name == "outage_probability":
                    ylabel += " (probability)"
                elif metric_name == "air_interface_latency_ms":
                    ylabel += " (ms)"
                elif metric_name == "energy_per_bit_pj":
                    ylabel += " (pJ)"
                elif metric_name == "spectral_efficiency":
                    ylabel += " (bits/s/Hz)"
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_title(f"{ylabel} - {csi_str.upper()} CSI", fontsize=14)
                ax.grid(True, alpha=0.3)
                if metric_name in ["ber", "bler", "outage_probability"]:
                    ax.set_yscale('log')
                # Add 6G target line for critical metrics
                if metric_name == "air_interface_latency_ms":
                    ax.axhline(y=0.1, color='r', linestyle='--', linewidth=2, label='6G Target (< 0.1 ms)')
                    ax.legend(fontsize=10)
                elif metric_name == "energy_per_bit_pj":
                    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='6G Target (1 pJ/bit)')
                    ax.legend(fontsize=10)
                elif metric_name == "outage_probability":
                    ax.axhline(y=1e-6, color='r', linestyle='--', linewidth=2, label='6G Target (< 1e-6)')
                    ax.legend(fontsize=10)
                plt.tight_layout()
                plot_file = plots_dir / f"{metric_name}_overall_{csi_str}_run{run_idx}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved plot: {plot_file.name}")
    
    # Create comparison plots (both CSI conditions on same plot)
    for metric_name in overall_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False
        for run_idx, run in enumerate(runs):
            csi_str = "Perfect" if run.get("perfect_csi") else "Imperfect"
            metrics_list = run.get("metrics", [])
            if not metrics_list:
                continue
            
            ebno_values = [m["ebno_db"] for m in metrics_list]
            metric_values = []
            for metric_entry in metrics_list:
                overall = metric_entry.get("overall", {})
                value = overall.get(metric_name)
                metric_values.append(value if value is not None else np.nan)
            
            if metric_values and not all(np.isnan(metric_values)):
                ax.plot(ebno_values, metric_values, marker='o', linewidth=2, 
                       markersize=8, label=f"{csi_str} CSI")
                has_data = True
        
        if has_data:
            ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
            ylabel = metric_name.upper().replace('_', ' ')
            # Add units to ylabel
            if metric_name == "sinr_db" or metric_name == "snr_db":
                ylabel += " (dB)"
            elif metric_name == "channel_capacity":
                ylabel += " (bits/s/Hz)"
            elif metric_name == "outage_probability":
                ylabel += " (probability)"
            elif metric_name == "air_interface_latency_ms":
                ylabel += " (ms)"
            elif metric_name == "energy_per_bit_pj":
                ylabel += " (pJ)"
            elif metric_name == "spectral_efficiency":
                ylabel += " (bits/s/Hz)"
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f"{ylabel} Comparison", fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            if metric_name in ["ber", "bler", "outage_probability"]:
                ax.set_yscale('log')
            # Add 6G target line for critical metrics
            if metric_name == "air_interface_latency_ms":
                ax.axhline(y=0.1, color='r', linestyle='--', linewidth=2, label='6G Target (< 0.1 ms)')
                ax.legend(fontsize=10)
            elif metric_name == "energy_per_bit_pj":
                ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='6G Target (1 pJ/bit)')
                ax.legend(fontsize=10)
            elif metric_name == "outage_probability":
                ax.axhline(y=1e-6, color='r', linestyle='--', linewidth=2, label='6G Target (< 1e-6)')
                ax.legend(fontsize=10)
            plt.tight_layout()
            plot_file = plots_dir / f"{metric_name}_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved comparison plot: {plot_file.name}")


def main():
    # Load 6G-compliant parameters
    config_path = project_root / "results" / "max_params_config.json"
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
        print("  3. Run GPU check: python scripts/check_gpu.py")
        print("  4. See docs/GPU_SETUP.md for detailed instructions")
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
            
            # Save matrices and create plots for each metric
            print()
            print("=" * 80)
            print("Saving metric matrices and plots...")
            print("=" * 80)
            save_metric_matrices_and_plots(results, run_dir)
            
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

