#!/usr/bin/env python3
"""
Comprehensive comparison script for channel estimation methods.

This script runs simulations for all available channel estimation methods,
collects comprehensive metrics, and generates detailed comparison visualizations.

Available estimators:
- ls_nn: LS with nearest neighbor interpolation
- ls_lin: LS with linear interpolation
- neural: Neural network-based estimator
- ls_smooth: Smoothed LS estimator (2D smoothing)
- ls_temporal: Temporal EMA estimator

Metrics collected:
- BER (Bit Error Rate)
- BLER (Block Error Rate)
- NMSE (Normalized Mean Squared Error)
- EVM (Error Vector Magnitude)
- SINR (Signal-to-Interference-plus-Noise Ratio)
- Decoder iterations
- Throughput
- Spectral efficiency
- Fairness

Visualizations:
- BER vs Eb/No
- BLER vs Eb/No
- NMSE vs Eb/No
- SINR vs Eb/No
- Decoder iterations vs Eb/No
- Comprehensive comparison dashboard
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import run_simulation, configure_env, setup_gpu


# Available channel estimation methods
ESTIMATORS = {
    "ls_nn": {
        "name": "LS (Nearest Neighbor)",
        "type": "ls_nn",
        "weights": None,
        "kwargs": {},
    },
    "ls_lin": {
        "name": "LS (Linear)",
        "type": "ls_lin",
        "weights": None,
        "kwargs": {},
    },
    "neural": {
        "name": "Neural Network",
        "type": "neural",
        "weights": "artifacts/neural_channel_estimator.weights.h5",
        "kwargs": {"hidden_units": [32, 32]},
    },
    "ls_smooth": {
        "name": "LS (Smoothed)",
        "type": "ls_smooth",
        "weights": None,
        "kwargs": {"kernel_time": 3, "kernel_freq": 5},
    },
    "ls_temporal": {
        "name": "LS (Temporal EMA)",
        "type": "ls_temporal",
        "weights": None,
        "kwargs": {"alpha": 0.7},
    },
}


def run_all_estimators(
    scenario: str = "umi",
    ebno_db_range: Optional[np.ndarray] = None,
    batch_size: int = 128,
    max_mc_iter: int = 1000,
    num_target_block_errors: int = 1000,
    target_bler: float = 1e-3,
    perfect_csi: bool = False,
    output_dir: str = "results",
    estimator_list: Optional[List[str]] = None,
    gpu_num: int = 0,
    force_cpu: bool = False,
) -> Dict[str, Any]:
    """
    Run simulations for all specified channel estimation methods.
    
    Args:
        scenario: Channel scenario ("umi", "uma", "rma")
        ebno_db_range: Array of Eb/No values in dB
        batch_size: Batch size for simulation
        max_mc_iter: Maximum Monte Carlo iterations
        num_target_block_errors: Target number of block errors
        target_bler: Target BLER for early stopping
        perfect_csi: Whether to use perfect CSI (False for channel estimation)
        output_dir: Output directory for results
        estimator_list: List of estimator keys to run (None = all)
        gpu_num: GPU device number
        force_cpu: Force CPU execution
        
    Returns:
        Dictionary containing all results organized by estimator
    """
    if ebno_db_range is None:
        ebno_db_range = np.arange(-3.0, 10.0, 3.0)
    
    if estimator_list is None:
        estimator_list = list(ESTIMATORS.keys())
    
    # Filter to only requested estimators
    estimators_to_run = {k: v for k, v in ESTIMATORS.items() if k in estimator_list}
    
    if not estimators_to_run:
        raise ValueError(f"No valid estimators found. Available: {list(ESTIMATORS.keys())}")
    
    print("=" * 80)
    print("CHANNEL ESTIMATION METHODS COMPARISON")
    print("=" * 80)
    print(f"Scenario: {scenario.upper()}")
    print(f"Eb/No range: {ebno_db_range[0]:.1f} to {ebno_db_range[-1]:.1f} dB")
    print(f"Estimators to compare: {', '.join(estimators_to_run.keys())}")
    print(f"CSI: {'Perfect' if perfect_csi else 'Imperfect'}")
    print("=" * 80)
    
    all_results = {}
    start_time = time.time()
    
    for est_key, est_config in estimators_to_run.items():
        print(f"\n{'='*80}")
        print(f"Running simulation for: {est_config['name']} ({est_key})")
        print(f"{'='*80}")
        
        # Check if neural weights exist
        weights_path = est_config["weights"]
        if weights_path and not Path(weights_path).exists():
            print(f"⚠ Warning: Neural weights not found at {weights_path}")
            print(f"  Skipping {est_key}...")
            continue
        
        try:
            results = run_simulation(
                scenario=scenario,
                perfect_csi_list=[perfect_csi],
                ebno_db_range=ebno_db_range,
                batch_size=batch_size,
                max_mc_iter=max_mc_iter,
                num_target_block_errors=num_target_block_errors,
                target_bler=target_bler,
                save_results=False,  # We'll save combined results
                plot_results=False,  # We'll create comparison plots
                output_dir=output_dir,
                estimator_type=est_config["type"],
                estimator_weights=weights_path,
                estimator_kwargs=est_config["kwargs"],
            )
            
            all_results[est_key] = {
                "config": est_config,
                "results": results,
            }
            print(f"✓ Completed: {est_config['name']}")
            
        except Exception as e:
            print(f"✗ Error running {est_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_duration = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"All simulations completed in {total_duration:.2f} seconds")
    print(f"{'='*80}")
    
    return {
        "scenario": scenario,
        "ebno_db_range": ebno_db_range.tolist(),
        "perfect_csi": perfect_csi,
        "estimators": all_results,
        "total_duration": total_duration,
        "timestamp": datetime.now().isoformat(),
    }


def save_comparison_results(comparison_results: Dict[str, Any], output_dir: str) -> str:
    """Save comparison results to JSON file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    scenario = comparison_results["scenario"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/estimator_comparison_{scenario}_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(comparison_results)
    
    with open(filename, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✓ Comparison results saved to: {filename}")
    return filename


def create_comparison_visualizations(comparison_results: Dict[str, Any], output_dir: str):
    """Create comprehensive comparison visualizations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    scenario = comparison_results["scenario"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    estimators_data = comparison_results["estimators"]
    
    if not estimators_data:
        print("⚠ No data to visualize")
        return
    
    # Extract data for plotting
    plot_data = {}
    for est_key, est_data in estimators_data.items():
        results = est_data["results"]
        est_name = est_data["config"]["name"]
        
        # Get metrics from the first run (imperfect CSI)
        runs = results.get("runs", [])
        if not runs:
            continue
        
        # Find imperfect CSI run
        run_data = None
        for run in runs:
            if not run.get("perfect_csi", True):
                run_data = run
                break
        
        if run_data is None:
            continue
        
        metrics = run_data.get("metrics", [])
        if not metrics:
            continue
        
        ebno = [m["ebno_db"] for m in metrics]
        ber = [m["overall"].get("ber") for m in metrics]
        bler = [m["overall"].get("bler") for m in metrics]
        nmse_db = [m["overall"].get("nmse_db") for m in metrics]
        sinr_db = [m["overall"].get("sinr_db") for m in metrics]
        decoder_iter = [m["overall"].get("decoder_iter_avg") for m in metrics]
        evm_percent = [m["overall"].get("evm_percent") for m in metrics]
        
        plot_data[est_key] = {
            "name": est_name,
            "ebno": ebno,
            "ber": ber,
            "bler": bler,
            "nmse_db": nmse_db,
            "sinr_db": sinr_db,
            "decoder_iter": decoder_iter,
            "evm_percent": evm_percent,
        }
    
    if not plot_data:
        print("⚠ No valid data for visualization")
        return
    
    # Color and style mapping
    colors = {
        "ls_nn": "#1f77b4",      # blue
        "ls_lin": "#ff7f0e",      # orange
        "neural": "#2ca02c",      # green
        "ls_smooth": "#d62728",   # red
        "ls_temporal": "#9467bd", # purple
    }
    
    markers = {
        "ls_nn": "o",
        "ls_lin": "s",
        "neural": "^",
        "ls_smooth": "D",
        "ls_temporal": "v",
    }
    
    linestyles = {
        "ls_nn": "-",
        "ls_lin": "--",
        "neural": "-.",
        "ls_smooth": ":",
        "ls_temporal": "-",
    }
    
    # 1. BER and BLER comparison (side by side)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for est_key, data in plot_data.items():
        ebno = data["ebno"]
        ber = data["ber"]
        bler = data["bler"]
        
        # Filter out None values
        valid_indices = [i for i, (b, bl) in enumerate(zip(ber, bler)) if b is not None and bl is not None]
        if not valid_indices:
            continue
        
        ebno_valid = [ebno[i] for i in valid_indices]
        ber_valid = [ber[i] for i in valid_indices]
        bler_valid = [bler[i] for i in valid_indices]
        
        color = colors.get(est_key, "gray")
        marker = markers.get(est_key, "o")
        linestyle = linestyles.get(est_key, "-")
        
        ax1.semilogy(
            ebno_valid, ber_valid,
            color=color, marker=marker, linestyle=linestyle,
            label=data["name"], linewidth=2, markersize=8
        )
        ax2.semilogy(
            ebno_valid, bler_valid,
            color=color, marker=marker, linestyle=linestyle,
            label=data["name"], linewidth=2, markersize=8
        )
    
    ax1.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
    ax1.set_ylabel("BER", fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_title(f"Bit Error Rate Comparison - {scenario.upper()}", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim([1e-6, 1])
    
    ax2.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
    ax2.set_ylabel("BLER", fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_title(f"Block Error Rate Comparison - {scenario.upper()}", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim([1e-6, 1])
    
    plt.tight_layout()
    filename1 = f"{output_dir}/comparison_ber_bler_{scenario}_{timestamp}.png"
    plt.savefig(filename1, dpi=300, bbox_inches='tight')
    plt.savefig(filename1.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved: {filename1}")
    plt.close()
    
    # 2. NMSE and SINR comparison
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for est_key, data in plot_data.items():
        ebno = data["ebno"]
        nmse_db = data["nmse_db"]
        sinr_db = data["sinr_db"]
        
        # Filter valid values
        valid_nmse = [(i, v) for i, v in enumerate(nmse_db) if v is not None]
        valid_sinr = [(i, v) for i, v in enumerate(sinr_db) if v is not None and np.isfinite(v)]
        
        color = colors.get(est_key, "gray")
        marker = markers.get(est_key, "o")
        linestyle = linestyles.get(est_key, "-")
        
        if valid_nmse:
            ebno_nmse = [ebno[i] for i, _ in valid_nmse]
            nmse_valid = [v for _, v in valid_nmse]
            ax1.plot(
                ebno_nmse, nmse_valid,
                color=color, marker=marker, linestyle=linestyle,
                label=data["name"], linewidth=2, markersize=8
            )
        
        if valid_sinr:
            ebno_sinr = [ebno[i] for i, _ in valid_sinr]
            sinr_valid = [v for _, v in valid_sinr]
            ax2.plot(
                ebno_sinr, sinr_valid,
                color=color, marker=marker, linestyle=linestyle,
                label=data["name"], linewidth=2, markersize=8
            )
    
    ax1.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
    ax1.set_ylabel("NMSE (dB)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Channel Estimation NMSE - {scenario.upper()}", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.invert_yaxis()  # Lower NMSE is better
    
    ax2.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
    ax2.set_ylabel("SINR (dB)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"Signal-to-Interference-plus-Noise Ratio - {scenario.upper()}", fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    filename2 = f"{output_dir}/comparison_nmse_sinr_{scenario}_{timestamp}.png"
    plt.savefig(filename2, dpi=300, bbox_inches='tight')
    plt.savefig(filename2.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved: {filename2}")
    plt.close()
    
    # 3. Decoder iterations and EVM
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for est_key, data in plot_data.items():
        ebno = data["ebno"]
        decoder_iter = data["decoder_iter"]
        evm_percent = data["evm_percent"]
        
        valid_iter = [(i, v) for i, v in enumerate(decoder_iter) if v is not None]
        valid_evm = [(i, v) for i, v in enumerate(evm_percent) if v is not None]
        
        color = colors.get(est_key, "gray")
        marker = markers.get(est_key, "o")
        linestyle = linestyles.get(est_key, "-")
        
        if valid_iter:
            ebno_iter = [ebno[i] for i, _ in valid_iter]
            iter_valid = [v for _, v in valid_iter]
            ax1.plot(
                ebno_iter, iter_valid,
                color=color, marker=marker, linestyle=linestyle,
                label=data["name"], linewidth=2, markersize=8
            )
        
        if valid_evm:
            ebno_evm = [ebno[i] for i, _ in valid_evm]
            evm_valid = [v for _, v in valid_evm]
            ax2.plot(
                ebno_evm, evm_valid,
                color=color, marker=marker, linestyle=linestyle,
                label=data["name"], linewidth=2, markersize=8
            )
    
    ax1.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
    ax1.set_ylabel("Avg Decoder Iterations", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"LDPC Decoder Iterations - {scenario.upper()}", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    
    ax2.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
    ax2.set_ylabel("EVM (%)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f"Error Vector Magnitude - {scenario.upper()}", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    filename3 = f"{output_dir}/comparison_decoder_evm_{scenario}_{timestamp}.png"
    plt.savefig(filename3, dpi=300, bbox_inches='tight')
    plt.savefig(filename3.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved: {filename3}")
    plt.close()
    
    # 4. Comprehensive dashboard (all metrics in one figure)
    fig4 = plt.figure(figsize=(20, 12))
    gs = fig4.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    axes = [
        fig4.add_subplot(gs[0, 0]),  # BER
        fig4.add_subplot(gs[0, 1]),  # BLER
        fig4.add_subplot(gs[0, 2]),  # NMSE
        fig4.add_subplot(gs[1, 0]),  # SINR
        fig4.add_subplot(gs[1, 1]),  # Decoder Iter
        fig4.add_subplot(gs[1, 2]),  # EVM
        fig4.add_subplot(gs[2, :]),   # Combined BER/BLER
    ]
    
    for est_key, data in plot_data.items():
        ebno = data["ebno"]
        color = colors.get(est_key, "gray")
        marker = markers.get(est_key, "o")
        linestyle = linestyles.get(est_key, "-")
        label = data["name"]
        
        # BER
        valid_ber = [(i, v) for i, v in enumerate(data["ber"]) if v is not None]
        if valid_ber:
            ebno_v = [ebno[i] for i, _ in valid_ber]
            ber_v = [v for _, v in valid_ber]
            axes[0].semilogy(ebno_v, ber_v, color=color, marker=marker, linestyle=linestyle,
                            label=label, linewidth=2, markersize=6)
        
        # BLER
        valid_bler = [(i, v) for i, v in enumerate(data["bler"]) if v is not None]
        if valid_bler:
            ebno_v = [ebno[i] for i, _ in valid_bler]
            bler_v = [v for _, v in valid_bler]
            axes[1].semilogy(ebno_v, bler_v, color=color, marker=marker, linestyle=linestyle,
                            label=label, linewidth=2, markersize=6)
        
        # NMSE
        valid_nmse = [(i, v) for i, v in enumerate(data["nmse_db"]) if v is not None]
        if valid_nmse:
            ebno_v = [ebno[i] for i, _ in valid_nmse]
            nmse_v = [v for _, v in valid_nmse]
            axes[2].plot(ebno_v, nmse_v, color=color, marker=marker, linestyle=linestyle,
                        label=label, linewidth=2, markersize=6)
        
        # SINR
        valid_sinr = [(i, v) for i, v in enumerate(data["sinr_db"]) if v is not None and np.isfinite(v)]
        if valid_sinr:
            ebno_v = [ebno[i] for i, _ in valid_sinr]
            sinr_v = [v for _, v in valid_sinr]
            axes[3].plot(ebno_v, sinr_v, color=color, marker=marker, linestyle=linestyle,
                        label=label, linewidth=2, markersize=6)
        
        # Decoder Iter
        valid_iter = [(i, v) for i, v in enumerate(data["decoder_iter"]) if v is not None]
        if valid_iter:
            ebno_v = [ebno[i] for i, _ in valid_iter]
            iter_v = [v for _, v in valid_iter]
            axes[4].plot(ebno_v, iter_v, color=color, marker=marker, linestyle=linestyle,
                        label=label, linewidth=2, markersize=6)
        
        # EVM
        valid_evm = [(i, v) for i, v in enumerate(data["evm_percent"]) if v is not None]
        if valid_evm:
            ebno_v = [ebno[i] for i, _ in valid_evm]
            evm_v = [v for _, v in valid_evm]
            axes[5].plot(ebno_v, evm_v, color=color, marker=marker, linestyle=linestyle,
                        label=label, linewidth=2, markersize=6)
        
        # Combined BER/BLER
        if valid_ber:
            axes[6].semilogy(ebno_v, ber_v, color=color, marker=marker, linestyle=linestyle,
                           label=f"{label} (BER)", linewidth=2, markersize=6, alpha=0.7)
        if valid_bler:
            ebno_v = [ebno[i] for i, _ in valid_bler]
            bler_v = [v for _, v in valid_bler]
            axes[6].semilogy(ebno_v, bler_v, color=color, marker=marker, linestyle='--',
                           label=f"{label} (BLER)", linewidth=2, markersize=6, alpha=0.7)
    
    # Configure subplots
    axes[0].set_xlabel(r"$E_b/N_0$ (dB)")
    axes[0].set_ylabel("BER")
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Bit Error Rate", fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].set_ylim([1e-6, 1])
    
    axes[1].set_xlabel(r"$E_b/N_0$ (dB)")
    axes[1].set_ylabel("BLER")
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Block Error Rate", fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].set_ylim([1e-6, 1])
    
    axes[2].set_xlabel(r"$E_b/N_0$ (dB)")
    axes[2].set_ylabel("NMSE (dB)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Channel Estimation NMSE", fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].invert_yaxis()
    
    axes[3].set_xlabel(r"$E_b/N_0$ (dB)")
    axes[3].set_ylabel("SINR (dB)")
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title("Signal-to-Interference-plus-Noise Ratio", fontweight='bold')
    axes[3].legend(fontsize=8)
    
    axes[4].set_xlabel(r"$E_b/N_0$ (dB)")
    axes[4].set_ylabel("Avg Decoder Iterations")
    axes[4].grid(True, alpha=0.3)
    axes[4].set_title("LDPC Decoder Iterations", fontweight='bold')
    axes[4].legend(fontsize=8)
    
    axes[5].set_xlabel(r"$E_b/N_0$ (dB)")
    axes[5].set_ylabel("EVM (%)")
    axes[5].grid(True, alpha=0.3)
    axes[5].set_title("Error Vector Magnitude", fontweight='bold')
    axes[5].legend(fontsize=8)
    
    axes[6].set_xlabel(r"$E_b/N_0$ (dB)")
    axes[6].set_ylabel("Error Rate")
    axes[6].set_yscale('log')
    axes[6].grid(True, alpha=0.3)
    axes[6].set_title(f"BER/BLER Comparison - {scenario.upper()}", fontweight='bold')
    axes[6].legend(fontsize=8, ncol=2)
    axes[6].set_ylim([1e-6, 1])
    
    plt.suptitle(f"Channel Estimation Methods Comparison - {scenario.upper()}", 
                fontsize=16, fontweight='bold', y=0.995)
    
    filename4 = f"{output_dir}/comparison_dashboard_{scenario}_{timestamp}.png"
    plt.savefig(filename4, dpi=300, bbox_inches='tight')
    plt.savefig(filename4.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✓ Saved: {filename4}")
    plt.close()
    
    print(f"\n✓ All visualizations saved to {output_dir}/")


def print_comparison_summary(comparison_results: Dict[str, Any]):
    """Print a summary table comparing all estimators."""
    print("\n" + "=" * 100)
    print("CHANNEL ESTIMATION METHODS COMPARISON SUMMARY")
    print("=" * 100)
    print(f"Scenario: {comparison_results['scenario'].upper()}")
    print(f"CSI: {'Perfect' if comparison_results['perfect_csi'] else 'Imperfect'}")
    print("=" * 100)
    
    estimators_data = comparison_results["estimators"]
    if not estimators_data:
        print("No results available.")
        return
    
    # Collect all Eb/No values
    all_ebno = set()
    for est_data in estimators_data.values():
        results = est_data["results"]
        for run in results.get("runs", []):
            if not run.get("perfect_csi", True):
                for metric in run.get("metrics", []):
                    all_ebno.add(metric.get("ebno_db"))
    
    ebno_list = sorted(all_ebno)
    
    # Print header
    print(f"\n{'Estimator':<20} | {'Eb/No':>8} | {'BER':>12} | {'BLER':>12} | {'NMSE (dB)':>12} | {'SINR (dB)':>12}")
    print("-" * 100)
    
    # Print data for each estimator at each Eb/No
    for est_key, est_data in estimators_data.items():
        est_name = est_data["config"]["name"]
        results = est_data["results"]
        
        # Find imperfect CSI run
        run_data = None
        for run in results.get("runs", []):
            if not run.get("perfect_csi", True):
                run_data = run
                break
        
        if run_data is None:
            continue
        
        metrics_dict = {m["ebno_db"]: m for m in run_data.get("metrics", [])}
        
        for ebno in ebno_list:
            metric = metrics_dict.get(ebno)
            if metric is None:
                continue
            
            overall = metric.get("overall", {})
            ber = overall.get("ber")
            bler = overall.get("bler")
            nmse_db = overall.get("nmse_db")
            sinr_db = overall.get("sinr_db")
            
            print(
                f"{est_name:<20} | {ebno:>8.1f} | "
                f"{(ber if ber is not None else float('nan')):>12.3e} | "
                f"{(bler if bler is not None else float('nan')):>12.3e} | "
                f"{(nmse_db if nmse_db is not None else float('nan')):>12.3f} | "
                f"{(sinr_db if sinr_db is not None and np.isfinite(sinr_db) else float('nan')):>12.3f}"
            )
    
    print("=" * 100)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare all channel estimation methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        default='umi',
        choices=['umi', 'uma', 'rma'],
        help='Channel scenario (default: umi)'
    )
    
    parser.add_argument(
        '--estimators',
        nargs='+',
        default=None,
        choices=list(ESTIMATORS.keys()),
        help=f'Estimators to compare (default: all). Available: {", ".join(ESTIMATORS.keys())}'
    )
    
    parser.add_argument(
        '--ebno-min',
        type=float,
        default=-3.0,
        help='Minimum Eb/No in dB (default: -3.0)'
    )
    
    parser.add_argument(
        '--ebno-max',
        type=float,
        default=9.0,
        help='Maximum Eb/No in dB (default: 9.0)'
    )
    
    parser.add_argument(
        '--ebno-step',
        type=float,
        default=3.0,
        help='Eb/No step size in dB (default: 3.0)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for simulation (default: 128)'
    )
    
    parser.add_argument(
        '--max-iter',
        type=int,
        default=1000,
        help='Maximum Monte Carlo iterations (default: 1000)'
    )
    
    parser.add_argument(
        '--target-block-errors',
        type=int,
        default=1000,
        help='Target number of block errors (default: 1000)'
    )
    
    parser.add_argument(
        '--target-bler',
        type=float,
        default=1e-3,
        help='Target BLER for early stopping (default: 1e-3)'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device number (default: 0)'
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU execution'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Configure environment
    configure_env(force_cpu=args.cpu, gpu_num=args.gpu)
    
    # Import TensorFlow after environment configuration
    import tensorflow as tf
    import numpy as np
    tf.get_logger().setLevel('ERROR')
    setup_gpu(args.gpu)
    
    # Set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Build Eb/No range
    ebno_db_range = np.arange(args.ebno_min, args.ebno_max + args.ebno_step, args.ebno_step)
    
    # Run comparisons
    comparison_results = run_all_estimators(
        scenario=args.scenario,
        ebno_db_range=ebno_db_range,
        batch_size=args.batch_size,
        max_mc_iter=args.max_iter,
        num_target_block_errors=args.target_block_errors,
        target_bler=args.target_bler,
        perfect_csi=False,  # Compare channel estimation methods
        output_dir=args.output_dir,
        estimator_list=args.estimators,
        gpu_num=args.gpu,
        force_cpu=args.cpu,
    )
    
    # Save results
    save_comparison_results(comparison_results, args.output_dir)
    
    # Create visualizations
    create_comparison_visualizations(comparison_results, args.output_dir)
    
    # Print summary
    print_comparison_summary(comparison_results)
    
    return comparison_results


if __name__ == "__main__":
    main()

