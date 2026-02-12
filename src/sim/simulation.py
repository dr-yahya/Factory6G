"""
Unified simulation loop for 6G Factory benchmarks.
Handles setup, execution, result saving, and plotting for:
1. Channel Estimator comparisons
2. Resource Manager comparisons
"""

import time
import gc
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

from src.models.model import Model
from src.sim.results import save_simulation_results, save_results_as_csv
from src.sim.plotting import plot_simulation_results

# ---------------------------------------------------------------------------
# Core Simulation Loop
# ---------------------------------------------------------------------------

def run_simulation_campaign(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for running a simulation campaign.
    
    The campaign type is determined by the config:
    - If `estimators` list is present: Compare Channel Estimators.
    - If `resource_managers` list is present: Compare Resource Managers.
    
    Args:
        config: Configuration dictionary containing simulation parameters.
        
    Returns:
        Dictionary containing all results.
    """
    print("=" * 70)
    print("  6G Smart Factory Simulation Campaign")
    print("=" * 70)
    
    # Defaults
    output_dir = config.get("output_dir", "results/simulation_campaign")
    ebno_db_range = np.array(config.get("ebno_db_range", [0, 5, 10, 15, 20]))
    batch_size = config.get("batch_size", 32)
    total_batches = config.get("total_batches", 10)
    system_config = config.get("system_config", {})
    
    # Determine mode
    if "estimators" in config:
        mode = "estimator_comparison"
        items_to_compare = config["estimators"]
        print(f"Mode: Channel Estimator Comparison")
        print(f"Items: {items_to_compare}")
    elif "resource_managers" in config:
        mode = "resource_manager_comparison"
        items_to_compare = config["resource_managers"]
        print(f"Mode: Resource Manager Comparison")
        print(f"Items: {items_to_compare}")
    else:
        raise ValueError("Config must specify either 'estimators' or 'resource_managers' list.")
        
    print(f"Eb/No Range: {ebno_db_range} dB")
    print(f"Batch Size: {batch_size}, Total Batches: {total_batches}")
    print("-" * 70)

    # Storage for aggregated results
    # Structure: { item_name: { "ber": [], "throughput": [], "latency": [] } }
    aggregated_results = {item: {"ber": [], "throughput": [], "latency": []} for item in items_to_compare}
    
    # --- Main Loop ---
    # We iterate Eb/No first, then items? Or items then Eb/No?
    # iterating Eb/No outer allows easier progress tracking per SNR point.
    
    for ebno_db in ebno_db_range:
        print(f"\nTarget Eb/No: {ebno_db:.1f} dB")
        
        for item_name in items_to_compare:
            print(f"  Running {item_name:20s} ... ", end="", flush=True)
            
            # Setup Model based on mode
            tf.keras.backend.clear_session()
            gc.collect()
            
            try:
                model = _build_model(mode, item_name, system_config, config)
                
                # Run batches
                metrics = _run_batches(model, batch_size, total_batches, float(ebno_db))
                
                # Store metrics
                aggregated_results[item_name]["ber"].append(metrics["ber"])
                aggregated_results[item_name]["throughput"].append(metrics["throughput"])
                aggregated_results[item_name]["latency"].append(metrics["latency"])
                
                print(f"BER: {metrics['ber']:.2e}, Thr: {metrics['throughput']:.2f}, Lat: {metrics['latency']*1000:.2f}ms")
                
                del model
                
            except Exception as e:
                print(f"FAILED: {e}")
                # Append NaNs or defaults to keep list lengths consistent
                aggregated_results[item_name]["ber"].append(1.0)
                aggregated_results[item_name]["throughput"].append(0.0)
                aggregated_results[item_name]["latency"].append(0.0)
                import traceback
                traceback.print_exc()

    # --- Save Results ---
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save as JSON (Raw data)
    full_results = {
        "config": config,
        "results": aggregated_results,
        "ebno_db_range": ebno_db_range.tolist()
    }
    json_path = save_simulation_results(full_results, output_dir)
    
    # 2. Save as CSV (Tabular data for easy reading)
    csv_path = save_results_as_csv(full_results, output_dir)
    
    # --- Generate Plots ---
    if config.get("plot_results", True):
        _plot_campaign_results(full_results, output_dir, mode)
        
    return full_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(mode: str, item_name: str, system_config: dict, global_config: dict) -> Model:
    """Factory to create the Model instance based on configuration."""
    
    # Default base config
    base_config = system_config.copy()
    
    if mode == "estimator_comparison":
        # item_name is the estimator type (e.g., "ls", "pso", "perfect")
        perfect_csi = (item_name.lower() == "perfect")
        estimator_type = "ls" if perfect_csi else item_name # Default to LS structure even for perfect to avoid errors if logic expects it
        
        return Model(
            config=base_config,
            estimator_type=estimator_type,
            perfect_csi=perfect_csi
        )
        
    elif mode == "resource_manager_comparison":
        # item_name is the manager type (e.g., "round_robin", "cnn")
        from src.models.resource_manager import (
            StaticResourceManager,
            RoundRobinResourceManager,
            MaxThroughputResourceManager,
            ProportionalFairResourceManager,
        )
        from src.models.cnn_resource_manager import CNNResourceManager
        
        num_ut = base_config.get("num_ut", 8)
        num_active = global_config.get("num_active_users", 2)
        
        name_lower = item_name.lower()
        manager = None
        
        if "static" in name_lower:
            manager = StaticResourceManager(active_ut_mask=[1]*num_ut)
        elif "round" in name_lower:
            manager = RoundRobinResourceManager(num_active=num_active)
        elif "max" in name_lower:
            manager = MaxThroughputResourceManager(num_active=num_active)
        elif "prop" in name_lower or "pf" in name_lower:
            manager = ProportionalFairResourceManager(num_active=num_active)
        elif "cnn" in name_lower:
            model_path = global_config.get("cnn_model_path", "models/cnn_resource_manager.h5")
            if os.path.exists(model_path):
                manager = CNNResourceManager(model_path=model_path)
            else:
                 raise FileNotFoundError(f"CNN model not found: {model_path}")
        else:
            raise ValueError(f"Unknown resource manager: {item_name}")

        return Model(
            config=base_config,
            estimator_type="lmmse", # Fixed good estimator for manager benchmarks
            resource_manager=manager,
            perfect_csi=False
        )
        
    raise ValueError(f"Unknown mode: {mode}")


def _run_batches(model: Model, batch_size: int, total_batches: int, ebno_db: float) -> Dict[str, float]:
    """Runs the simulation batches and calculates average metrics."""
    total_errors = 0
    total_bits = 0
    total_throughput = 0.0 # bits successfully transferred
    latency_accum = 0.0
    
    for b in range(total_batches):
        res = model.run_batch(batch_size, ebno_db, include_details=True)
        
        # BER inputs
        bits = res["bits"]
        bits_hat = res["bits_hat"]
        errors = np.sum(bits != bits_hat)
        total_errors += errors
        total_bits += bits.size
        
        # Throughput (successful bits)
        total_throughput += max(0, bits.size - errors)
        
        # Latency
        latency_accum += res["latency_sec"]

    avg_ber = total_errors / total_bits if total_bits > 0 else 0.0
    avg_throughput = total_throughput / total_batches # Average bits per batch
    avg_latency = latency_accum / total_batches
    
    return {
        "ber": avg_ber,
        "throughput": avg_throughput,
        "latency": avg_latency
    }


def _plot_campaign_results(full_results: dict, output_dir: str, mode: str):
    """Generates comparison plots."""
    results = full_results["results"]
    ebno_range = full_results["ebno_db_range"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    markers = ["o", "s", "D", "^", "v", "x", "*", "+"]
    
    for idx, (name, metrics) in enumerate(results.items()):
        m = markers[idx % len(markers)]
        
        # Plot 1: BER
        ber = [max(v, 1e-12) for v in metrics["ber"]]
        ax1.semilogy(ebno_range, ber, marker=m, label=name, linewidth=2)
        
        # Plot 2: Secondary Metric (Throughput or Latency)
        if mode == "resource_manager_comparison":
            ax2.plot(ebno_range, metrics["throughput"], marker=m, label=name, linewidth=2)
            ylabel2 = "Avg Throughput (bits/batch)"
            title2 = "Throughput Comparison"
        else:
            latency_ms = [l * 1000 for l in metrics["latency"]]
            ax2.plot(ebno_range, latency_ms, marker=m, label=name, linewidth=2)
            ylabel2 = "Latency (ms)"
            title2 = "Latency Comparison"

    ax1.set_xlabel("Eb/No (dB)")
    ax1.set_ylabel("BER")
    ax1.set_title("BER Comparison")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()
    
    ax2.set_xlabel("Eb/No (dB)")
    ax2.set_ylabel(ylabel2)
    ax2.set_title(title2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(f"Campaign Results: {mode.replace('_', ' ').title()}")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "campaign_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Plots saved to {plot_path}")
