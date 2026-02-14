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
from src.components.antenna import AntennaConfig
from src.components.transmitter import Transmitter
from src.components.channel import ChannelModel
from src.components.receiver import Receiver
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.mimo import StreamManagement

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
    # Storage for aggregated results
    # Structure: { item_name: { "ber": [], "throughput": [], "latency": [] } }
    aggregated_results = {item: {"ber": [], "throughput": [], "latency": []} for item in items_to_compare}
    
    # --- Component Initialization (Shared) ---
    print("\nInitializing shared simulation components...")
    # Using LMMSE receiver for all RM comparisons to ensure fair channel estimation benchmark
    # If mode is estimators, we might not be able to share Receiver, but can share Channel/Tx.
    
    # We need to create the components on CPU to avoid RNG issues
    with tf.device("/CPU:0"):
        # 1. Config & Grid
        num_ofdm_symbols = system_config.get("num_ofdm_symbols", 14)
        fft_size = system_config.get("fft_size", 512)
        subcarrier_spacing = system_config.get("subcarrier_spacing", 30e3)
        num_tx = system_config.get("num_ut", 8) 
        num_streams_per_tx = system_config.get("num_ut_ant", 1)
        cyclic_prefix_length = system_config.get("cyclic_prefix_length", 20)
        pilot_ofdm_symbol_indices = system_config.get("pilot_ofdm_symbol_indices", [2, 11])
        
        rx_tx_association = np.zeros([1, num_tx])
        rx_tx_association[0, :] = 1
        
        rg = ResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=num_tx,
            num_streams_per_tx=num_streams_per_tx,
            cyclic_prefix_length=cyclic_prefix_length,
            pilot_pattern="kronecker",
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices
        )
        
        sm = StreamManagement(rx_tx_association, num_streams_per_tx)
        
        # 2. Components
        antenna_config = AntennaConfig(system_config)
        transmitter = Transmitter(system_config, rg)
        channel = ChannelModel(system_config, antenna_config, rg)
        
        # 3. Default Receiver (LMMSE) for RM mode
        # For Estimator mode, we might need different receivers, so we might only share Tx/Channel.
        default_receiver = None
        if mode == "resource_manager_comparison":
            from sionna.phy.ofdm import LSChannelEstimator
            from src.components.estimators import LMMSEChannelEstimator
            # Use LMMSE for fair comparison of RMs
            estimator = LMMSEChannelEstimator(rg)
            encoder = transmitter._encoder
            default_receiver = Receiver(
                system_config, rg, sm, encoder, 
                perfect_csi=False, 
                channel_estimator=estimator
            )

    shared_components = {
        "rg": rg,
        "sm": sm,
        "antenna_config": antenna_config,
        "transmitter": transmitter,
        "channel": channel,
        "receiver": default_receiver,
        "estimator_type": "lmmse",
        "perfect_csi": False
    }

    # Storage for intermediate sums: [item][ebno_idx] -> {errors, bits, throughput, latency}
    intermediate_results = {
        item: [
            {"errors": 0, "bits": 0, "throughput": 0.0, "latency": 0.0} 
            for _ in ebno_db_range
        ] for item in items_to_compare
    }

    # --- Model Initialization (Once) ---
    print("Initializing models...")
    models = {}
    for item_name in items_to_compare:
        # Build Model (wrapper) reusing components
        # If mode is estimators, we might need to swap receiver/estimator type
        # But channel is same.
        
        # Handling Estimator Mode nuances
        current_shared = shared_components.copy()
        if mode == "estimator_comparison":
            # We can't reuse the LMMSE receiver if testing LS or Perfect
            # But we reuse Channel/Tx/RG
            if "receiver" in current_shared:
                del current_shared["receiver"]
        
        models[item_name] = _build_model(mode, item_name, system_config, config, reused_components=current_shared)

    # --- Main Loop (Inverted for Efficiency) ---
    print(f"Starting Campaign: {total_batches} Batches (Batch Size {batch_size})")
    print("Optimization: Reusing channel realizations across comparison items.")
    print("Optimization: Reusing Model instances (no reload overhead).")
    
    import time
    global_start = time.time()
    
    for b in range(total_batches):
        batch_start = time.time()
        print(f"Batch {b+1}/{total_batches} ... ", end="", flush=True)
        
        # 1. Generate Channel Topology (Once per batch)
        # This is the expensive step!
        with tf.device("/CPU:0"):
            channel.set_topology(batch_size)
            
        print(f"[Channel Gen] ", end="", flush=True)
            
        # 2. Iterate Eb/No
        # For a fixed channel, we sweep SNR.
        # Ideally we could even vectorise SNR but Model takes scalar.
        
        for e_idx, ebno_val in enumerate(ebno_db_range):
            ebno_val = float(ebno_val)
            
            # 3. Iterate Managers/Estimators
            for item_name in items_to_compare:
                model = models[item_name]
                
                # Run batch (No topology gen)
                metrics = model.run_batch(batch_size, ebno_val, include_details=True, regenerate_topology=False)
                
                # Accumulate
                stats = intermediate_results[item_name][e_idx]
                
                # Parse metrics from model result
                bits = metrics["bits"]
                bits_hat = metrics["bits_hat"]
                errors = np.sum(bits != bits_hat)
                
                stats["errors"] += errors
                stats["bits"] += bits.size
                stats["throughput"] += max(0, bits.size - errors)
                stats["latency"] += metrics["latency_sec"]
                
        batch_time = time.time() - batch_start
        print(f"Done ({batch_time:.2f}s)")

    # --- Final Aggregation ---
    print("\nAggregating final results...")
    for item_name in items_to_compare:
        for e_idx, _ in enumerate(ebno_db_range):
            stats = intermediate_results[item_name][e_idx]
            
            avg_ber = stats["errors"] / stats["bits"] if stats["bits"] > 0 else 0.0
            avg_thr = stats["throughput"] / total_batches
            avg_lat = stats["latency"] / total_batches
            
            aggregated_results[item_name]["ber"].append(avg_ber)
            aggregated_results[item_name]["throughput"].append(avg_thr)
            aggregated_results[item_name]["latency"].append(avg_lat)
            
    total_time = time.time() - global_start
    print(f"Total Simulation Time: {total_time:.2f}s")

    # --- Save Results ---
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save as JSON (Raw data)
    full_results = {
        "config": config,
        "results": aggregated_results,
        "ebno_db_range": ebno_db_range.tolist()
    }
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if mode == "resource_manager_comparison":
        filename_base = f"simulation_results_resource_managers_{timestamp}"
    else:
        estimator_type = system_config.get("estimator_type", "unknown_est")
        filename_base = f"simulation_results_{estimator_type}_{timestamp}"

    # 1. Save as JSON (Raw data)
    full_results = {
        "config": config,
        "results": aggregated_results,
        "ebno_db_range": ebno_db_range.tolist(),
        "timestamp": timestamp
    }
    json_path = os.path.join(output_dir, f"{filename_base}.json")
    save_simulation_results(full_results, output_dir, filename=f"{filename_base}.json")
    
    # 2. Save as CSV (Tabular data for easy reading)
    csv_path = os.path.join(output_dir, f"{filename_base}.csv")
    save_results_as_csv(full_results, output_dir, filename=f"{filename_base}.csv")
    
    # --- Generate Plots ---
    if config.get("plot_results", True):
        _plot_campaign_results(full_results, output_dir, mode, timestamp)
        
    return full_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(mode: str, item_name: str, system_config: dict, global_config: dict, reused_components: dict = None) -> Model:
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
            perfect_csi=perfect_csi,
            reused_components=reused_components
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
            perfect_csi=False,
            reused_components=reused_components
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


def _plot_campaign_results(full_results: dict, output_dir: str, mode: str, timestamp: str):
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
            # Apply smoothing for smoother visualization of low-batch runs
            throughput = metrics["throughput"]
            if len(throughput) > 3:
                # Simple 3-point moving average for smoothing
                throughput_smooth = []
                for i in range(len(throughput)):
                    if i == 0:
                        val = (throughput[0] + throughput[1]) / 2
                    elif i == len(throughput) - 1:
                        val = (throughput[-2] + throughput[-1]) / 2
                    else:
                        val = (throughput[i-1] + throughput[i] + throughput[i+1]) / 3
                    throughput_smooth.append(val)
                ax2.plot(ebno_range, throughput_smooth, marker=m, label=f"{name} (smoothed)", linewidth=2, linestyle='--')
                ax2.plot(ebno_range, throughput, marker=m, label=name, linewidth=1, alpha=0.3) # Show raw data faintly
            else:
                ax2.plot(ebno_range, throughput, marker=m, label=name, linewidth=2)
                
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
    
    title_text = mode.replace('_', ' ').title().replace("Comparison", "Comparison") # Ensure consistency
    if mode == "estimator_comparison":
        title_text = "Estimator Comparison"
        filename_base = "estimator_comparison"
    elif mode == "resource_manager_comparison":
        title_text = "Resource Manager Comparison"
        filename_base = "resource_manager_comparison"
    else:
        filename_base = "campaign_comparison"

    plt.suptitle(title_text)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{filename_base}_{timestamp}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✓ Plots saved to {plot_path}")
