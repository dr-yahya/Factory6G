#!/usr/bin/env python3
"""
Compare Sionna Baseline (Pre-enhancement) vs 3GPP Rel-20 Baseline.

This script runs two simulations:
1. Sionna Baseline: Standard 5G/6G baseline (e.g., 30 kHz SCS, UMi).
2. 3GPP Rel-20 Baseline: Approximated by using different parameters (e.g., 15 kHz SCS or different numerology)
   or simply a reference comparison. 
   
   Note: Since "Rel-20" is future, we will approximate it as a "Standard 5G-Advanced" setup 
   vs the "Sionna Baseline" which we will treat as the "Pre-enhancement" (maybe unoptimized).
   
   Actually, per user request: "6g baseline sionna before enhnacment" vs "3gpp relaese 20 baseline".
   We will define:
   - "Sionna Baseline (Pre-enhancement)": The default '6g_baseline' config (30 kHz SCS).
   - "3GPP Rel-20 Baseline": A more aggressive/advanced config (e.g., 60 kHz SCS, lower latency target)
     OR if the user implies Sionna is *worse* than Rel-20, we might adjust accordingly.
     
   Let's assume:
   - Sionna Baseline: 30 kHz SCS, 14 symbols/slot.
   - Rel-20 Baseline: 60 kHz SCS, 14 symbols/slot (Lower latency).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sim.runner import run_simulation
from src.components.config import SystemConfig

def run_comparison():
    print("=" * 80)
    print("Comparing Sionna Baseline vs 3GPP Rel-20 Baseline")
    print("=" * 80)
    
    # Common parameters
    scenario = "umi"
    ebno_db_range = np.arange(0.0, 21.0, 5.0)
    batch_size = 4 # Reduced from 32 to avoid OOM
    max_mc_iter = 20 # Reduced for speed
    output_dir = "results/baseline_comparison"
    
    # 1. Sionna Baseline (Pre-enhancement)
    # Representing a standard 5G-like baseline often used as starting point
    print("\nRunning Sionna Baseline (Pre-enhancement)...")
    config_sionna = SystemConfig(
        scenario=scenario,
        subcarrier_spacing=30e3, # 30 kHz
        channel_model_type="rayleigh", # Fast model for speed
        num_ut=4,
        num_ut_ant=1,
        num_bs_ant=16
    )
    
    results_sionna = run_simulation(
        scenario=scenario,
        perfect_csi_list=[True],
        ebno_db_range=ebno_db_range,
        batch_size=batch_size,
        max_mc_iter=max_mc_iter,
        config=config_sionna,
        save_results=True,
        plot_results=False,
        output_dir=output_dir,
        profile_name="sionna_baseline"
    )
    
    # 2. 3GPP Rel-20 Baseline
    # Representing an advanced baseline (e.g., 60 kHz SCS for lower latency)
    print("\nRunning 3GPP Rel-20 Baseline...")
    config_rel20 = SystemConfig(
        scenario=scenario,
        subcarrier_spacing=60e3, # 60 kHz (Lower latency)
        channel_model_type="rayleigh",
        num_ut=4,
        num_ut_ant=1,
        num_bs_ant=16
    )
    
    results_rel20 = run_simulation(
        scenario=scenario,
        perfect_csi_list=[True],
        ebno_db_range=ebno_db_range,
        batch_size=batch_size,
        max_mc_iter=max_mc_iter,
        config=config_rel20,
        save_results=True,
        plot_results=False,
        output_dir=output_dir,
        profile_name="rel20_baseline"
    )
    
    # 3. Plot Comparison
    print("\nGenerating Comparison Plot...")
    plot_comparison(results_sionna, results_rel20, output_dir)

def plot_comparison(res_sionna, res_rel20, output_dir):
    """Plot BER and Latency with LARGE fonts."""
    
    # Extract data
    # Assuming single run (Perfect CSI)
    metrics_sionna = res_sionna["runs"][0]["metrics"]
    metrics_rel20 = res_rel20["runs"][0]["metrics"]
    
    ebno = [m["ebno_db"] for m in metrics_sionna]
    
    ber_sionna = [m["overall"]["ber"] for m in metrics_sionna]
    lat_sionna = [m["overall"]["latency_sec"] * 1000 for m in metrics_sionna] # ms
    
    ber_rel20 = [m["overall"]["ber"] for m in metrics_rel20]
    lat_rel20 = [m["overall"]["latency_sec"] * 1000 for m in metrics_rel20] # ms
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Font sizes
    TITLE_SIZE = 20
    LABEL_SIZE = 18
    TICK_SIZE = 16
    LEGEND_SIZE = 16
    
    # BER Plot
    ax1.semilogy(ebno, ber_sionna, 'b-o', linewidth=3, markersize=10, label='Sionna Baseline (Pre-enhancement)')
    ax1.semilogy(ebno, ber_rel20, 'r-s', linewidth=3, markersize=10, label='3GPP Rel-20 Baseline')
    ax1.set_xlabel('Eb/No (dB)', fontsize=LABEL_SIZE)
    ax1.set_ylabel('Bit Error Rate (BER)', fontsize=LABEL_SIZE)
    ax1.set_title('BER Comparison', fontsize=TITLE_SIZE)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    
    # Latency Plot
    ax2.plot(ebno, lat_sionna, 'b-o', linewidth=3, markersize=10, label='Sionna Baseline (30 kHz SCS)')
    ax2.plot(ebno, lat_rel20, 'r-s', linewidth=3, markersize=10, label='3GPP Rel-20 Baseline (60 kHz SCS)')
    ax2.set_xlabel('Eb/No (dB)', fontsize=LABEL_SIZE)
    ax2.set_ylabel('Latency (ms)', fontsize=LABEL_SIZE)
    ax2.set_title('Latency Comparison', fontsize=TITLE_SIZE)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=LEGEND_SIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    
    # Adjust layout
    plt.tight_layout()
    
    plot_path = Path(output_dir) / "baseline_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Comparison plot saved to: {plot_path.absolute()}")

if __name__ == "__main__":
    # Configure environment
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run_comparison()
