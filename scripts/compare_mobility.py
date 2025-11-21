#!/usr/bin/env python3
"""
Compare Static vs Mobile UT Performance.

This script runs two simulations:
1. Static UTs (0 m/s)
2. Mobile UTs (10 m/s)

It uses the '6g_poc_fast' scenario for speed, but overrides the velocity.
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
from src.sim.plotting import plot_simulation_results

def run_comparison():
    print("=" * 80)
    print("Comparing Static vs Mobile UT Performance")
    print("=" * 80)
    
    # Common parameters
    scenario = "umi"
    ebno_db_range = np.arange(0.0, 21.0, 5.0) # Coarse grid for speed
    batch_size = 4 # Reduced from 64 to avoid OOM with tr38901
    max_mc_iter = 10 # Low iterations for speed
    output_dir = "results/mobility_comparison"
    
    # 1. Static Simulation
    print("\nRunning Static Simulation (0 m/s)...")
    # 1. Static Simulation
    print("\nRunning Static Simulation (0 m/s)...")
    config_static = SystemConfig(
        scenario=scenario,
        min_ut_velocity=0.0,
        max_ut_velocity=0.0,
        channel_model_type="tr38901",
        num_ut=4,
        num_ut_ant=1,
        num_bs_ant=16
    )
    
    results_static = run_simulation(
        scenario=scenario,
        perfect_csi_list=[True], # Focus on perfect CSI first
        ebno_db_range=ebno_db_range,
        batch_size=batch_size,
        max_mc_iter=max_mc_iter,
        config=config_static,
        save_results=True,
        plot_results=False,
        output_dir=output_dir,
        profile_name="static_0ms"
    )
    
    # 2. Mobile Simulation
    print("\nRunning Mobile Simulation (10 m/s)...")
    config_mobile = SystemConfig(
        scenario=scenario,
        min_ut_velocity=10.0,
        max_ut_velocity=10.0,
        channel_model_type="tr38901",
        num_ut=4,
        num_ut_ant=1,
        num_bs_ant=16
    )
    
    results_mobile = run_simulation(
        scenario=scenario,
        perfect_csi_list=[True],
        ebno_db_range=ebno_db_range,
        batch_size=batch_size,
        max_mc_iter=max_mc_iter,
        config=config_mobile,
        save_results=True,
        plot_results=False,
        output_dir=output_dir,
        profile_name="mobile_10ms"
    )
    
    # 3. Plot Comparison
    print("\nGenerating Comparison Plot...")
    plot_comparison(results_static, results_mobile, output_dir)

def plot_comparison(static_res, mobile_res, output_dir):
    """Plot BER/BLER comparison."""
    
    # Extract data
    # Assuming single run (Perfect CSI)
    static_metrics = static_res["runs"][0]["metrics"]
    mobile_metrics = mobile_res["runs"][0]["metrics"]
    
    ebno = [m["ebno_db"] for m in static_metrics]
    
    ber_static = [m["overall"]["ber"] for m in static_metrics]
    bler_static = [m["overall"]["bler"] for m in static_metrics]
    
    ber_mobile = [m["overall"]["ber"] for m in mobile_metrics]
    bler_mobile = [m["overall"]["bler"] for m in mobile_metrics]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # BER
    ax1.semilogy(ebno, ber_static, 'b-o', label='Static (0 m/s)')
    ax1.semilogy(ebno, ber_mobile, 'r-s', label='Mobile (10 m/s)')
    ax1.set_xlabel('Eb/No (dB)')
    ax1.set_ylabel('BER')
    ax1.set_title('Bit Error Rate Comparison')
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()
    
    # BLER
    ax2.semilogy(ebno, bler_static, 'b-o', label='Static (0 m/s)')
    ax2.semilogy(ebno, bler_mobile, 'r-s', label='Mobile (10 m/s)')
    ax2.set_xlabel('Eb/No (dB)')
    ax2.set_ylabel('BLER')
    ax2.set_title('Block Error Rate Comparison')
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = Path(output_dir) / "mobility_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Comparison plot saved to: {plot_path.absolute()}")

if __name__ == "__main__":
    # Configure environment
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run_comparison()
