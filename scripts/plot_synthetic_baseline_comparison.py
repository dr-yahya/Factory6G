#!/usr/bin/env python3
"""
Generate Synthetic Baseline Comparison Plots (Separate Files).

Compares:
1. Sionna Baseline (Pre-enhancement)
2. 3GPP Rel-20 Baseline

Changes:
- Eb/No limited to 12 dB.
- Separate files for BER and Latency.
- "Pre-enhancement" label on both.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_synthetic_comparison():
    output_dir = Path("results/baseline_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Actual Data (Sionna Baseline)
    json_path = Path("results/6g_baseline/simulation_results_umi_ls_lin_6g_baseline_20251117_160802.json")
    print(f"Loading baseline data from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Extract metrics
    # The JSON structure has 'runs' list. We assume the first run is what we want.
    # We need to match Eb/No from the 'metrics' list inside the run.
    
    run_metrics = data["runs"][0]["metrics"]
    
    # Arrays to store extracted data
    ebno_actual = []
    ber_actual = []
    lat_actual = []
    
    for m in run_metrics:
        e = m["ebno_db"]
        # Filter range 0 to 12 dB
        if 0.0 <= e <= 12.0:
            ebno_actual.append(e)
            ber_actual.append(m["overall"]["ber"])
            # Latency in JSON is 'air_interface_latency_ms' (which seems to be wall-clock ms based on values)
            lat_actual.append(m["overall"]["air_interface_latency_ms"])
            
    ebno_actual = np.array(ebno_actual)
    ber_actual = np.array(ber_actual)
    lat_actual = np.array(lat_actual)
    
    # Sort by Eb/No just in case
    sort_idx = np.argsort(ebno_actual)
    ebno_actual = ebno_actual[sort_idx]
    ber_actual = ber_actual[sort_idx]
    lat_actual = lat_actual[sort_idx]
    
    # 2. Generate Synthetic Data matching User Specified Ranges
    # User Request:
    # 1. 6G Sionna Baseline: 1e-1 to 1e-2
    # 2. 6G Sionna Enhanced: 1e-1.5 to 1e-2.5
    # 3. 3GPP Rel-20: 1e-1 to 1e-7
    # Range is 0 to 12 dB.
    
    # We assume log-linear behavior (straight lines on semilog y-axis).
    # log10(BER) = slope * EbNo + intercept
    
    # Curve 1: Baseline (1e-1 to 1e-2)
    # At 0dB: -1. At 12dB: -2.
    # Slope = (-2 - (-1)) / 12 = -1/12
    ber_actual = 10 ** (-1 - (1/12) * ebno_actual)
    
    # Curve 2: Enhanced (1e-1.5 to 1e-2.5)
    # At 0dB: -1.5. At 12dB: -2.5.
    # Slope = (-2.5 - (-1.5)) / 12 = -1/12 (Same slope, shifted)
    ber_enhanced = 10 ** (-1.5 - (1/12) * ebno_actual)
    
    # Curve 3: Rel-20 (1e-1 to 1e-7)
    # At 0dB: -1. At 12dB: -7.
    # Slope = (-7 - (-1)) / 12 = -6/12 = -0.5
    ber_rel20 = 10 ** (-1 - 0.5 * ebno_actual)

    # Synthetic Rel-20 Latency:
    lat_rel20 = lat_actual * 0.1
    
    # Synthetic Enhanced Channel Estimation Latency:
    # Also closer to baseline.
    lat_enhanced = lat_actual * 0.8 # 20% improvement over baseline
    
    # Font sizes
    TITLE_SIZE = 24
    LABEL_SIZE = 20
    TICK_SIZE = 18
    LEGEND_SIZE = 18
    
    # ---------------------------------------------------------
    # Plot 1: BER
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(12, 9))
    
    ax1.semilogy(ebno_actual, ber_actual, 'b-o', linewidth=4, markersize=12, label='6G Sionna Baseline (Pre-enhancement)')
    ax1.semilogy(ebno_actual, ber_rel20, 'r-s', linewidth=4, markersize=12, label='3GPP Rel-20 Baseline')
    ax1.semilogy(ebno_actual, ber_enhanced, 'g-^', linewidth=4, markersize=12, label='6G Sionna with Enhanced Channel Estimation')
    
    ax1.set_xlabel('Eb/No (dB)', fontsize=LABEL_SIZE)
    ax1.set_ylabel('Bit Error Rate (BER)', fontsize=LABEL_SIZE)
    ax1.set_title('BER Performance', fontsize=TITLE_SIZE, fontweight='bold')
    ax1.grid(True, which="both", alpha=0.3, linewidth=1.5)
    ax1.legend(fontsize=LEGEND_SIZE)
    ax1.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    
    plt.tight_layout()
    plot_path_ber = output_dir / "baseline_comparison_ber.png"
    plt.savefig(plot_path_ber, dpi=300)
    print(f"BER plot saved to: {plot_path_ber.absolute()}")
    plt.close(fig1)
    
    # ---------------------------------------------------------
    # Plot 2: Latency
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    
    ax2.plot(ebno_actual, lat_actual, 'b-o', linewidth=4, markersize=12, label='6G Sionna Baseline (Pre-enhancement)')
    ax2.plot(ebno_actual, lat_rel20, 'r-s', linewidth=4, markersize=12, label='3GPP Rel-20 Baseline')
    ax2.plot(ebno_actual, lat_enhanced, 'g-^', linewidth=4, markersize=12, label='6G Sionna with Enhanced Channel Estimation')
    
    ax2.set_xlabel('Eb/No (dB)', fontsize=LABEL_SIZE)
    ax2.set_ylabel('Latency (ms)', fontsize=LABEL_SIZE)
    ax2.set_title('End-to-End Latency', fontsize=TITLE_SIZE, fontweight='bold')
    ax2.grid(True, which="both", alpha=0.3, linewidth=1.5)
    ax2.legend(fontsize=LEGEND_SIZE)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    # ax2.set_ylim(0, 1.5) # Remove fixed limit as actual data might be large
    
    plt.tight_layout()
    plot_path_lat = output_dir / "baseline_comparison_latency.png"
    plt.savefig(plot_path_lat, dpi=300)
    print(f"Latency plot saved to: {plot_path_lat.absolute()}")
    plt.close(fig2)

if __name__ == "__main__":
    plot_synthetic_comparison()
