#!/usr/bin/env python3
"""
Create baseline results structure with example metrics and plots for 6G simulation.

This script generates a baseline results directory structure with placeholder
matrices and plots to demonstrate the expected output format.
Based on 3GPP Release 20 (initiated 2025) - Formal 6G Studies (IMT-2030).
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 3GPP Release information
TGPP_RELEASE = "Release 20"
TGPP_RELEASE_YEAR = "2025"
TGPP_RELEASE_DESCRIPTION = "Formal 6G Studies (IMT-2030) - Bridge between 5G-Advanced and 6G with AI/ML, ISAC, and NTN enhancements"
TGPP_RELEASE_NOTE = "Release 20 marks the commencement of formal studies into 6G use cases, performance requirements, and enabling technologies. Release 21 will start normative 6G work with first 6G specifications."

def create_baseline_results():
    """Create baseline results structure with example data based on 3GPP Release 20."""
    
    # Create baseline directory with 3GPP Release 20 naming
    baseline_dir = project_root / "results" / "3gpp_release20_baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    matrices_dir = baseline_dir / "matrices"
    plots_dir = baseline_dir / "plots"
    matrices_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print(f"Creating 3GPP {TGPP_RELEASE} Baseline Results")
    print(f"3GPP {TGPP_RELEASE} ({TGPP_RELEASE_YEAR}): {TGPP_RELEASE_DESCRIPTION}")
    print(f"{TGPP_RELEASE_NOTE}")
    print("=" * 80)
    print(f"Baseline directory: {baseline_dir}")
    print()
    
    # Simulation parameters (from min_6g_params_config.json)
    # Generate 25 points for smooth plots: from -5 to 9 dB
    ebno_db_range = np.linspace(-5.0, 9.0, 25)  # 25 points for smooth curves
    num_ebno = len(ebno_db_range)
    num_streams = 16  # 8 UTs * 2 antennas = 16 streams
    
    # Create example data for both CSI conditions
    csi_conditions = [
        ("imperfect", False),
        ("perfect", True)
    ]
    
    # Metrics to process (only metrics with 6G requirements)
    # 6G Requirements:
    # - BER/BLER: < 10^-9
    # - Latency: < 0.1 ms
    # - Energy per bit: < 1 pJ/bit
    # - Outage probability: < 10^-6
    # - Throughput/Spectral Efficiency: 6G targets (1 Tbps peak, 100 bits/s/Hz)
    per_stream_metrics = ["ber", "bler", "throughput_bits", "outage_probability"]
    overall_metrics = ["ber", "bler", "throughput_bits", "spectral_efficiency", 
                       "outage_probability", "air_interface_latency_ms", "energy_per_bit_pj"]
    
    print("Generating example matrices...")
    print()
    
    # Generate data for each CSI condition (save matrices only, no individual plots)
    for csi_str, perfect_csi in csi_conditions:
        print(f"Processing {csi_str.upper()} CSI condition...")
        
        # Generate per-stream metrics (matrices: [num_ebno, num_streams])
        for metric_name in per_stream_metrics:
            # Create realistic example data
            matrix = np.zeros((num_ebno, num_streams))
            
            if metric_name == "ber":
                # BER decreases exponentially with Eb/No - 6G target: < 10^-9
                # Perfect CSI meets 6G target at ~5 dB, Imperfect at ~9 dB
                if perfect_csi:
                    # Perfect CSI: meets 6G target (< 10^-9) around 5 dB
                    base_ber = 1e-2
                    ebno_threshold = 5.0  # Eb/No where 6G target is met
                else:
                    # Imperfect CSI: meets 6G target (< 10^-9) around 9 dB
                    base_ber = 1e-1
                    ebno_threshold = 9.0
                for i, ebno in enumerate(ebno_db_range):
                    stream_variation = 1 + 0.1 * np.random.randn(num_streams)
                    # Exponential decay, reaching 6G target (< 10^-9) at threshold
                    # Formula: ber = base_ber * 10^(-(ebno - start) / slope)
                    # At threshold: 1e-9 = base_ber * 10^(-(threshold - start) / slope)
                    # Solving: slope = (threshold - start) / log10(base_ber / 1e-9)
                    start_ebno = -5.0
                    target_ber = 1e-9
                    slope = (ebno_threshold - start_ebno) / np.log10(base_ber / target_ber)
                    ber_value = base_ber * 10**(-(ebno - start_ebno) / slope)
                    # Ensure 6G target is met at and above threshold
                    if ebno >= ebno_threshold:
                        ber_value = np.minimum(ber_value, target_ber * 0.1)  # Below 6G target
                    matrix[i, :] = ber_value * stream_variation
                    matrix[i, :] = np.clip(matrix[i, :], 1e-11, 1.0)
            elif metric_name == "bler":
                # BLER similar to BER but typically higher - 6G target: < 10^-9
                # Perfect CSI meets 6G target at ~5 dB, Imperfect at ~9 dB
                if perfect_csi:
                    base_bler = 5e-2
                    ebno_threshold = 5.0
                else:
                    base_bler = 2e-1
                    ebno_threshold = 9.0
                for i, ebno in enumerate(ebno_db_range):
                    stream_variation = 1 + 0.15 * np.random.randn(num_streams)
                    # Similar formula for BLER
                    start_ebno = -5.0
                    target_bler = 1e-9
                    slope = (ebno_threshold - start_ebno) / np.log10(base_bler / target_bler)
                    bler_value = base_bler * 10**(-(ebno - start_ebno) / slope)
                    if ebno >= ebno_threshold:
                        bler_value = np.minimum(bler_value, target_bler * 0.1)
                    matrix[i, :] = bler_value * stream_variation
                    matrix[i, :] = np.clip(matrix[i, :], 1e-11, 1.0)
            elif metric_name == "throughput_bits":
                # Throughput increases with Eb/No - 6G target: 1 Tbps peak, 1 Gbps user-experienced
                # Scale to show progression toward 6G targets (normalized for per-stream)
                base_throughput = 1e6 if not perfect_csi else 2e6  # bits per stream
                for i, ebno in enumerate(ebno_db_range):
                    stream_variation = 1 + 0.05 * np.random.randn(num_streams)
                    # Exponential growth toward 6G targets at high Eb/No
                    growth_factor = 1 + (ebno + 5) / 5  # Scale from -5 to 9 dB
                    matrix[i, :] = base_throughput * growth_factor * stream_variation
                    matrix[i, :] = np.maximum(matrix[i, :], 0)
            elif metric_name == "outage_probability":
                # Outage probability decreases with Eb/No - 6G target: < 10^-6
                # Perfect CSI meets 6G target at ~5 dB, Imperfect at ~9 dB
                if perfect_csi:
                    base_outage = 1e-4
                    ebno_threshold = 5.0
                else:
                    base_outage = 1e-2
                    ebno_threshold = 9.0
                for i, ebno in enumerate(ebno_db_range):
                    stream_variation = 1 + 0.1 * np.random.randn(num_streams)
                    # Outage probability: 6G target < 10^-6
                    start_ebno = -5.0
                    target_outage = 1e-6
                    slope = (ebno_threshold - start_ebno) / np.log10(base_outage / target_outage)
                    outage_value = base_outage * 10**(-(ebno - start_ebno) / slope)
                    if ebno >= ebno_threshold:
                        outage_value = np.minimum(outage_value, target_outage * 0.1)  # Below 6G target
                    matrix[i, :] = outage_value * stream_variation
                    matrix[i, :] = np.clip(matrix[i, :], 1e-9, 1.0)
            
            # Save matrix with 3GPP Release 20 prefix
            matrix_file = matrices_dir / f"3gpp_release20_{metric_name}_per_stream_{csi_str}_run0.npy"
            np.save(matrix_file, matrix)
            print(f"  ✓ Saved matrix: {matrix_file.name} (shape: {matrix.shape})")
        
        # Generate overall metrics (vectors: [num_ebno])
        for metric_name in overall_metrics:
            # Create realistic example data
            if metric_name == "ber":
                # 6G target: < 10^-9
                # Perfect CSI meets 6G target at ~5 dB, Imperfect at ~9 dB
                if perfect_csi:
                    base_ber = 1e-2
                    ebno_threshold = 5.0
                else:
                    base_ber = 1e-1
                    ebno_threshold = 9.0
                # Exponential decay toward 6G target
                start_ebno = -5.0
                target_ber = 1e-9
                slope = (ebno_threshold - start_ebno) / np.log10(base_ber / target_ber)
                vector = base_ber * 10**(-(ebno_db_range - start_ebno) / slope)
                # Ensure 6G target is met at and above threshold
                vector[ebno_db_range >= ebno_threshold] = np.minimum(vector[ebno_db_range >= ebno_threshold], target_ber * 0.1)
                vector = np.clip(vector, 1e-11, 1.0)
            elif metric_name == "bler":
                # 6G target: < 10^-9
                # Perfect CSI meets 6G target at ~5 dB, Imperfect at ~9 dB
                if perfect_csi:
                    base_bler = 5e-2
                    ebno_threshold = 5.0
                else:
                    base_bler = 2e-1
                    ebno_threshold = 9.0
                start_ebno = -5.0
                target_bler = 1e-9
                slope = (ebno_threshold - start_ebno) / np.log10(base_bler / target_bler)
                vector = base_bler * 10**(-(ebno_db_range - start_ebno) / slope)
                vector[ebno_db_range >= ebno_threshold] = np.minimum(vector[ebno_db_range >= ebno_threshold], target_bler * 0.1)
                vector = np.clip(vector, 1e-11, 1.0)
            elif metric_name == "outage_probability":
                # 6G target: < 10^-6
                # Perfect CSI meets 6G target at ~5 dB, Imperfect at ~9 dB
                if perfect_csi:
                    base_outage = 1e-4
                    ebno_threshold = 5.0
                else:
                    base_outage = 1e-2
                    ebno_threshold = 9.0
                start_ebno = -5.0
                target_outage = 1e-6
                slope = (ebno_threshold - start_ebno) / np.log10(base_outage / target_outage)
                vector = base_outage * 10**(-(ebno_db_range - start_ebno) / slope)
                vector[ebno_db_range >= ebno_threshold] = np.minimum(vector[ebno_db_range >= ebno_threshold], target_outage * 0.1)
                vector = np.clip(vector, 1e-9, 1.0)
            elif metric_name == "air_interface_latency_ms":
                # Latency decreases with Eb/No - 6G target: < 0.1 ms
                # Perfect CSI meets 6G target at ~5 dB, Imperfect at ~9 dB
                start_ebno = -5.0
                target_latency = 0.1  # ms
                if perfect_csi:
                    # Perfect CSI: starts at 0.08, reaches 0.08 at 5 dB (already meets target)
                    base_latency = 0.08  # ms
                    ebno_threshold = 5.0
                    # Linear decrease from base to target, then maintain
                    vector = np.where(ebno_db_range < ebno_threshold,
                                     base_latency - (base_latency - target_latency * 0.8) * (ebno_db_range - start_ebno) / (ebno_threshold - start_ebno),
                                     target_latency * 0.8)
                else:
                    # Imperfect CSI: starts at 0.15, reaches 0.1 at 9 dB
                    base_latency = 0.15  # ms
                    ebno_threshold = 9.0
                    # Exponential decay: latency = base_latency * exp(-(ebno - start) / decay_factor)
                    decay_factor = (ebno_threshold - start_ebno) / np.log(base_latency / target_latency)
                    vector = base_latency * np.exp(-(ebno_db_range - start_ebno) / decay_factor)
                    # Ensure 6G target is met at and above threshold
                    vector[ebno_db_range >= ebno_threshold] = np.minimum(vector[ebno_db_range >= ebno_threshold], target_latency * 0.8)
                vector = np.clip(vector, 0.01, 0.5)
            elif metric_name == "energy_per_bit_pj":
                # Energy per bit decreases with Eb/No - 6G target: 1 pJ/bit
                # Perfect CSI meets 6G target at ~5 dB, Imperfect at ~9 dB
                start_ebno = -5.0
                target_energy = 1.0  # pJ/bit
                if perfect_csi:
                    # Perfect CSI: starts at 0.8, already below target, maintain or improve slightly
                    base_energy = 0.8  # pJ/bit
                    ebno_threshold = 5.0
                    # Slight decrease from base, then maintain below target
                    vector = np.where(ebno_db_range < ebno_threshold,
                                     base_energy - (base_energy - target_energy * 0.7) * (ebno_db_range - start_ebno) / (ebno_threshold - start_ebno),
                                     target_energy * 0.7)
                else:
                    # Imperfect CSI: starts at 2.0, reaches 1.0 at 9 dB
                    base_energy = 2.0  # pJ/bit
                    ebno_threshold = 9.0
                    # Exponential decay: energy = base_energy * exp(-(ebno - start) / decay_factor)
                    decay_factor = (ebno_threshold - start_ebno) / np.log(base_energy / target_energy)
                    vector = base_energy * np.exp(-(ebno_db_range - start_ebno) / decay_factor)
                    # Ensure 6G target is met at and above threshold
                    vector[ebno_db_range >= ebno_threshold] = np.minimum(vector[ebno_db_range >= ebno_threshold], target_energy * 0.9)
                vector = np.clip(vector, 0.1, 5.0)
            elif metric_name == "throughput_bits":
                # 6G target: 1 Tbps peak, 1 Gbps user-experienced
                # Show progression toward 6G targets (scaled for demonstration)
                base_throughput = 1e8 if not perfect_csi else 2e8  # bits
                # Exponential growth toward 6G targets
                growth_factor = np.exp((ebno_db_range + 5) / 4)  # Scale from -5 to 9 dB
                vector = base_throughput * growth_factor
                vector = np.maximum(vector, 0)
            elif metric_name == "spectral_efficiency":
                # 6G target: Peak 100 bits/s/Hz
                base_se = 0.5 if not perfect_csi else 0.8
                # Exponential growth toward 6G target
                growth_factor = np.exp((ebno_db_range + 5) / 3)  # Scale from -5 to 9 dB
                vector = base_se * growth_factor
                # 6G target: 100 bits/s/Hz peak, allow up to 120 for demonstration
                vector = np.clip(vector, 0, 120.0)
            elif metric_name == "fairness_jain":
                # Fairness index (0-1, higher is better)
                base_fairness = 0.85 if not perfect_csi else 0.95
                vector = base_fairness + 0.05 * np.sin(ebno_db_range/5)
                vector = np.clip(vector, 0.5, 1.0)
            
            # Save vector with 3GPP Release 20 prefix
            vector_file = matrices_dir / f"3gpp_release20_{metric_name}_overall_{csi_str}_run0.npy"
            np.save(vector_file, vector)
            print(f"  ✓ Saved vector: {vector_file.name} (shape: {vector.shape})")
    
    # Create one comparison plot per metric (both CSI conditions on same plot)
    print()
    print("Creating comparison plots (one per metric)...")
    for metric_name in overall_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for csi_str, perfect_csi in csi_conditions:
            # Load or regenerate the vector
            if metric_name == "ber":
                # 6G target: < 10^-9
                # Perfect CSI meets 6G target at ~5 dB, Imperfect at ~9 dB
                if perfect_csi:
                    base_ber = 1e-2
                    ebno_threshold = 5.0
                else:
                    base_ber = 1e-1
                    ebno_threshold = 9.0
                start_ebno = -5.0
                target_ber = 1e-9
                slope = (ebno_threshold - start_ebno) / np.log10(base_ber / target_ber)
                metric_values = base_ber * 10**(-(ebno_db_range - start_ebno) / slope)
                metric_values[ebno_db_range >= ebno_threshold] = np.minimum(metric_values[ebno_db_range >= ebno_threshold], target_ber * 0.1)
                metric_values = np.clip(metric_values, 1e-11, 1.0)
            elif metric_name == "bler":
                # 6G target: < 10^-9
                # Perfect CSI meets 6G target at ~5 dB, Imperfect at ~9 dB
                if perfect_csi:
                    base_bler = 5e-2
                    ebno_threshold = 5.0
                else:
                    base_bler = 2e-1
                    ebno_threshold = 9.0
                start_ebno = -5.0
                target_bler = 1e-9
                slope = (ebno_threshold - start_ebno) / np.log10(base_bler / target_bler)
                metric_values = base_bler * 10**(-(ebno_db_range - start_ebno) / slope)
                metric_values[ebno_db_range >= ebno_threshold] = np.minimum(metric_values[ebno_db_range >= ebno_threshold], target_bler * 0.1)
                metric_values = np.clip(metric_values, 1e-11, 1.0)
            elif metric_name == "outage_probability":
                # 6G target: < 10^-6
                # Perfect CSI meets 6G target at ~5 dB, Imperfect at ~9 dB
                if perfect_csi:
                    base_outage = 1e-4
                    ebno_threshold = 5.0
                else:
                    base_outage = 1e-2
                    ebno_threshold = 9.0
                start_ebno = -5.0
                target_outage = 1e-6
                slope = (ebno_threshold - start_ebno) / np.log10(base_outage / target_outage)
                metric_values = base_outage * 10**(-(ebno_db_range - start_ebno) / slope)
                metric_values[ebno_db_range >= ebno_threshold] = np.minimum(metric_values[ebno_db_range >= ebno_threshold], target_outage * 0.1)
                metric_values = np.clip(metric_values, 1e-9, 1.0)
            elif metric_name == "air_interface_latency_ms":
                # 6G target: < 0.1 ms
                # Perfect CSI meets 6G target at ~5 dB, Imperfect at ~9 dB
                start_ebno = -5.0
                target_latency = 0.1  # ms
                if perfect_csi:
                    base_latency = 0.08
                    ebno_threshold = 5.0
                    metric_values = np.where(ebno_db_range < ebno_threshold,
                                            base_latency - (base_latency - target_latency * 0.8) * (ebno_db_range - start_ebno) / (ebno_threshold - start_ebno),
                                            target_latency * 0.8)
                else:
                    base_latency = 0.15
                    ebno_threshold = 9.0
                    decay_factor = (ebno_threshold - start_ebno) / np.log(base_latency / target_latency)
                    metric_values = base_latency * np.exp(-(ebno_db_range - start_ebno) / decay_factor)
                    metric_values[ebno_db_range >= ebno_threshold] = np.minimum(metric_values[ebno_db_range >= ebno_threshold], target_latency * 0.8)
                metric_values = np.clip(metric_values, 0.01, 0.5)
            elif metric_name == "energy_per_bit_pj":
                # 6G target: 1 pJ/bit
                # Perfect CSI meets 6G target at ~5 dB, Imperfect at ~9 dB
                start_ebno = -5.0
                target_energy = 1.0  # pJ/bit
                if perfect_csi:
                    base_energy = 0.8
                    ebno_threshold = 5.0
                    metric_values = np.where(ebno_db_range < ebno_threshold,
                                            base_energy - (base_energy - target_energy * 0.7) * (ebno_db_range - start_ebno) / (ebno_threshold - start_ebno),
                                            target_energy * 0.7)
                else:
                    base_energy = 2.0
                    ebno_threshold = 9.0
                    decay_factor = (ebno_threshold - start_ebno) / np.log(base_energy / target_energy)
                    metric_values = base_energy * np.exp(-(ebno_db_range - start_ebno) / decay_factor)
                    metric_values[ebno_db_range >= ebno_threshold] = np.minimum(metric_values[ebno_db_range >= ebno_threshold], target_energy * 0.9)
                metric_values = np.clip(metric_values, 0.1, 5.0)
            elif metric_name == "throughput_bits":
                # 6G target: 1 Tbps peak, 1 Gbps user-experienced
                base_throughput = 1e8 if not perfect_csi else 2e8
                growth_factor = np.exp((ebno_db_range + 5) / 4)
                metric_values = base_throughput * growth_factor
                metric_values = np.maximum(metric_values, 0)
            elif metric_name == "spectral_efficiency":
                # 6G target: Peak 100 bits/s/Hz
                base_se = 0.5 if not perfect_csi else 0.8
                growth_factor = np.exp((ebno_db_range + 5) / 3)
                metric_values = base_se * growth_factor
                metric_values = np.clip(metric_values, 0, 120.0)
            
            csi_label = "Perfect" if perfect_csi else "Imperfect"
            ax.plot(ebno_db_range, metric_values, marker='o', linewidth=2, 
                   markersize=8, label=f"{csi_label} CSI")
        
        ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
        ylabel = metric_name.upper().replace('_', ' ')
        # Add units
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
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} Comparison - 3GPP {TGPP_RELEASE} Baseline (6G Studies)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        if metric_name in ["ber", "bler", "outage_probability"]:
            ax.set_yscale('log')
        
        # Add 6G target lines for all critical metrics
        if metric_name == "ber":
            ax.axhline(y=1e-9, color='r', linestyle='--', linewidth=2, alpha=0.7, label='6G Target (< 10⁻⁹)')
            ax.legend(fontsize=10)
        elif metric_name == "bler":
            ax.axhline(y=1e-9, color='r', linestyle='--', linewidth=2, alpha=0.7, label='6G Target (< 10⁻⁹)')
            ax.legend(fontsize=10)
        elif metric_name == "air_interface_latency_ms":
            ax.axhline(y=0.1, color='r', linestyle='--', linewidth=2, alpha=0.7, label='6G Target (< 0.1 ms)')
            ax.legend(fontsize=10)
        elif metric_name == "energy_per_bit_pj":
            ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='6G Target (1 pJ/bit)')
            ax.legend(fontsize=10)
        elif metric_name == "outage_probability":
            ax.axhline(y=1e-6, color='r', linestyle='--', linewidth=2, alpha=0.7, label='6G Target (< 10⁻⁶)')
            ax.legend(fontsize=10)
        elif metric_name == "spectral_efficiency":
            ax.axhline(y=100.0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='6G Target (100 bits/s/Hz)')
            ax.legend(fontsize=10)
        elif metric_name == "throughput_bits":
            # Add annotation for 6G target (1 Tbps = 1e12 bits)
            ax.axhline(y=1e12, color='r', linestyle='--', linewidth=2, alpha=0.7, label='6G Target (1 Tbps)')
            ax.axhline(y=1e9, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='6G User Target (1 Gbps)')
            ax.legend(fontsize=10)
        plt.tight_layout()
        plot_file = plots_dir / f"3gpp_release20_{metric_name}_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved comparison plot: {plot_file.name}")
    
    # Create example simulation_results.json
    example_results = {
        "baseline_type": "3GPP Release 20 (Formal 6G Studies - IMT-2030)",
        "3gpp_release": TGPP_RELEASE,
        "3gpp_release_year": TGPP_RELEASE_YEAR,
        "3gpp_release_description": TGPP_RELEASE_DESCRIPTION,
        "3gpp_release_note": TGPP_RELEASE_NOTE,
        "6g_standards_timeline": {
            "release_19": "Bridge/Transition to 6G - Early 6G research and requirement studies (5G-Advanced Phase 2)",
            "release_20": "Formal 6G Studies (IMT-2030) - AI/ML, ISAC, NTN enhancements",
            "release_21": "Official start of normative 6G work - First 6G specifications"
        },
        "profile": "6G_Compliant",
        "scenario": "umi",
        "estimator": "ls",
        "ebno_db": ebno_db_range.tolist(),
        "config": {
            "fft_size": 512,
            "num_bs_ant": 32,
            "num_ut": 8,
            "num_ut_ant": 2,
            "num_ofdm_symbols": 14,
            "num_bits_per_symbol": 2,
            "coderate": 0.5
        },
        "runs": [
            {
                "perfect_csi": False,
                "metrics": [
                    {
                        "ebno_db": float(ebno),
                        "overall": {
                            "ber": float(np.clip(1e-1 * 10**(-(ebno - (-5)) / (9.0 - (-5)) * 2), 1e-11, 1.0)) if ebno < 9.0 else float(np.maximum(1e-1 * 10**(-(ebno - (-5)) / (9.0 - (-5)) * 2), 1e-10)),
                            "bler": float(np.clip(2e-1 * 10**(-(ebno - (-5)) / (9.0 - (-5)) * 2), 1e-11, 1.0)) if ebno < 9.0 else float(np.maximum(2e-1 * 10**(-(ebno - (-5)) / (9.0 - (-5)) * 2), 1e-10)),
                            "throughput_bits": int(1e8 * np.exp((ebno + 5) / 4)),
                            "spectral_efficiency": float(np.clip(0.5 * np.exp((ebno + 5) / 3), 0, 120.0)),
                            "outage_probability": float(np.clip(1e-2 * 10**(-(ebno - (-5)) / (9.0 - (-5)) * 1.5), 1e-9, 1.0)) if ebno < 9.0 else float(np.maximum(1e-2 * 10**(-(ebno - (-5)) / (9.0 - (-5)) * 1.5), 1e-8)),
                            "air_interface_latency_ms": float(0.15 * np.exp(-(ebno - (-5)) / ((9.0 - (-5)) / np.log(0.15 / 0.1)))) if ebno < 9.0 else float(np.minimum(0.15 * np.exp(-(ebno - (-5)) / ((9.0 - (-5)) / np.log(0.15 / 0.1))), 0.08)),
                            "energy_per_bit_pj": float(2.0 * np.exp(-(ebno - (-5)) / ((9.0 - (-5)) / np.log(2.0 / 1.0)))) if ebno < 9.0 else float(np.minimum(2.0 * np.exp(-(ebno - (-5)) / ((9.0 - (-5)) / np.log(2.0 / 1.0))), 0.9))
                        }
                    } for ebno in ebno_db_range
                ]
            },
            {
                "perfect_csi": True,
                "metrics": [
                    {
                        "ebno_db": float(ebno),
                        "overall": {
                            "ber": float(np.clip(1e-2 * 10**(-(ebno - (-5)) / (5.0 - (-5)) * 2), 1e-11, 1.0)) if ebno < 5.0 else float(np.maximum(1e-2 * 10**(-(ebno - (-5)) / (5.0 - (-5)) * 2), 1e-10)),
                            "bler": float(np.clip(5e-2 * 10**(-(ebno - (-5)) / (5.0 - (-5)) * 2), 1e-11, 1.0)) if ebno < 5.0 else float(np.maximum(5e-2 * 10**(-(ebno - (-5)) / (5.0 - (-5)) * 2), 1e-10)),
                            "throughput_bits": int(2e8 * np.exp((ebno + 5) / 4)),
                            "spectral_efficiency": float(np.clip(0.8 * np.exp((ebno + 5) / 3), 0, 120.0)),
                            "outage_probability": float(np.clip(1e-4 * 10**(-(ebno - (-5)) / (5.0 - (-5)) * 1.5), 1e-9, 1.0)) if ebno < 5.0 else float(np.maximum(1e-4 * 10**(-(ebno - (-5)) / (5.0 - (-5)) * 1.5), 1e-8)),
                            "air_interface_latency_ms": float(0.08 - (0.08 - 0.08) * (ebno - (-5)) / (5.0 - (-5))) if ebno < 5.0 else float(0.08),
                            "energy_per_bit_pj": float(0.8 - (0.8 - 0.7) * (ebno - (-5)) / (5.0 - (-5))) if ebno < 5.0 else float(0.7)
                        }
                    } for ebno in ebno_db_range
                ]
            }
        ],
        "duration": 3600.0
    }
    
    results_file = baseline_dir / "simulation_results.json"
    with open(results_file, 'w') as f:
        json.dump(example_results, f, indent=2)
    print(f"  ✓ Saved example results JSON: {results_file.name}")
    
    # Create README
    readme_content = f"""# 3GPP {TGPP_RELEASE} Baseline Results (6G Studies)

This directory contains baseline/example results for 6G smart factory simulations based on **3GPP {TGPP_RELEASE}** ({TGPP_RELEASE_YEAR}).

## 3GPP Release Information

- **Release**: {TGPP_RELEASE}
- **Year**: {TGPP_RELEASE_YEAR}
- **Description**: {TGPP_RELEASE_DESCRIPTION}
- **Note**: {TGPP_RELEASE_NOTE}

## 6G Standards Timeline (IMT-2030)

**Important**: {TGPP_RELEASE} marks the **commencement of formal 6G studies** and contains the first full 6G standards in alignment with IMT-2030 objectives.

- **Release 19** (2024): Bridge/Transition to 6G - Early 6G research and requirement studies (5G-Advanced Phase 2)
- **Release 20** ({TGPP_RELEASE_YEAR}): **Formal 6G Studies (IMT-2030)** - AI/ML for Air Interface, ISAC, NTN enhancements, Public Safety
- **Release 21** (Expected 2027+): Official start of normative 6G work - First 6G specifications

**Relevance**: {TGPP_RELEASE} provides the first formal 6G baseline with comprehensive 6G studies, making it the appropriate baseline for 6G simulation studies.

## Directory Structure

```
3gpp_release20_baseline/
├── matrices/          # NumPy arrays (.npy files) for each metric (3GPP Release 20)
├── plots/            # Visualization plots (.png files) for each metric (3GPP Release 20)
└── simulation_results.json  # Complete simulation results in JSON format
```

## Metrics

### Per-Stream Metrics (Matrices)
These metrics are computed per stream and saved as 2D matrices `[num_ebno_points, num_streams]`:
- **ber**: Bit Error Rate per stream
- **bler**: Block Error Rate per stream
- **throughput_bits**: Throughput in bits per stream
- **decoder_iter_avg**: Average LDPC decoder iterations per stream
- **sinr_db**: Signal-to-Interference-plus-Noise Ratio in dB per stream

### Overall Metrics (Vectors)
These metrics are computed overall and saved as 1D vectors `[num_ebno_points]`:
- **ber**: Overall Bit Error Rate
- **bler**: Overall Block Error Rate
- **nmse_db**: Normalized Mean Squared Error in dB (channel estimation quality)
- **evm_percent**: Error Vector Magnitude in percent (modulation quality)
- **sinr_db**: Overall Signal-to-Interference-plus-Noise Ratio in dB
- **decoder_iter_avg**: Average LDPC decoder iterations
- **throughput_bits**: Total throughput in bits
- **spectral_efficiency**: Bits per resource element
- **fairness_jain**: Jain's fairness index (0-1, higher is better)

## File Naming Convention

### Matrices
- Per-stream: `3gpp_release20_{{metric}}_per_stream_{{csi}}_run{{idx}}.npy`
  - Example: `3gpp_release20_ber_per_stream_imperfect_run0.npy`
- Overall: `3gpp_release20_{{metric}}_overall_{{csi}}_run{{idx}}.npy`
  - Example: `3gpp_release20_ber_overall_perfect_run0.npy`

### Plots
- Comparison: `3gpp_release20_{{metric}}_comparison.png` (both CSI conditions, 3GPP Release 20 baseline)
  - Example: `3gpp_release20_ber_comparison.png`

Where:
- `{{metric}}`: Metric name (ber, bler, sinr_db, etc.)
- `{{csi}}`: CSI condition (imperfect or perfect)
- `{{idx}}`: Run index (typically 0)

## Loading Results

### Load NumPy Arrays
```python
import numpy as np

# Load per-stream matrix
ber_matrix = np.load('matrices/3gpp_release20_ber_per_stream_imperfect_run0.npy')
print(f"Shape: {{ber_matrix.shape}}")  # [num_ebno, num_streams]

# Load overall vector
ber_vector = np.load('matrices/3gpp_release20_ber_overall_perfect_run0.npy')
print(f"Shape: {{ber_vector.shape}}")  # [num_ebno]
```

### Load JSON Results
```python
import json

with open('simulation_results.json', 'r') as f:
    results = json.load(f)

# Access metrics
for run in results['runs']:
    csi_str = "Perfect" if run['perfect_csi'] else "Imperfect"
    print(f"{{csi_str}} CSI:")
    for metric in run['metrics']:
        ebno = metric['ebno_db']
        ber = metric['overall']['ber']
        print(f"  Eb/No={{ebno:.1f}} dB: BER={{ber:.3e}}")
```

## Simulation Parameters

- **Scenario**: UMi (Urban Microcell)
- **Estimator**: LS (Least Squares channel estimation)
- **Eb/No Range**: -5 to 9 dB (step: 2 dB)
- **FFT Size**: 512
- **BS Antennas**: 32
- **User Terminals**: 8
- **UT Antennas**: 2
- **OFDM Symbols**: 14
- **Modulation**: QPSK (2 bits/symbol)
- **Code Rate**: 0.5

## Notes

- This is baseline/example data for demonstration purposes based on **3GPP {TGPP_RELEASE}**
- Actual simulation results will replace these when simulations are run
- All plots are saved at 300 DPI for publication quality
- Matrices use NumPy format for efficient storage and loading
- All metrics, plots, and files are labeled with 3GPP Release 20 for traceability

## References

- 3GPP {TGPP_RELEASE} ({TGPP_RELEASE_YEAR}): {TGPP_RELEASE_DESCRIPTION}
- 6G (IMT-2030) Timeline: Release 20 (Formal Studies - Current), Release 21 (Normative Work)
- For more information on 3GPP releases, visit: https://www.3gpp.org/specifications/releases
- IMT-2030 Framework: https://www.itu.int/en/ITU-R/study-groups/rsg5/rwp5d/imt-2030/Pages/default.aspx
"""
    
    readme_file = baseline_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    print(f"  ✓ Saved README: {readme_file.name}")
    
    print()
    print("=" * 80)
    print("Baseline results created successfully!")
    print(f"Location: {baseline_dir}")
    print("=" * 80)

if __name__ == "__main__":
    create_baseline_results()

