"""
Plotting utilities for 6G simulation results.

This module provides functions to generate various plots from simulation results,
including per-stream metrics, overall metrics, and comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any


# Metric definitions
PER_STREAM_METRICS = [
    "ber", "bler", "throughput_bits", "decoder_iter_avg", "sinr_db",
    "snr_db", "channel_capacity", "outage_probability"
]

OVERALL_METRICS = [
    "ber", "bler", "nmse_db", "evm_percent", "sinr_db", "snr_db",
    "decoder_iter_avg", "throughput_bits", "spectral_efficiency", "fairness_jain",
    "channel_capacity", "outage_probability", "air_interface_latency_ms", "energy_per_bit_pj"
]


def _get_ylabel(metric_name: str) -> str:
    """Get formatted y-axis label for a metric."""
    ylabel = metric_name.upper().replace('_', ' ')
    if metric_name in ["sinr_db", "snr_db"]:
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
    return ylabel


def _add_6g_target_lines(ax, metric_name: str):
    """Add 6G target lines to plot if applicable."""
    if metric_name == "air_interface_latency_ms":
        ax.axhline(y=0.1, color='r', linestyle='--', linewidth=2, label='6G Target (< 0.1 ms)')
        ax.legend(fontsize=10)
    elif metric_name == "energy_per_bit_pj":
        ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='6G Target (1 pJ/bit)')
        ax.legend(fontsize=10)
    elif metric_name == "outage_probability":
        ax.axhline(y=1e-6, color='r', linestyle='--', linewidth=2, label='6G Target (< 1e-6)')
        ax.legend(fontsize=10)


def save_metric_matrices_and_plots(
    results: dict,
    run_dir: Path,
    baseline_results: Optional[dict] = None
):
    """
    Save matrices (numpy arrays) and plots for each metric, organized per run.
    Includes baseline comparison if baseline_results is provided.
    
    Args:
        results: Simulation results dictionary
        run_dir: Directory to save results for this run
        baseline_results: Optional baseline results dictionary for comparison
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir = run_dir / "matrices"
    plots_dir = run_dir / "plots"
    matrices_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    runs = results.get("runs", [])
    
    # Collect data for each metric across all runs and Eb/No points
    for run_idx, run in enumerate(runs):
        csi_str = "perfect" if run.get("perfect_csi") else "imperfect"
        metrics_list = run.get("metrics", [])
        
        if not metrics_list:
            continue
        
        # Extract Eb/No values
        ebno_values = [m["ebno_db"] for m in metrics_list]
        
        # Process per-stream metrics (matrices: [num_ebno, num_streams])
        for metric_name in PER_STREAM_METRICS:
            metric_data = []
            for metric_entry in metrics_list:
                per_stream = metric_entry.get("per_stream", {})
                data = per_stream.get(metric_name)
                if data is not None:
                    metric_data.append(np.array(data))
                else:
                    # Fill with NaN if missing - find num_streams from first valid entry
                    prev_data = next(
                        (m.get("per_stream", {}).get(metric_name)
                         for m in metrics_list
                         if m.get("per_stream", {}).get(metric_name) is not None),
                        None
                    )
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
                ylabel = _get_ylabel(metric_name)
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
        for metric_name in OVERALL_METRICS:
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
                ylabel = _get_ylabel(metric_name)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_title(f"{ylabel} - {csi_str.upper()} CSI", fontsize=14)
                ax.grid(True, alpha=0.3)
                if metric_name in ["ber", "bler", "outage_probability"]:
                    ax.set_yscale('log')
                _add_6g_target_lines(ax, metric_name)
                plt.tight_layout()
                plot_file = plots_dir / f"{metric_name}_overall_{csi_str}_run{run_idx}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ Saved plot: {plot_file.name}")
    
    # Create comparison plots (both CSI conditions on same plot)
    for metric_name in OVERALL_METRICS:
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False
        
        # Plot 6G simulation results
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
                       markersize=8, label=f"6G Simulation - {csi_str} CSI")
                has_data = True
        
        # Plot baseline results if available
        if baseline_results is not None:
            baseline_runs = baseline_results.get("runs", [])
            for run in baseline_runs:
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
                    # Use dashed line style for baseline
                    ax.plot(ebno_values, metric_values, marker='s', linewidth=2,
                           markersize=6, linestyle='--', alpha=0.7,
                           label=f"3GPP Release 19 Baseline - {csi_str} CSI")
                    has_data = True
        
        if has_data:
            ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=12)
            ylabel = _get_ylabel(metric_name)
            ax.set_ylabel(ylabel, fontsize=12)
            title = f"{ylabel} Comparison"
            if baseline_results is not None:
                title += " (6G vs 3GPP Release 19 Baseline)"
            ax.set_title(title, fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            if metric_name in ["ber", "bler", "outage_probability"]:
                ax.set_yscale('log')
            _add_6g_target_lines(ax, metric_name)
            plt.tight_layout()
            plot_file = plots_dir / f"{metric_name}_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved comparison plot: {plot_file.name}")


def plot_simulation_results(results: dict, output_dir: str):
    """
    Generate and save plots for simulation results.
    
    Args:
        results: Simulation results dictionary
        output_dir: Output directory for plots
    """
    from datetime import datetime

    scenario = results.get("scenario", "unknown").upper()
    estimator = results.get("estimator", "estimator").upper()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.set_xlabel(r"$E_b/N_0$ (dB)")
    ax1.set_ylabel("BER")
    ax1.set_yscale('log')
    ax1.grid(which="both", alpha=0.3)
    ax1.set_title(f"Bit Error Rate - {scenario} ({estimator})")

    ax2.set_xlabel(r"$E_b/N_0$ (dB)")
    ax2.set_ylabel("BLER")
    ax2.set_yscale('log')
    ax2.grid(which="both", alpha=0.3)
    ax2.set_title(f"Block Error Rate - {scenario} ({estimator})")

    colors = ['r', 'b', 'g', 'm', 'c']
    linestyles = ['-', '--', '-.', ':']

    runs = results.get("runs", [])
    for idx, run in enumerate(runs):
        metrics = run.get("metrics", [])
        if not metrics:
            continue
        ebno = [m["ebno_db"] for m in metrics]
        ber = [m["overall"].get("ber") for m in metrics]
        bler = [m["overall"].get("bler") for m in metrics]
        if not any(val is not None for val in ber):
            continue
        csi_str = "Perfect" if run.get("perfect_csi") else "Imperfect"
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]

        ax1.semilogy(
            ebno,
            ber,
            color=color,
            linestyle=linestyle,
            marker='o',
            label=f"{csi_str} CSI",
            linewidth=2,
            markersize=6,
        )
        ax2.semilogy(
            ebno,
            bler,
            color=color,
            linestyle=linestyle,
            marker='s',
            label=f"{csi_str} CSI",
            linewidth=2,
            markersize=6,
        )

    ax1.set_ylim([1e-6, 1])
    ax2.set_ylim([1e-6, 1])
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    plt.tight_layout()

    plot_filename = f"{output_dir}/simulation_plot_{scenario}_{estimator}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_filename}")

    pdf_filename = f"{output_dir}/simulation_plot_{scenario}_{estimator}_{timestamp}.pdf"
    plt.savefig(pdf_filename, bbox_inches='tight')
    plt.close()

