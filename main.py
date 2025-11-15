#!/usr/bin/env python3
from __future__ import annotations
"""
Main simulation script for 6G Smart Factory Physical Layer System.

This script runs BER/BLER simulations for the complete OFDM-MIMO system with
different configurations (scenarios, CSI conditions, channel estimators, etc.).
It provides a comprehensive command-line interface for system evaluation and
performance analysis.

Theory:
    BER/BLER Simulation:
    
    The script performs Monte Carlo simulations to estimate:
    - Bit Error Rate (BER): P(b̂ ≠ b) = E[I(b̂ ≠ b)]
    - Block Error Rate (BLER): P(∃ i: b̂[i] ≠ b[i]) = 1 - (1 - BER)^n
    
    Simulation Process:
    1. Generate random information bits
    2. Transmit through system (encoder → modulator → channel → receiver → decoder)
    3. Compare received bits with transmitted bits
    4. Compute error rates: BER = (# bit errors) / (# total bits)
    5. Repeat for multiple channel realizations (Monte Carlo)
    
    Stopping Criteria:
    - Maximum iterations: Stop after max_mc_iter channel realizations
    - Target block errors: Stop after num_target_block_errors errors
    - Target BLER: Stop if BLER < target_bler (early stopping)
    
    Eb/No Range:
    - Eb/No: Energy per bit to noise power spectral density ratio
    - Eb/No (dB) = 10·log10(Eb/N0)
    - Higher Eb/No: Better performance (lower BER/BLER)
    - Typical range: -5 dB to 20 dB for evaluation
    
    Performance Analysis:
    - Compare perfect vs imperfect CSI
    - Compare different channel estimators
    - Compare different scenarios (UMi, UMa, RMa)
    - Generate performance curves (BER/BLER vs Eb/No)

Usage:
    python main.py [options]

Examples:
    # Run default simulation (UMi, perfect and imperfect CSI)
    python main.py

    # Run with specific scenario
    python main.py --scenario uma

    # Run only perfect CSI
    python main.py --perfect-csi-only

    # Custom Eb/No range
    python main.py --ebno-min -5 --ebno-max 15 --ebno-step 2
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Optional

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from src.models.resource_manager import StaticResourceManager
from src.sim.scenarios import SCENARIO_PRESETS, ScenarioSpec


class MetricsAccumulator:
    """Accumulates per-batch metrics to compute BER/BLER and diagnostics."""

    def __init__(self, config: "SystemConfig"):
        self.config = config
        shape = (config.num_tx, config.num_streams_per_tx)
        self.bit_errors = np.zeros(shape, dtype=np.int64)
        self.bits_total = np.zeros(shape, dtype=np.int64)
        self.block_errors = np.zeros(shape, dtype=np.int64)
        self.blocks_total = np.zeros(shape, dtype=np.int64)
        self.success_bits = np.zeros(shape, dtype=np.int64)
        self.decoder_iter_sum = np.zeros(shape, dtype=np.float64)
        self.decoder_iter_count = np.zeros(shape, dtype=np.int64)
        self.nmse_num = 0.0
        self.nmse_den = 0.0
        self.evm_num = 0.0
        self.evm_den = 0.0
        self.sinr_sum = np.zeros(shape, dtype=np.float64)
        self.sinr_count = np.zeros(shape, dtype=np.int64)
        
        # New metrics: Physical layer latency, energy, outage, capacity, SNR
        self.latency_sum = 0.0  # Total latency in seconds
        self.latency_count = 0  # Number of latency measurements
        self.energy_sum = 0.0  # Total energy consumed in Joules
        self.outage_count = np.zeros(shape, dtype=np.int64)  # Outage events (SINR < threshold)
        self.outage_threshold_db = -5.0  # Outage threshold in dB
        self.snr_sum = np.zeros(shape, dtype=np.float64)  # SNR (without interference)
        self.snr_count = np.zeros(shape, dtype=np.int64)

    def update(self, batch: dict):
        bits = batch["bits"]
        bits_hat = batch["bits_hat"]
        diff = np.not_equal(bits, bits_hat)
        bit_errors = diff.sum(axis=-1).sum(axis=0)
        self.bit_errors += bit_errors

        num_blocks_batch = bits.shape[0]
        bits_per_block = bits.shape[-1]
        self.bits_total += bits_per_block * num_blocks_batch
        self.blocks_total += num_blocks_batch

        block_error_mask = diff.any(axis=-1)
        self.block_errors += block_error_mask.astype(np.int64).sum(axis=0)
        success_counts = (~block_error_mask).sum(axis=0)
        self.success_bits += success_counts * bits_per_block

        decoder_iter = batch["decoder_iterations"]
        self.decoder_iter_sum += decoder_iter.sum(axis=0)
        self.decoder_iter_count += num_blocks_batch

        h = batch["channel"]
        h_hat = batch["channel_hat"]
        diff_h = h - h_hat
        self.nmse_num += np.sum(np.abs(diff_h) ** 2)
        self.nmse_den += np.sum(np.abs(h) ** 2) + 1e-12

        x = batch["qam"]
        x_hat = batch["qam_hat"]
        self.evm_num += np.sum(np.abs(x_hat - x) ** 2)
        self.evm_den += np.sum(np.abs(x) ** 2) + 1e-12

        no_eff = batch["no_eff"]
        signal_power = np.abs(x) ** 2
        sinr_linear = np.divide(
            signal_power,
            no_eff,
            out=np.zeros_like(signal_power, dtype=np.float64),
            where=no_eff > 0,
        )
        sinr_linear = sinr_linear.reshape(-1, self.config.num_tx, self.config.num_streams_per_tx)
        self.sinr_sum += sinr_linear.sum(axis=0)
        self.sinr_count += sinr_linear.shape[0]
        
        # Compute SNR (signal power / noise power, without interference)
        # For SNR, we use the noise variance directly (no interference component)
        noise_power = batch.get("noise_power", None)
        if noise_power is None:
            # If noise_power not provided, use no_eff as approximation (will include some interference)
            noise_power = no_eff
        else:
            # Ensure noise_power has the same shape as signal_power for division
            if np.isscalar(noise_power) or (isinstance(noise_power, np.ndarray) and noise_power.size == 1):
                # Broadcast scalar noise power to match signal_power shape
                noise_power = np.full_like(signal_power, float(noise_power))
            elif isinstance(noise_power, np.ndarray):
                # Reshape to match signal_power if needed
                if noise_power.shape != signal_power.shape:
                    # Try to broadcast
                    try:
                        noise_power = np.broadcast_to(noise_power, signal_power.shape)
                    except:
                        # If broadcast fails, use mean
                        noise_power = np.full_like(signal_power, float(np.mean(noise_power)))
        
        snr_linear = np.divide(
            signal_power,
            noise_power,
            out=np.zeros_like(signal_power, dtype=np.float64),
            where=noise_power > 0,
        )
        snr_linear = snr_linear.reshape(-1, self.config.num_tx, self.config.num_streams_per_tx)
        self.snr_sum += snr_linear.sum(axis=0)
        self.snr_count += snr_linear.shape[0]
        
        # Compute outage probability: P(SINR < threshold)
        sinr_db = np.where(sinr_linear > 0, 10 * np.log10(sinr_linear), -np.inf)
        outage_mask = sinr_db < self.outage_threshold_db
        self.outage_count += outage_mask.astype(np.int64).sum(axis=0)
        
        # Track latency if provided
        if "latency_sec" in batch:
            self.latency_sum += batch["latency_sec"]
            self.latency_count += 1
        
        # Track energy if provided
        if "energy_joules" in batch:
            self.energy_sum += batch["energy_joules"]

    def total_block_errors(self) -> int:
        return int(self.block_errors.sum())

    def total_blocks(self) -> int:
        return int(self.blocks_total.sum())

    def current_overall_bler(self) -> Optional[float]:
        total_blocks = self.total_blocks()
        if total_blocks == 0:
            return None
        return self.total_block_errors() / total_blocks

    def finalize(self) -> dict:
        total_bits = self.bits_total.sum()
        total_blocks = self.blocks_total.sum()

        ber_per_stream = np.divide(
            self.bit_errors,
            self.bits_total,
            out=np.zeros_like(self.bit_errors, dtype=np.float64),
            where=self.bits_total > 0,
        )
        bler_per_stream = np.divide(
            self.block_errors,
            self.blocks_total,
            out=np.zeros_like(self.block_errors, dtype=np.float64),
            where=self.blocks_total > 0,
        )

        overall_ber = float(self.bit_errors.sum() / total_bits) if total_bits > 0 else None
        overall_bler = float(self.block_errors.sum() / total_blocks) if total_blocks > 0 else None

        nmse = self.nmse_num / self.nmse_den if self.nmse_den > 0 else None
        nmse_db = float(10 * np.log10(nmse)) if nmse and nmse > 0 else None

        evm_ratio = self.evm_num / self.evm_den if self.evm_den > 0 else None
        evm_rms = float(np.sqrt(evm_ratio)) if evm_ratio is not None else None
        evm_percent = float(evm_rms * 100) if evm_rms is not None else None

        sinr_linear = np.divide(
            self.sinr_sum,
            self.sinr_count,
            out=np.zeros_like(self.sinr_sum),
            where=self.sinr_count > 0,
        )
        with np.errstate(divide="ignore"):
            sinr_db = np.where(sinr_linear > 0, 10 * np.log10(sinr_linear), -np.inf)

        decoder_iter_avg_per_stream = np.divide(
            self.decoder_iter_sum,
            self.decoder_iter_count,
            out=np.zeros_like(self.decoder_iter_sum),
            where=self.decoder_iter_count > 0,
        )
        decoder_iter_avg = float(self.decoder_iter_sum.sum() / self.decoder_iter_count.sum()) if self.decoder_iter_count.sum() > 0 else None

        throughput_bits_per_stream = self.success_bits
        throughput_per_ut = throughput_bits_per_stream.sum(axis=1)
        fairness = None
        if np.any(throughput_per_ut > 0):
            fairness = float(
                (throughput_per_ut.sum() ** 2) /
                (len(throughput_per_ut) * np.sum(throughput_per_ut ** 2))
            )

        total_re = (
            total_blocks
            * self.config.num_ofdm_symbols
            * self.config.fft_size
        )
        spectral_eff = (
            float(throughput_bits_per_stream.sum() / total_re)
            if total_re > 0
            else None
        )
        
        # Compute SNR (Signal-to-Noise Ratio, without interference)
        snr_linear = np.divide(
            self.snr_sum,
            self.snr_count,
            out=np.zeros_like(self.snr_sum),
            where=self.snr_count > 0,
        )
        with np.errstate(divide="ignore"):
            snr_db = np.where(snr_linear > 0, 10 * np.log10(snr_linear), -np.inf)
        
        # Compute Channel Capacity: C = log2(1 + SINR)
        # Use accumulated SINR for capacity calculation
        capacity_per_stream = np.where(
            sinr_linear > 0,
            np.log2(1 + sinr_linear),
            np.zeros_like(sinr_linear)
        )
        
        # Compute Outage Probability: P(SINR < threshold)
        outage_prob_per_stream = np.divide(
            self.outage_count,
            self.blocks_total,
            out=np.zeros_like(self.outage_count, dtype=np.float64),
            where=self.blocks_total > 0,
        )
        overall_outage_prob = (
            float(self.outage_count.sum() / total_blocks)
            if total_blocks > 0
            else None
        )
        
        # Compute Air Interface Latency (average)
        avg_latency_ms = (
            float(self.latency_sum / self.latency_count * 1000)
            if self.latency_count > 0
            else None
        )
        
        # Compute Energy per Bit (PHY): Energy / Successful Bits
        energy_per_bit_pj = None
        if self.energy_sum > 0 and throughput_bits_per_stream.sum() > 0:
            energy_per_bit_joules = self.energy_sum / throughput_bits_per_stream.sum()
            energy_per_bit_pj = float(energy_per_bit_joules * 1e12)  # Convert to pJ
        
        # Overall SNR
        overall_snr_db = (
            float(np.mean(snr_db[np.isfinite(snr_db)]))
            if np.any(np.isfinite(snr_db))
            else None
        )
        
        # Overall Channel Capacity
        overall_capacity = (
            float(np.mean(capacity_per_stream[capacity_per_stream > 0]))
            if np.any(capacity_per_stream > 0)
            else None
        )

        return {
            "per_stream": {
                "ber": ber_per_stream.tolist(),
                "bler": bler_per_stream.tolist(),
                "throughput_bits": throughput_bits_per_stream.tolist(),
                "decoder_iter_avg": decoder_iter_avg_per_stream.tolist(),
                "sinr_linear": sinr_linear.tolist(),
                "sinr_db": sinr_db.tolist(),
                "snr_linear": snr_linear.tolist(),
                "snr_db": snr_db.tolist(),
                "channel_capacity": capacity_per_stream.tolist(),
                "outage_probability": outage_prob_per_stream.tolist(),
            },
            "overall": {
                "ber": overall_ber,
                "bler": overall_bler,
                "nmse": nmse,
                "nmse_db": nmse_db,
                "evm_rms": evm_rms,
                "evm_percent": evm_percent,
                "sinr_db": float(np.mean(sinr_db[np.isfinite(sinr_db)])) if np.any(np.isfinite(sinr_db)) else None,
                "snr_db": overall_snr_db,
                "decoder_iter_avg": decoder_iter_avg,
                "throughput_bits": int(throughput_bits_per_stream.sum()),
                "spectral_efficiency": spectral_eff,
                "fairness_jain": fairness,
                "channel_capacity": overall_capacity,
                "outage_probability": overall_outage_prob,
                "air_interface_latency_ms": avg_latency_ms,
                "energy_per_bit_pj": energy_per_bit_pj,
            },
            "counts": {
                "bit_errors": int(self.bit_errors.sum()),
                "total_bits": int(total_bits),
                "block_errors": int(self.block_errors.sum()),
                "total_blocks": int(total_blocks),
                "outage_events": int(self.outage_count.sum()),
            },
        }


def configure_env(force_cpu: bool, gpu_num: int | None):
    """
    Configure environment variables BEFORE importing TensorFlow/Sionna.
    This avoids noisy CUDA library errors when GPU runtime is unavailable.
    """
    # Reduce TensorFlow log verbosity
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if force_cpu:
        # Fully disable GPU visibility for TF/XLA/JAX
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif gpu_num is not None and os.getenv("CUDA_VISIBLE_DEVICES") is None:
        # Respect explicit GPU selection if provided
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"


def setup_gpu(gpu_num: int = 0):
    """Configure GPU settings after TensorFlow is imported."""
    import tensorflow as tf  # imported late to respect env config

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✓ Using GPU: {gpus[0]}")
        except RuntimeError as e:
            print(f"Warning: {e}")
    else:
        print("⚠ No GPU found, using CPU")


def run_simulation(
    scenario: str = "umi",
    perfect_csi_list: Optional[list] = None,
    ebno_db_range: Optional[np.ndarray] = None,
    batch_size: int = 128,
    max_mc_iter: int = 1000,
    num_target_block_errors: int = 1000,
    target_bler: float = 1e-3,
    config: "SystemConfig" = None,
    save_results: bool = True,
    plot_results: bool = True,
    output_dir: str = "results",
    estimator_type: str = "ls",
    estimator_weights: str | None = None,
    estimator_kwargs: dict | None = None,
    resource_manager=None,
    resource_manager_config: Optional[dict] = None,
    profile_name: Optional[str] = None,
):
    """Run Monte Carlo simulation collecting extended diagnostics."""
    from src.models.model import Model
    from src.components.config import SystemConfig

    perfect_csi_list = perfect_csi_list or [False]
    if ebno_db_range is None:
        ebno_db_range = np.arange(0.0, 11.0, 1.0)
    ebno_db_range = np.asarray(ebno_db_range, dtype=float)

    if save_results or plot_results:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {
        "profile": profile_name,
        "scenario": scenario,
        "estimator": estimator_type,
        "ebno_db": ebno_db_range.tolist(),
        "config": config.__dict__ if config else None,
        "resource_manager": resource_manager_config,
        "runs": [],
        "duration": None,
    }

    print("=" * 80)
    print("6G Smart Factory Physical Layer Simulation")
    print("=" * 80)
    print(f"Scenario: {scenario.upper()}")
    print(f"Estimator: {estimator_type.upper()}")
    print(f"Eb/No grid: {ebno_db_range[0]:.2f} → {ebno_db_range[-1]:.2f} dB (Δ {ebno_db_range[1] - ebno_db_range[0]:.2f} dB)" if len(ebno_db_range) > 1 else f"Eb/No: {ebno_db_range[0]:.2f} dB")
    print(f"Batch size: {batch_size}")
    print(f"CSI conditions: {perfect_csi_list}")
    print("=" * 80)

    start_time = time.time()

    for perfect_csi in perfect_csi_list:
        csi_str = "Perfect" if perfect_csi else "Imperfect"
        print(f"\n[{csi_str} CSI] Running simulation...")

        model_kwargs = dict(
            scenario=scenario,
            perfect_csi=perfect_csi,
            estimator_type=estimator_type,
            estimator_weights=estimator_weights,
            estimator_kwargs=estimator_kwargs,
            resource_manager=resource_manager,
        )
        if config is not None:
            model_kwargs["config"] = config

        model = Model(**model_kwargs)

        run_entry = {
            "perfect_csi": perfect_csi,
            "metrics": [],
        }

        for ebno in ebno_db_range:
            accumulator = MetricsAccumulator(model.get_config())
            iterations = 0
            t0 = time.time()

            while iterations < max_mc_iter:
                batch_results = model.run_batch(batch_size, float(ebno), include_details=True)
                accumulator.update(batch_results)
                iterations += 1
                
                # Clear TensorFlow cache periodically to prevent memory buildup
                if iterations % 10 == 0:
                    import gc
                    gc.collect()
                    try:
                        import tensorflow as tf
                        tf.keras.backend.clear_session()
                    except:
                        pass

                if accumulator.total_block_errors() >= num_target_block_errors:
                    break
                current_bler = accumulator.current_overall_bler()
                if target_bler is not None and current_bler is not None and current_bler <= target_bler:
                    break

            finalized = accumulator.finalize()
            finalized["ebno_db"] = float(ebno)
            finalized["iterations"] = iterations
            finalized["duration_sec"] = time.time() - t0
            run_entry["metrics"].append(finalized)

            ber_value = finalized["overall"]["ber"]
            bler_value = finalized["overall"]["bler"]
            if ber_value is not None:
                if bler_value is None:
                    bler_value = float("nan")
                summary_line = (
                    f"  Eb/No={ebno:>4.1f} dB | iterations={iterations} | "
                    f"BER={ber_value:.3e} | BLER={bler_value:.3e}"
                )
            else:
                summary_line = f"  Eb/No={ebno:>4.1f} dB | iterations={iterations}"
            print(summary_line)

        results["runs"].append(run_entry)
        print(f"[{csi_str} CSI] ✓ completed")

    results["duration"] = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"Simulation completed in {results['duration']:.2f} seconds")
    print("=" * 80)

    output_path = None
    if save_results:
        output_path = save_simulation_results(results, output_dir)
        results["results_file"] = output_path

    if plot_results:
        plot_simulation_results(results, output_dir)

    return results


def save_simulation_results(results: dict, output_dir: str) -> str:
    """Persist simulation results to disk and return the path."""
    import json
    from datetime import datetime

    scenario = results.get("scenario", "unknown")
    estimator = results.get("estimator", "est")
    profile = results.get("profile")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_suffix = f"_{profile}" if profile else ""
    filename = f"{output_dir}/simulation_results_{scenario}_{estimator}{profile_suffix}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {filename}")
    return filename


def plot_simulation_results(results: dict, output_dir: str):
    """Generate and save plots for simulation results."""
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


def print_results_summary(results: dict):
    """Print concise summary tables for the collected metrics."""
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Scenario : {results.get('scenario', 'unknown')}")
    print(f"Estimator: {results.get('estimator', 'N/A').upper()}")
    if results.get("profile"):
        print(f"Profile : {results['profile']}")

    runs = results.get("runs", [])
    if not runs:
        print("No metrics available.")
        return

    for run in runs:
        csi_str = "Perfect" if run.get("perfect_csi") else "Imperfect"
        print(f"\n[{csi_str} CSI]")
        print("-" * 80)
        header = f"{'Eb/No [dB]':>10} | {'BER':>12} | {'BLER':>12} | {'SINR (dB)':>10} | {'NMSE (dB)':>10} | {'Avg Iter':>9}"
        print(header)
        print("-" * len(header))
        for metric in run.get("metrics", []):
            ebno = metric.get("ebno_db")
            overall = metric.get("overall", {})
            ber = overall.get("ber")
            bler = overall.get("bler")
            sinr_db = overall.get("sinr_db")
            nmse_db = overall.get("nmse_db")
            iter_avg = overall.get("decoder_iter_avg")
            print(
                f"{ebno:10.2f} | "
                f"{(ber if ber is not None else float('nan')):>12.3e} | "
                f"{(bler if bler is not None else float('nan')):>12.3e} | "
                f"{(sinr_db if sinr_db is not None else float('nan')):>10.3f} | "
                f"{(nmse_db if nmse_db is not None else float('nan')):>10.3f} | "
                f"{(iter_avg if iter_avg is not None else float('nan')):>9.3f}"
            )
    print("=" * 80)


def main():
    """Main entry point"""
    # Build CLI without importing TensorFlow yet
    parser = argparse.ArgumentParser(
        description="6G Smart Factory Physical Layer Simulation",
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
        '--scenario-profile',
        nargs='+',
        default=None,
        help=f"Run one or more predefined scenario presets ({', '.join(SCENARIO_PRESETS.keys())})."
    )

    parser.add_argument(
        '--list-scenarios',
        action='store_true',
        help='List available scenario presets and exit.'
    )

    parser.add_argument(
        '--estimator',
        nargs='+',
        default=['ls_nn'],
        choices=['ls', 'ls_nn', 'ls_lin', 'neural'],
        help='Channel estimator(s): ls, ls_nn, ls_lin, neural (multiple allowed)'
    )

    parser.add_argument(
        '--neural-weights',
        type=str,
        default=None,
        help='Path to pretrained neural estimator weights (required when using --estimator neural).'
    )

    parser.add_argument(
        '--neural-hidden-units',
        type=int,
        nargs='+',
        default=[32, 32],
        help='Hidden layer sizes for the neural estimator (default: 32 32).'
    )
    
    parser.add_argument(
        '--perfect-csi-only',
        action='store_true',
        help='Run only perfect CSI simulation'
    )
    
    parser.add_argument(
        '--imperfect-csi-only',
        action='store_true',
        help='Run only imperfect CSI simulation'
    )
    
    parser.add_argument(
        '--ebno-min',
        type=float,
        default=-5.0,
        help='Minimum Eb/No in dB (default: -5.0)'
    )
    
    parser.add_argument(
        '--ebno-max',
        type=float,
        default=15.0,
        help='Maximum Eb/No in dB (default: 15.0)'
    )
    
    parser.add_argument(
        '--ebno-step',
        type=float,
        default=2.0,
        help='Eb/No step size in dB (default: 2.0)'
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
        help='Force CPU execution and silence CUDA library errors'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip generating plots'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving results'
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
    
    # Resource Management (Static) flags
    parser.add_argument(
        '--use-static-rm',
        action='store_true',
        help='Enable Static Resource Manager (scheduling/power control)'
    )
    parser.add_argument(
        '--active-ut-mask',
        type=int,
        nargs='+',
        default=None,
        help='Active UT mask, e.g., 1 0 1 0'
    )
    parser.add_argument(
        '--per-ut-power',
        type=float,
        nargs='+',
        default=None,
        help='Per-UT power scaling (linear), e.g., 1.0 0.5 2.0 0.5'
    )
    parser.add_argument(
        '--pilot-reuse-factor',
        type=int,
        default=None,
        help='Pilot reuse factor placeholder (integer)'
    )
    
    args = parser.parse_args()

    if args.list_scenarios:
        print("Available scenario profiles:")
        for key, spec in SCENARIO_PRESETS.items():
            print(f"  - {key}: {spec.description}")
        return
    
    # Configure environment BEFORE importing TensorFlow/Sionna
    configure_env(force_cpu=args.cpu, gpu_num=args.gpu)

    # Now safe to import TensorFlow/Sionna and configure runtime
    import tensorflow as tf
    import sionna
    tf.get_logger().setLevel('ERROR')
    setup_gpu(args.gpu)
    
    # Set random seed (Sionna >=0.16 has no global phy seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    results_all = []

    if args.scenario_profile:
        profile_names = [name.lower() for name in args.scenario_profile]
        for profile_name in profile_names:
            if profile_name not in SCENARIO_PRESETS:
                print(f"⚠ Unknown scenario profile '{profile_name}'. Available: {', '.join(SCENARIO_PRESETS.keys())}")
                continue
            spec = SCENARIO_PRESETS[profile_name]
            scenario_name = spec.channel_scenario or args.scenario
            ebno_db_range = np.arange(spec.ebno_min, spec.ebno_max + spec.ebno_step, spec.ebno_step)

            rm = None
            if spec.resource_manager:
                rm = StaticResourceManager(**spec.resource_manager)

            estimators = spec.estimators or args.estimator
            for estimator in estimators:
                estimator_kwargs = {}
                if spec.estimator_kwargs:
                    if estimator in spec.estimator_kwargs and isinstance(spec.estimator_kwargs[estimator], dict):
                        estimator_kwargs = dict(spec.estimator_kwargs[estimator])
                    elif isinstance(spec.estimator_kwargs, dict):
                        estimator_kwargs = dict(spec.estimator_kwargs)
                estimator_weights = spec.estimator_weights
                if estimator == 'neural':
                    estimator_kwargs.setdefault('hidden_units', args.neural_hidden_units)
                    if estimator_weights is None:
                        estimator_weights = args.neural_weights

                profile_output_dir = Path(args.output_dir) / spec.name
                results = run_simulation(
                    scenario=scenario_name,
                    perfect_csi_list=spec.perfect_csi,
                    ebno_db_range=ebno_db_range,
                    batch_size=spec.batch_size,
                    max_mc_iter=spec.max_iter,
                    num_target_block_errors=spec.target_block_errors,
                    target_bler=spec.target_bler,
                    save_results=not args.no_save,
                    plot_results=not args.no_plot,
                    output_dir=str(profile_output_dir),
                    estimator_type=estimator,
                    estimator_weights=estimator_weights,
                    estimator_kwargs=estimator_kwargs,
                    resource_manager=rm,
                    resource_manager_config=spec.resource_manager,
                    profile_name=spec.name,
                )
                if spec.description:
                    print(f"\nDescription: {spec.description}")
                print_results_summary(results)
                results_all.append(results)
    else:
        # Determine CSI conditions from CLI switches
        if args.perfect_csi_only:
            perfect_csi_list = [True]
        elif args.imperfect_csi_only:
            perfect_csi_list = [False]
        else:
            perfect_csi_list = [True, False]

        ebno_db_range = np.arange(args.ebno_min, args.ebno_max + args.ebno_step, args.ebno_step)

        rm_config = None
        resource_manager = None
        if args.use_static_rm:
            rm_config = {
                "active_ut_mask": args.active_ut_mask,
                "per_ut_power": args.per_ut_power,
                "pilot_reuse_factor": args.pilot_reuse_factor,
            }
            rm_config = {k: v for k, v in rm_config.items() if v is not None}
            resource_manager = StaticResourceManager(**rm_config) if rm_config else None
        
        for estimator in args.estimator:
            estimator_kwargs = {}
            estimator_weights = None
            if estimator == 'neural':
                estimator_kwargs['hidden_units'] = args.neural_hidden_units
                estimator_weights = args.neural_weights
                if estimator_weights is None:
                    print("⚠ Neural estimator selected without --neural-weights. Provide weights before running.")

            results = run_simulation(
                scenario=args.scenario,
                perfect_csi_list=perfect_csi_list,
                ebno_db_range=ebno_db_range,
                batch_size=args.batch_size,
                max_mc_iter=args.max_iter,
                num_target_block_errors=args.target_block_errors,
                target_bler=args.target_bler,
                save_results=not args.no_save,
                plot_results=not args.no_plot,
                output_dir=args.output_dir,
                estimator_type=estimator,
                estimator_weights=estimator_weights,
                estimator_kwargs=estimator_kwargs,
                resource_manager=resource_manager,
                resource_manager_config=rm_config,
            )

            print_results_summary(results)
            results_all.append(results)

    if not results_all:
        return None

    return results_all if len(results_all) > 1 else results_all[0]


if __name__ == "__main__":
    main()

