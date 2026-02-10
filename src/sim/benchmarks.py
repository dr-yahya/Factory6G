#!/usr/bin/env python3
"""
Benchmark runners for 6G channel estimators and resource managers.

Provides two benchmark suites accessible via `python main.py --benchmark`:
  - estimators:        BER & latency comparison across channel estimators
  - resource-managers: BER & throughput comparison across scheduling strategies
"""

import os
import time
import gc

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.models.model import Model
from src.components.config import SystemConfig


# ---------------------------------------------------------------------------
# Channel-estimator benchmark
# ---------------------------------------------------------------------------

def run_estimator_benchmark(
    estimators: list[str] | None = None,
    ebno_db_range: np.ndarray | None = None,
    batch_size: int = 32,
    total_batches: int = 8,
    output_dir: str = "results/benchmarks",
) -> dict:
    """
    Compare channel estimators.

    Returns a dict mapping estimator keys to ``{"ber": [...], "latency_ms": [...]}``.
    A comparison plot is saved to *output_dir*.
    """
    print("=" * 70)
    print("  6G Channel Estimator Benchmark  —  BER & Latency")
    print("=" * 70)

    if ebno_db_range is None:
        ebno_db_range = np.arange(0, 21, 5)  # 0, 5, 10, 15, 20 dB

    if estimators is None:
        estimators = ["ls", "dft", "lmmse", "pso", "perfect"]

    labels = {
        "ls": "LS (NN interp.)",
        "dft": "DFT-based",
        "lmmse": "Approx. LMMSE",
        "pso": "PSO-enhanced",
        "perfect": "Perfect CSI",
    }
    # Fallback for unknown labels
    for est in estimators:
        if est not in labels:
            labels[est] = est.upper()

    results = {est: {"ber": [], "latency_ms": []} for est in estimators}

    base_config = SystemConfig(
        num_bs_ant=32,
        num_ut=8,
        num_ut_ant=1,
        fft_size=512,
        num_ofdm_symbols=14,
        num_bits_per_symbol=2,  # QPSK
        coderate=0.5,
        channel_model_type="tr38901",
        num_decoding_iter=20,
    )

    # --- run simulations ---
    for ebno_db in ebno_db_range:
        print(f"\n{'─' * 60}")
        print(f"  Eb/No = {ebno_db:.0f} dB")
        print(f"{'─' * 60}")

        for est_type in estimators:
            perfect_csi = est_type == "perfect"
            
            # Cleanup previous iteration
            tf.keras.backend.clear_session()
            gc.collect()

            try:
                model = Model(
                    config=base_config,
                    estimator_type=est_type if not perfect_csi else "ls",
                    perfect_csi=perfect_csi,
                )

                total_errors = 0
                total_bits = 0
                latency_accum = 0.0

                t0 = time.time()
                for b in range(total_batches):
                    # run_batch handles devices internally
                    batch = model.run_batch(batch_size, float(ebno_db), include_details=True)
                    bits = batch["bits"]
                    bits_hat = batch["bits_hat"]
                    errors = np.sum(bits != bits_hat)
                    total_errors += errors
                    total_bits += bits.size
                    latency_accum += batch["latency_sec"]
                    
                    if b % 10 == 0:
                        gc.collect()

                avg_ber = total_errors / total_bits if total_bits > 0 else 0.0
                avg_latency_ms = (latency_accum / total_batches) * 1000

                results[est_type]["ber"].append(avg_ber)
                results[est_type]["latency_ms"].append(avg_latency_ms)

                elapsed = time.time() - t0
                print(
                    f"  {labels.get(est_type, est_type):20s}  "
                    f"BER={avg_ber:.3e}  "
                    f"Latency={avg_latency_ms:.3f} ms  "
                    f"({elapsed:.1f}s)"
                )
                
                del model
                
            except Exception as e:
                print(f"  {est_type}: Failed - {e}")
                # Ensure lists are aligned
                if len(results[est_type]["ber"]) < len(results[est_type]["latency_ms"]) + 1:
                     results[est_type]["ber"].append(1.0)
                if len(results[est_type]["latency_ms"]) < len(results[est_type]["ber"]):
                     results[est_type]["latency_ms"].append(0.0)

    # --- plot results ---
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    markers = ["o", "s", "D", "^", "v", "x", "*", "+"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c", "#34495e", "#7f8c8d"]

    for idx, est_type in enumerate(estimators):
        m = markers[idx % len(markers)]
        c = colors[idx % len(colors)]
        lbl = labels.get(est_type, est_type)

        ber_vals = results[est_type]["ber"]
        ber_plot = [max(v, 1e-12) for v in ber_vals]
        ax1.semilogy(ebno_db_range, ber_plot, marker=m, color=c,
                     label=lbl, linewidth=2, markersize=8)

        ax2.plot(ebno_db_range, results[est_type]["latency_ms"],
                 marker=m, color=c, label=lbl, linewidth=2, markersize=8)

    ax1.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=13)
    ax1.set_ylabel("Bit Error Rate (BER)", fontsize=13)
    ax1.set_title("BER vs Eb/No", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_ylim([1e-6, 1])

    ax2.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=13)
    ax2.set_ylabel("Air-Interface Latency (ms)", fontsize=13)
    ax2.set_title("Latency vs Eb/No", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"6G Channel Estimator Benchmark\n"
        f"(MIMO {base_config.num_bs_ant}×{base_config.num_ut}, "
        f"QPSK, LDPC R={base_config.coderate}, "
        f"TR 38.901 UMi)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "estimator_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Benchmark complete. Plot saved to {plot_path}")
    return results


# ---------------------------------------------------------------------------
# Resource-manager benchmark
# ---------------------------------------------------------------------------

def run_resource_manager_benchmark(
    managers_list: list[str] | None = None,
    ebno_db_range: np.ndarray | None = None,
    batch_size: int = 32,
    total_batches: int = 10,
    num_active: int = 2,
    cnn_model_path: str = "models/cnn_resource_manager.h5",
    output_dir: str = "results/benchmarks",
) -> dict:
    """
    Compare resource managers.

    Returns a dict mapping manager names to ``{"ber": [...], "throughput": [...]}``.
    A comparison plot is saved to *output_dir*.
    """
    from src.models.resource_manager import (
        StaticResourceManager,
        RoundRobinResourceManager,
        MaxThroughputResourceManager,
        ProportionalFairResourceManager,
    )
    from src.models.cnn_resource_manager import CNNResourceManager

    print("=" * 70)
    print("  6G Resource Management Benchmark  —  BER & Throughput")
    print("=" * 70)

    if ebno_db_range is None:
        ebno_db_range = np.array([0, 10, 20])

    if managers_list is None:
        managers_list = ["static", "round_robin", "max_throughput", "pf", "cnn"]

    # --- Factory logic to create managers based on config strings ---
    active_managers = {}
    for name in managers_list:
        name_lower = name.lower()
        if "static" in name_lower:
            mgr = StaticResourceManager(active_ut_mask=[1] * 8)
            label = "Static (All)"
        elif "round" in name_lower or "robin" in name_lower:
            mgr = RoundRobinResourceManager(num_active=num_active)
            label = "Round Robin"
        elif "max" in name_lower or "throughput" in name_lower:
            mgr = MaxThroughputResourceManager(num_active=num_active)
            label = "Max Throughput"
        elif "prop" in name_lower or "pf" in name_lower:
            mgr = ProportionalFairResourceManager(num_active=num_active)
            label = "Prop. Fair"
        elif "cnn" in name_lower:
            if os.path.exists(cnn_model_path):
                mgr = CNNResourceManager(model_path=cnn_model_path)
                label = "CNN Manager"
            else:
                print(f"Warning: CNN model not found at {cnn_model_path}, skipping.")
                continue
        else:
            print(f"Unknown resource manager: {name}")
            continue
        
        active_managers[label] = mgr
    
    if not active_managers:
        print("No valid resource managers selected.")
        return {}
        
    results = {name: {"ber": [], "throughput": []} for name in active_managers}

    base_config = SystemConfig(
        num_bs_ant=32,
        num_ut=8,
        num_ut_ant=1,
        fft_size=512,
        num_ofdm_symbols=14,
        num_bits_per_symbol=2,  # QPSK
        coderate=0.5,
        channel_model_type="tr38901",
        num_decoding_iter=20,
    )

    # Iteration
    for ebno_db in ebno_db_range:
        print(f"\nProcessing Eb/No = {ebno_db} dB...")
        for name, manager in active_managers.items():
            print(f"  Manager: {name}", end="", flush=True)
            
            tf.keras.backend.clear_session()
            gc.collect()

            try:
                # IMPORTANT: GPU allowed by default (no forced CPU block)
                model = Model(
                    config=base_config,
                    estimator_type="lmmse",
                    resource_manager=manager,
                    perfect_csi=False,
                )

                total_errors = 0
                total_bits = 0
                total_success_bits = 0

                start_time = time.time()
                for b in range(total_batches):
                    batch = model.run_batch(batch_size, float(ebno_db), include_details=False)
                    b_orig = batch["bits"]
                    b_hat = batch["bits_hat"]
                    
                    errors = np.sum(b_orig != b_hat)
                    batch_bits = b_orig.size
                    
                    total_errors += errors
                    total_bits += batch_bits
                    total_success_bits += max(0, batch_bits - errors)

                    if b % 10 == 0:
                        print(".", end="", flush=True)

                avg_ber = total_errors / total_bits if total_bits > 0 else 0.0
                results[name]["ber"].append(avg_ber)
                results[name]["throughput"].append(total_success_bits)

                elapsed = time.time() - start_time
                print(f" Done. BER: {avg_ber:.2e} ({elapsed:.1f}s)")
                
                del model

            except Exception as e:
                print(f" Failed: {e}")
                import traceback
                traceback.print_exc()

    # --- plot results ---
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for name in active_managers:
        if not results[name]["ber"]: continue
        ax1.semilogy(ebno_db_range, results[name]["ber"], marker="o", label=name)
        ax2.plot(ebno_db_range, results[name]["throughput"], marker="s", label=name)

    ax1.grid(True, which="both", ls="-")
    ax1.set_xlabel("Eb/No (dB)")
    ax1.set_ylabel("Bit Error Rate (BER)")
    ax1.set_title("BER Comparison")
    ax1.legend()

    ax2.grid(True, which="both", ls="-")
    ax2.set_xlabel("Eb/No (dB)")
    ax2.set_ylabel("Total Success Bits")
    ax2.set_title("Throughput Comparison")
    ax2.legend()

    plt.suptitle(
        f"6G Resource Manager Benchmark\n"
        f"(MIMO 32x8, Scen: UMi, num_active={num_active})"
    )

    plot_path = os.path.join(output_dir, "resource_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nBenchmark complete. Plot saved to {plot_path}")
    return results
