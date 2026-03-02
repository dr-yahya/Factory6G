"""
Unified simulation loop for 6G Factory benchmarks.
Handles setup, execution, result saving, and plotting for:
1. Channel Estimator comparisons
2. Resource Manager comparisons
"""

import os
import time
import math
import numpy as np
import tensorflow as tf
from typing import Dict, Any
from statistics import NormalDist

from src.models.model import Model
from src.sim.results import save_simulation_results, save_results_as_csv
from src.components.antenna import AntennaConfig
from src.components.transmitter import Transmitter
from src.components.channel import ChannelModel
from src.components.receiver import Receiver
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.mimo import StreamManagement

# ---------------------------------------------------------------------------
# Core Simulation Loop
# ---------------------------------------------------------------------------

def run_simulation_loop(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for running a simulation loop.
    
    The loop type is determined by the config:
    - If `estimators` list is present: Compare Channel Estimators.
    - If `resource_managers` list is present: Compare Resource Managers.
    
    Args:
        config: Configuration dictionary containing simulation parameters.
        
    Returns:
        Dictionary containing all results.
    """
    print("=" * 70)
    print("  6G Smart Factory Simulation Loop")
    print("=" * 70)
    
    # Defaults
    output_dir = config.get("output_dir", "results/runs")
    ebno_db_range = np.array(config.get("ebno_db_range", [0, 5, 10, 15, 20]))
    batch_size = config.get("batch_size", 32)
    min_batches = int(config.get("total_batches", 10))
    max_mc_batches = max(
        min_batches,
        int(config.get("max_mc_batches", config.get("confidence_max_batches", 20000))),
    )
    system_config = config.get("system_config", {})
    target_block_errors = config.get("target_block_errors", 1000)
    target_block_errors = None if target_block_errors is None else int(target_block_errors)
    target_ber = config.get("target_ber")
    confidence_level = float(config.get("confidence_level", 0.95))
    min_total_bits = int(config.get("min_total_bits", 0))
    
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
    print(
        f"Batch Size: {batch_size}, Min Batches: {min_batches}, "
        f"Max Batches: {max_mc_batches}"
    )
    print(
        f"Stopping: target_block_errors={target_block_errors}, "
        f"target_ber={target_ber}, min_total_bits={min_total_bits}"
    )
    print("-" * 70)

    # Storage for aggregated results per method.
    aggregated_results = {
        item: {
            "ber": [],
            "ber_upper_confidence": [],
            "throughput": [],
            "latency": [],
            "bit_errors": [],
            "total_bits": [],
            "block_errors": [],
            "total_blocks": [],
            "num_batches": [],
        } for item in items_to_compare
    }
    method_runtime_sec = {item: 0.0 for item in items_to_compare}
    
    global_start = time.time()
    if mode == "estimator_comparison":
        print("\nInitializing estimator models...")
        models = {
            item_name: _build_model(mode, item_name, system_config, config, reused_components=None)
            for item_name in items_to_compare
        }

        print("Starting estimator Monte Carlo sweep...")
        for item_name in items_to_compare:
            print(f"Estimator: {item_name}")
            model = models[item_name]
            for ebno_val in ebno_db_range:
                metrics = _run_batches(
                    model,
                    batch_size,
                    min_batches,
                    max_mc_batches,
                    float(ebno_val),
                    target_block_errors=target_block_errors,
                    target_ber=target_ber,
                    confidence_level=confidence_level,
                    min_total_bits=min_total_bits,
                )
                aggregated_results[item_name]["ber"].append(metrics["ber"])
                aggregated_results[item_name]["ber_upper_confidence"].append(
                    _ber_upper_confidence_bound(
                        metrics["bit_errors"], metrics["total_bits"], confidence_level
                    )
                )
                aggregated_results[item_name]["throughput"].append(metrics["throughput"])
                aggregated_results[item_name]["latency"].append(metrics["latency"])
                aggregated_results[item_name]["bit_errors"].append(metrics["bit_errors"])
                aggregated_results[item_name]["total_bits"].append(metrics["total_bits"])
                aggregated_results[item_name]["block_errors"].append(metrics["block_errors"])
                aggregated_results[item_name]["total_blocks"].append(metrics["total_blocks"])
                aggregated_results[item_name]["num_batches"].append(metrics["num_batches"])
                method_runtime_sec[item_name] += metrics["runtime_sec"]
                print(
                    f"  Eb/No={float(ebno_val):4.1f} dB"
                    f" | BER={metrics['ber']:.3e}"
                    f" | BERub={aggregated_results[item_name]['ber_upper_confidence'][-1]:.3e}"
                    f" | Latency={metrics['latency'] * 1000:.3f} ms"
                    f" | Batches={metrics['num_batches']}"
                    f" | Stop={metrics['stop_reason']}"
                )
    else:
        print("\nInitializing shared simulation components...")
        with tf.device("/CPU:0"):
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
                pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
            )
            sm = StreamManagement(rx_tx_association, num_streams_per_tx)
            antenna_config = AntennaConfig(system_config)
            transmitter = Transmitter(system_config, rg)
            channel = ChannelModel(system_config, antenna_config, rg)

            from src.components.estimators import LMMSEChannelEstimator
            estimator = LMMSEChannelEstimator(rg)
            default_receiver = Receiver(
                system_config,
                rg,
                sm,
                transmitter._encoder,
                perfect_csi=False,
                channel_estimator=estimator,
            )

        shared_components = {
            "rg": rg,
            "sm": sm,
            "antenna_config": antenna_config,
            "transmitter": transmitter,
            "channel": channel,
            "receiver": default_receiver,
            "estimator_type": "lmmse",
            "perfect_csi": False,
        }
        intermediate_results = {
            item: [
                {
                    "errors": 0,
                    "bits": 0,
                    "block_errors": 0,
                    "blocks": 0,
                    "throughput": 0.0,
                    "latency": 0.0,
                    "num_batches": 0,
                    "done": False,
                    "stop_reason": None,
                }
                for _ in ebno_db_range
            ] for item in items_to_compare
        }
        models = {
            item_name: _build_model(mode, item_name, system_config, config, reused_components=shared_components)
            for item_name in items_to_compare
        }

        print("Starting resource-manager Monte Carlo sweep...")
        print("Optimization: Reusing channel realizations across resource managers.")
        total_points = len(items_to_compare) * len(ebno_db_range)
        for b in range(max_mc_batches):
            remaining = sum(
                1
                for item_name in items_to_compare
                for stats in intermediate_results[item_name]
                if not stats["done"]
            )
            if remaining == 0:
                break
            batch_start = time.time()
            print(
                f"Batch {b+1}/{max_mc_batches} "
                f"(active points {remaining}/{total_points}) ... ",
                end="",
                flush=True,
            )

            with tf.device("/CPU:0"):
                channel.set_topology(batch_size)
            print("[Channel Gen] ", end="", flush=True)

            for e_idx, ebno_val in enumerate(ebno_db_range):
                for item_name in items_to_compare:
                    stats = intermediate_results[item_name][e_idx]
                    if stats["done"]:
                        continue
                    t_method_start = time.perf_counter()
                    metrics = models[item_name].run_batch(
                        batch_size,
                        float(ebno_val),
                        include_details=False,
                        regenerate_topology=False,
                    )
                    method_runtime_sec[item_name] += (time.perf_counter() - t_method_start)
                    batch_stats = _extract_error_stats(metrics["bits"], metrics["bits_hat"])
                    stats["errors"] += batch_stats["bit_errors"]
                    stats["bits"] += batch_stats["total_bits"]
                    stats["block_errors"] += batch_stats["block_errors"]
                    stats["blocks"] += batch_stats["total_blocks"]
                    stats["throughput"] += max(0, batch_stats["total_bits"] - batch_stats["bit_errors"])
                    stats["latency"] += _estimate_air_interface_latency(system_config)
                    stats["num_batches"] += 1
                    stop_reason = _mc_stop_reason(
                        num_batches=stats["num_batches"],
                        total_bits=stats["bits"],
                        total_block_errors=stats["block_errors"],
                        target_block_errors=target_block_errors,
                        total_bit_errors=stats["errors"],
                        target_ber=target_ber,
                        confidence_level=confidence_level,
                        min_batches=min_batches,
                        min_total_bits=min_total_bits,
                    )
                    if stop_reason is not None:
                        stats["done"] = True
                        stats["stop_reason"] = stop_reason
            print(f"Done ({time.time() - batch_start:.2f}s)")

        for item_name in items_to_compare:
            for stats in intermediate_results[item_name]:
                if stats["num_batches"] > 0 and stats["stop_reason"] is None:
                    stats["stop_reason"] = "max_mc_batches"

        print("\nAggregating final results...")
        for item_name in items_to_compare:
            for e_idx, _ in enumerate(ebno_db_range):
                stats = intermediate_results[item_name][e_idx]
                avg_ber = stats["errors"] / stats["bits"] if stats["bits"] > 0 else 0.0
                aggregated_results[item_name]["ber"].append(avg_ber)
                aggregated_results[item_name]["ber_upper_confidence"].append(
                    _ber_upper_confidence_bound(stats["errors"], stats["bits"], confidence_level)
                )
                num_batches = max(1, stats["num_batches"])
                aggregated_results[item_name]["throughput"].append(stats["throughput"] / num_batches)
                aggregated_results[item_name]["latency"].append(stats["latency"] / num_batches)
                aggregated_results[item_name]["bit_errors"].append(int(stats["errors"]))
                aggregated_results[item_name]["total_bits"].append(int(stats["bits"]))
                aggregated_results[item_name]["block_errors"].append(int(stats["block_errors"]))
                aggregated_results[item_name]["total_blocks"].append(int(stats["blocks"]))
                aggregated_results[item_name]["num_batches"].append(int(stats["num_batches"]))

    total_time = time.time() - global_start
    print(f"Total Simulation Time: {total_time:.2f}s")

    # --- Save Results ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_label = "resource_managers" if mode == "resource_manager_comparison" else "estimators"
    run_dir = os.path.join(output_dir, f"{timestamp}_{run_label}")
    os.makedirs(run_dir, exist_ok=True)

    if mode == "resource_manager_comparison":
        filename_base = "simulation_results_resource_managers"
    else:
        estimator_type = system_config.get("estimator_type", "unknown_est")
        filename_base = f"simulation_results_{estimator_type}"

    # 1. Save as JSON (Raw data)
    full_results = {
        "config": config,
        "results": aggregated_results,
        "ebno_db_range": ebno_db_range.tolist(),
        "timestamp": timestamp,
        "run_label": run_label,
        "run_dir": run_dir,
        "mode": mode,
        "confidence_level": confidence_level,
        "method_runtime_sec": method_runtime_sec,
    }
    save_simulation_results(full_results, run_dir, filename=f"{filename_base}.json")
    
    # 2. Save as CSV (Tabular data for easy reading)
    save_results_as_csv(full_results, run_dir, filename=f"{filename_base}.csv")
    
    # --- Generate Plots ---
    _plot_simulation_results(full_results, run_dir, mode)
        
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
        estimator_kwargs_config = global_config.get("estimator_kwargs", {})
        estimator_kwargs = {}
        if isinstance(estimator_kwargs_config, dict):
            direct = estimator_kwargs_config.get(item_name)
            if isinstance(direct, dict):
                estimator_kwargs = direct
            elif any(isinstance(v, dict) for v in estimator_kwargs_config.values()):
                estimator_kwargs = {}
            else:
                estimator_kwargs = estimator_kwargs_config
        
        return Model(
            config=base_config,
            estimator_type=estimator_type,
            estimator_kwargs=estimator_kwargs,
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


def _run_batches(
    model: Model,
    batch_size: int,
    min_batches: int,
    max_mc_batches: int,
    ebno_db: float,
    target_block_errors: int | None = 1000,
    target_ber: float | None = None,
    confidence_level: float = 0.95,
    min_total_bits: int = 0,
) -> Dict[str, float]:
    """Runs the simulation batches and calculates average metrics."""
    total_errors = 0
    total_bits = 0
    total_block_errors = 0
    total_blocks = 0
    total_throughput = 0.0 # bits successfully transferred
    latency_accum = 0.0
    num_batches_run = 0
    stop_reason = "max_mc_batches"
    runtime_start = time.perf_counter()

    for _ in range(max_mc_batches):
        res = model.run_batch(batch_size, ebno_db, include_details=False)
        num_batches_run += 1
        
        batch_stats = _extract_error_stats(res["bits"], res["bits_hat"])
        total_errors += batch_stats["bit_errors"]
        total_bits += batch_stats["total_bits"]
        total_block_errors += batch_stats["block_errors"]
        total_blocks += batch_stats["total_blocks"]
        
        # Throughput (successful bits)
        total_throughput += max(0, batch_stats["total_bits"] - batch_stats["bit_errors"])
        
        # Latency
        latency_accum += _estimate_air_interface_latency(model.get_config())

        candidate_stop_reason = _mc_stop_reason(
            num_batches=num_batches_run,
            total_bits=total_bits,
            total_block_errors=total_block_errors,
            target_block_errors=target_block_errors,
            total_bit_errors=total_errors,
            target_ber=target_ber,
            confidence_level=confidence_level,
            min_batches=min_batches,
            min_total_bits=min_total_bits,
        )
        if candidate_stop_reason is not None:
            stop_reason = candidate_stop_reason
            break

    avg_ber = total_errors / total_bits if total_bits > 0 else 0.0
    avg_throughput = total_throughput / num_batches_run if num_batches_run > 0 else 0.0
    avg_latency = latency_accum / num_batches_run if num_batches_run > 0 else 0.0
    
    return {
        "ber": avg_ber,
        "throughput": avg_throughput,
        "latency": avg_latency,
        "bit_errors": int(total_errors),
        "total_bits": int(total_bits),
        "block_errors": int(total_block_errors),
        "total_blocks": int(total_blocks),
        "runtime_sec": time.perf_counter() - runtime_start,
        "num_batches": num_batches_run,
        "stop_reason": stop_reason,
    }


def _extract_error_stats(bits: np.ndarray, bits_hat: np.ndarray) -> Dict[str, int]:
    """Return bit and block error counters for one Monte Carlo batch."""
    diff = np.not_equal(bits, bits_hat)
    block_error_mask = np.any(diff, axis=-1)
    return {
        "bit_errors": int(diff.sum()),
        "total_bits": int(bits.size),
        "block_errors": int(block_error_mask.sum()),
        "total_blocks": int(block_error_mask.size),
    }


def _estimate_air_interface_latency(system_config: dict) -> float:
    """Compute slot latency from OFDM parameters without a full detailed probe."""
    subcarrier_spacing = float(system_config.get("subcarrier_spacing", 30e3))
    fft_size = float(system_config.get("fft_size", 512))
    cyclic_prefix_length = float(system_config.get("cyclic_prefix_length", 20))
    num_ofdm_symbols = float(system_config.get("num_ofdm_symbols", 14))
    symbol_duration = 1.0 / subcarrier_spacing
    cyclic_prefix_ratio = cyclic_prefix_length / max(fft_size, 1.0)
    return symbol_duration * (1.0 + cyclic_prefix_ratio) * num_ofdm_symbols


def _mc_stop_reason(
    num_batches: int,
    total_bits: int,
    total_block_errors: int,
    target_block_errors: int | None,
    total_bit_errors: int,
    target_ber: float | None,
    confidence_level: float,
    min_batches: int,
    min_total_bits: int,
) -> str | None:
    """Return the active Monte Carlo stop reason, or None to continue."""
    if num_batches < min_batches or total_bits < min_total_bits:
        return None
    if target_block_errors is None and target_ber is None:
        return "min_evidence"
    if target_block_errors is not None and total_block_errors >= target_block_errors:
        return "target_block_errors"
    if target_ber is not None:
        ber_upper = _ber_upper_confidence_bound(total_bit_errors, total_bits, confidence_level)
        if ber_upper <= target_ber:
            return "target_ber"
    return None


def _zero_error_upper_bound(total_bits: int, confidence_level: float) -> float:
    """One-sided upper BER bound for zero observed errors."""
    alpha = max(1e-12, 1.0 - confidence_level)
    return -math.log(alpha) / max(total_bits, 1)


def _ber_upper_confidence_bound(bit_errors: int, total_bits: int, confidence_level: float) -> float:
    """One-sided BER upper confidence bound using Wilson score interval."""
    if total_bits <= 0:
        return float("nan")
    if bit_errors <= 0:
        return _zero_error_upper_bound(total_bits, confidence_level)

    p_hat = bit_errors / total_bits
    z = NormalDist().inv_cdf(max(1e-12, min(1 - 1e-12, confidence_level)))
    denom = 1.0 + (z * z / total_bits)
    center = p_hat + (z * z / (2.0 * total_bits))
    radius = z * math.sqrt((p_hat * (1.0 - p_hat) / total_bits) + ((z * z) / (4.0 * total_bits * total_bits)))
    return min(1.0, (center + radius) / denom)


def _plot_simulation_results(full_results: dict, output_dir: str, mode: str):
    """Generate method-comparison plots for BER, latency, throughput, and tradeoff."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = full_results["results"]
    ebno_range = full_results["ebno_db_range"]

    if mode == "estimator_comparison":
        ber_name = "estimator_ber_comparison.png"
        ber_bound_name = "estimator_ber_confidence_bound.png"
        lat_name = "estimator_latency_comparison.png"
        throughput_name = "estimator_throughput_comparison.png"
        tradeoff_name = "estimator_ber_latency_tradeoff.png"
        runtime_name = "estimator_runtime_bar.png"
        label_title = "Estimator"
    else:
        ber_name = "resource_manager_ber_comparison.png"
        ber_bound_name = "resource_manager_ber_confidence_bound.png"
        lat_name = "resource_manager_latency_comparison.png"
        throughput_name = "resource_manager_throughput_comparison.png"
        tradeoff_name = "resource_manager_ber_latency_tradeoff.png"
        runtime_name = "resource_manager_runtime_bar.png"
        label_title = "Resource Manager"

    markers = ["o", "s", "D", "^", "v", "x", "*", "+"]

    fig_ber, ax_ber = plt.subplots(1, 1, figsize=(9, 6))
    fig_ber_bound, ax_ber_bound = plt.subplots(1, 1, figsize=(9, 6))
    fig_lat, ax_lat = plt.subplots(1, 1, figsize=(9, 6))
    fig_thr, ax_thr = plt.subplots(1, 1, figsize=(9, 6))
    fig_tradeoff, ax_tradeoff = plt.subplots(1, 1, figsize=(9, 6))
    for idx, (name, metrics) in enumerate(results.items()):
        marker = markers[idx % len(markers)]
        ber = [max(v, 1e-12) for v in metrics["ber"]]
        ber_upper = [max(v, 1e-12) for v in metrics["ber_upper_confidence"]]
        latency_ms = [l * 1000 for l in metrics["latency"]]
        throughput = metrics["throughput"]
        ax_ber.semilogy(ebno_range, ber, marker=marker, label=name, linewidth=2)
        ax_ber_bound.semilogy(ebno_range, ber_upper, marker=marker, label=name, linewidth=2)
        ax_lat.plot(ebno_range, latency_ms, marker=marker, label=name, linewidth=2)
        ax_thr.plot(ebno_range, throughput, marker=marker, label=name, linewidth=2)
        ax_tradeoff.scatter(latency_ms, ber, marker=marker, label=name, s=60)

    ax_ber.set_xlabel("Eb/No (dB)")
    ax_ber.set_ylabel("BER")
    ax_ber.set_title(f"{label_title} BER Comparison")
    ax_ber.grid(True, which="both", alpha=0.3)
    ax_ber.legend()

    ax_ber_bound.set_xlabel("Eb/No (dB)")
    ax_ber_bound.set_ylabel("BER Upper Bound")
    ax_ber_bound.set_title(f"{label_title} BER Confidence Bound")
    ax_ber_bound.grid(True, which="both", alpha=0.3)
    ax_ber_bound.legend()

    ax_lat.set_xlabel("Eb/No (dB)")
    ax_lat.set_ylabel("Latency (ms)")
    ax_lat.set_title(f"{label_title} Latency Comparison")
    ax_lat.grid(True, alpha=0.3)
    ax_lat.legend()

    ax_thr.set_xlabel("Eb/No (dB)")
    ax_thr.set_ylabel("Throughput (bits/batch)")
    ax_thr.set_title(f"{label_title} Throughput Comparison")
    ax_thr.grid(True, alpha=0.3)
    ax_thr.legend()

    ax_tradeoff.set_xlabel("Latency (ms)")
    ax_tradeoff.set_ylabel("BER")
    ax_tradeoff.set_yscale("log")
    ax_tradeoff.set_title(f"{label_title} BER-Latency Tradeoff")
    ax_tradeoff.grid(True, which="both", alpha=0.3)
    ax_tradeoff.legend()

    fig_ber.tight_layout()
    fig_ber_bound.tight_layout()
    fig_lat.tight_layout()
    fig_thr.tight_layout()
    fig_tradeoff.tight_layout()

    ber_path = os.path.join(output_dir, ber_name)
    ber_bound_path = os.path.join(output_dir, ber_bound_name)
    lat_path = os.path.join(output_dir, lat_name)
    throughput_path = os.path.join(output_dir, throughput_name)
    tradeoff_path = os.path.join(output_dir, tradeoff_name)
    runtime_path = os.path.join(output_dir, runtime_name)
    fig_ber.savefig(ber_path, dpi=300)
    fig_ber_bound.savefig(ber_bound_path, dpi=300)
    fig_lat.savefig(lat_path, dpi=300)
    fig_thr.savefig(throughput_path, dpi=300)
    fig_tradeoff.savefig(tradeoff_path, dpi=300)

    runtime_map = full_results.get("method_runtime_sec", {})
    if runtime_map:
        names = list(runtime_map.keys())
        values = [runtime_map[n] for n in names]
        fig_runtime, ax_runtime = plt.subplots(1, 1, figsize=(9, 6))
        ax_runtime.bar(names, values)
        ax_runtime.set_ylabel("Runtime (sec)")
        ax_runtime.set_title(f"{label_title} Runtime by Method")
        ax_runtime.grid(True, axis="y", alpha=0.3)
        fig_runtime.tight_layout()
        fig_runtime.savefig(runtime_path, dpi=300)
        plt.close(fig_runtime)

    plt.close(fig_ber)
    plt.close(fig_ber_bound)
    plt.close(fig_lat)
    plt.close(fig_thr)
    plt.close(fig_tradeoff)
    print(f"[OK] Plot saved to {ber_path}")
    print(f"[OK] Plot saved to {ber_bound_path}")
    print(f"[OK] Plot saved to {lat_path}")
    print(f"[OK] Plot saved to {throughput_path}")
    print(f"[OK] Plot saved to {tradeoff_path}")
    if runtime_map:
        print(f"[OK] Plot saved to {runtime_path}")


def run_simulation_campaign(config: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible wrapper."""
    return run_simulation_loop(config)
