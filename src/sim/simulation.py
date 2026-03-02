from __future__ import annotations

import math
import os
import time
from statistics import NormalDist
from typing import Any

import numpy as np

from src.models.model import Model
from src.models.resource_manager import create_resource_manager
from src.sim.config import Factory6GConfig
from src.sim.results import save_results_as_csv, save_simulation_results


def run_simulation_loop(config: Factory6GConfig) -> dict[str, dict[str, Any]]:
    print("=" * 70)
    print("  6G Smart Factory Simulation Loop")
    print("=" * 70)
    print(f"Targets: {config.simulation.targets}")
    print(f"Scenario: {config.system.scenario}")
    print(f"Eb/No Range: {config.monte_carlo.ebno_db_range}")
    print("-" * 70)

    results_by_target: dict[str, dict[str, Any]] = {}
    for target in config.simulation.targets:
        print(f"\n=== Running {target} ===")
        if target == "estimators":
            full_results = _run_estimator_comparison(config)
        elif target == "resource_managers":
            full_results = _run_resource_manager_comparison(config)
        else:
            raise ValueError(f"Unknown simulation target '{target}'.")
        results_by_target[target] = full_results
    return results_by_target


def _run_estimator_comparison(config: Factory6GConfig) -> dict[str, Any]:
    system_config = config.system_runtime_config
    items_to_compare = config.estimators.enabled
    ebno_db_range = np.array(config.monte_carlo.ebno_db_range, dtype=float)
    batch_size = config.monte_carlo.batch_size
    min_batches = config.monte_carlo.min_batches
    max_mc_batches = config.monte_carlo.max_batches
    target_block_errors = config.monte_carlo.target_block_errors
    target_ber = config.monte_carlo.target_ber
    confidence_level = config.monte_carlo.confidence_level
    min_total_bits = config.monte_carlo.min_total_bits

    aggregated_results = _initialize_aggregated_results(items_to_compare)
    method_runtime_sec = {item: 0.0 for item in items_to_compare}

    models = {
        name: Model(
            config=system_config,
            estimator_type=name,
            estimator_kwargs=_resolve_estimator_kwargs(config.estimators.kwargs, name),
            perfect_csi=(name.lower() == "perfect"),
        )
        for name in items_to_compare
    }

    global_start = time.time()
    for item_name in items_to_compare:
        print(f"Estimator: {item_name}")
        model = models[item_name]
        for ebno_val in ebno_db_range:
            metrics = _run_batches(
                model=model,
                batch_size=batch_size,
                min_batches=min_batches,
                max_mc_batches=max_mc_batches,
                ebno_db=float(ebno_val),
                target_block_errors=target_block_errors,
                target_ber=target_ber,
                confidence_level=confidence_level,
                min_total_bits=min_total_bits,
            )
            _append_metrics(aggregated_results[item_name], metrics, confidence_level)
            method_runtime_sec[item_name] += metrics["runtime_sec"]
            print(
                f"  Eb/No={float(ebno_val):4.1f} dB"
                f" | BER={metrics['ber']:.3e}"
                f" | BERub={aggregated_results[item_name]['ber_upper_confidence'][-1]:.3e}"
                f" | Latency={metrics['latency'] * 1000:.3f} ms"
                f" | Batches={metrics['num_batches']}"
                f" | Stop={metrics['stop_reason']}"
            )

    total_time = time.time() - global_start
    print(f"Total Simulation Time: {total_time:.2f}s")
    return _finalize_results(
        config=config,
        mode="estimator_comparison",
        run_label="estimators",
        filename_base="simulation_results_estimators",
        ebno_db_range=ebno_db_range,
        aggregated_results=aggregated_results,
        method_runtime_sec=method_runtime_sec,
    )


def _run_resource_manager_comparison(config: Factory6GConfig) -> dict[str, Any]:
    system_config = config.system_runtime_config
    items_to_compare = config.resource_managers.enabled
    ebno_db_range = np.array(config.monte_carlo.ebno_db_range, dtype=float)
    batch_size = config.monte_carlo.batch_size
    min_batches = config.monte_carlo.min_batches
    max_mc_batches = config.monte_carlo.max_batches
    target_block_errors = config.monte_carlo.target_block_errors
    target_ber = config.monte_carlo.target_ber
    confidence_level = config.monte_carlo.confidence_level
    min_total_bits = config.monte_carlo.min_total_bits

    num_ut = int(system_config.get("num_ut", 8))
    if config.resource_managers.num_active_users > num_ut:
        raise ValueError(
            "resource_managers.num_active_users must be <= system.num_ut."
        )

    aggregated_results = _initialize_aggregated_results(items_to_compare)
    method_runtime_sec = {item: 0.0 for item in items_to_compare}
    shared_model = Model(config=system_config, estimator_type="lmmse", perfect_csi=False)
    managers = {
        name: create_resource_manager(
            name,
            num_ut=num_ut,
            num_active=config.resource_managers.num_active_users,
            cnn_model_path=config.resource_managers.cnn_model_path,
        )
        for name in items_to_compare
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
        ]
        for item in items_to_compare
    }

    total_points = len(items_to_compare) * len(ebno_db_range)
    global_start = time.time()
    for batch_index in range(max_mc_batches):
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
            f"Batch {batch_index + 1}/{max_mc_batches} "
            f"(active points {remaining}/{total_points}) ... ",
            end="",
            flush=True,
        )

        for ebno_index, ebno_val in enumerate(ebno_db_range):
            if all(intermediate_results[name][ebno_index]["done"] for name in items_to_compare):
                continue
            context = shared_model.prepare_batch_context(
                batch_size=batch_size,
                ebno_db=float(ebno_val),
                include_feedback=True,
            )
            for item_name, manager in managers.items():
                stats = intermediate_results[item_name][ebno_index]
                if stats["done"]:
                    continue
                start = time.perf_counter()
                feedback = context.feedback if manager.needs_channel_feedback else None
                directives = manager.get_runtime_directives(system_config, float(ebno_val), feedback=feedback)
                metrics = shared_model.run_batch(context, directives=directives, include_details=False)
                method_runtime_sec[item_name] += time.perf_counter() - start

                batch_stats = _extract_error_stats(metrics["bits"], metrics["bits_hat"])
                stats["errors"] += batch_stats["bit_errors"]
                stats["bits"] += batch_stats["total_bits"]
                stats["block_errors"] += batch_stats["block_errors"]
                stats["blocks"] += batch_stats["total_blocks"]
                stats["throughput"] += max(0, batch_stats["total_bits"] - batch_stats["bit_errors"])
                stats["latency"] += _estimate_air_interface_latency(system_config)
                stats["num_batches"] += 1
                stats["stop_reason"] = _mc_stop_reason(
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
                if stats["stop_reason"] is not None:
                    stats["done"] = True

        print(f"Done ({time.time() - batch_start:.2f}s)")

    for item_name in items_to_compare:
        for stats in intermediate_results[item_name]:
            if stats["num_batches"] > 0 and stats["stop_reason"] is None:
                stats["stop_reason"] = "max_mc_batches"
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
    return _finalize_results(
        config=config,
        mode="resource_manager_comparison",
        run_label="resource_managers",
        filename_base="simulation_results_resource_managers",
        ebno_db_range=ebno_db_range,
        aggregated_results=aggregated_results,
        method_runtime_sec=method_runtime_sec,
    )


def _run_batches(
    model: Model,
    batch_size: int,
    min_batches: int,
    max_mc_batches: int,
    ebno_db: float,
    target_block_errors: int | None,
    target_ber: float | None,
    confidence_level: float,
    min_total_bits: int,
) -> dict[str, Any]:
    total_errors = 0
    total_bits = 0
    total_block_errors = 0
    total_blocks = 0
    total_throughput = 0.0
    latency_accum = 0.0
    num_batches_run = 0
    stop_reason = "max_mc_batches"
    runtime_start = time.perf_counter()

    for _ in range(max_mc_batches):
        context = model.prepare_batch_context(batch_size, ebno_db, include_feedback=False)
        res = model.run_batch(context, include_details=False)
        num_batches_run += 1

        batch_stats = _extract_error_stats(res["bits"], res["bits_hat"])
        total_errors += batch_stats["bit_errors"]
        total_bits += batch_stats["total_bits"]
        total_block_errors += batch_stats["block_errors"]
        total_blocks += batch_stats["total_blocks"]
        total_throughput += max(0, batch_stats["total_bits"] - batch_stats["bit_errors"])
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


def _initialize_aggregated_results(items_to_compare: list[str]) -> dict[str, dict[str, list[Any]]]:
    return {
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
        }
        for item in items_to_compare
    }


def _resolve_estimator_kwargs(
    estimator_kwargs_config: dict[str, dict[str, Any]],
    item_name: str,
) -> dict[str, Any]:
    direct = estimator_kwargs_config.get(item_name)
    if isinstance(direct, dict):
        return direct
    return {}


def _append_metrics(
    aggregate: dict[str, list[Any]],
    metrics: dict[str, Any],
    confidence_level: float,
) -> None:
    aggregate["ber"].append(metrics["ber"])
    aggregate["ber_upper_confidence"].append(
        _ber_upper_confidence_bound(metrics["bit_errors"], metrics["total_bits"], confidence_level)
    )
    aggregate["throughput"].append(metrics["throughput"])
    aggregate["latency"].append(metrics["latency"])
    aggregate["bit_errors"].append(metrics["bit_errors"])
    aggregate["total_bits"].append(metrics["total_bits"])
    aggregate["block_errors"].append(metrics["block_errors"])
    aggregate["total_blocks"].append(metrics["total_blocks"])
    aggregate["num_batches"].append(metrics["num_batches"])


def _extract_error_stats(bits: np.ndarray, bits_hat: np.ndarray) -> dict[str, int]:
    diff = np.not_equal(bits, bits_hat)
    block_error_mask = np.any(diff, axis=-1)
    return {
        "bit_errors": int(diff.sum()),
        "total_bits": int(bits.size),
        "block_errors": int(block_error_mask.sum()),
        "total_blocks": int(block_error_mask.size),
    }


def _estimate_air_interface_latency(system_config: dict[str, Any]) -> float:
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
    alpha = max(1e-12, 1.0 - confidence_level)
    return -math.log(alpha) / max(total_bits, 1)


def _ber_upper_confidence_bound(bit_errors: int, total_bits: int, confidence_level: float) -> float:
    if total_bits <= 0:
        return float("nan")
    if bit_errors <= 0:
        return _zero_error_upper_bound(total_bits, confidence_level)
    p_hat = bit_errors / total_bits
    z = NormalDist().inv_cdf(max(1e-12, min(1.0 - 1e-12, confidence_level)))
    denom = 1.0 + (z * z / total_bits)
    center = p_hat + (z * z / (2.0 * total_bits))
    radius = z * math.sqrt(
        (p_hat * (1.0 - p_hat) / total_bits) + ((z * z) / (4.0 * total_bits * total_bits))
    )
    return min(1.0, (center + radius) / denom)


def _finalize_results(
    *,
    config: Factory6GConfig,
    mode: str,
    run_label: str,
    filename_base: str,
    ebno_db_range: np.ndarray,
    aggregated_results: dict[str, dict[str, list[Any]]],
    method_runtime_sec: dict[str, float],
) -> dict[str, Any]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.simulation.output_dir, f"{timestamp}_{run_label}")
    os.makedirs(run_dir, exist_ok=True)
    full_results = {
        "config": config.to_dict(),
        "results": aggregated_results,
        "ebno_db_range": ebno_db_range.tolist(),
        "timestamp": timestamp,
        "run_label": run_label,
        "run_dir": run_dir,
        "mode": mode,
        "confidence_level": config.monte_carlo.confidence_level,
        "method_runtime_sec": method_runtime_sec,
    }
    save_simulation_results(full_results, run_dir, filename=f"{filename_base}.json")
    save_results_as_csv(full_results, run_dir, filename=f"{filename_base}.csv")
    if config.simulation.plot_results:
        _plot_simulation_results(full_results, run_dir, mode)
    return full_results


def _plot_simulation_results(full_results: dict[str, Any], output_dir: str, mode: str) -> None:
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
        latency_ms = [lat * 1000 for lat in metrics["latency"]]
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
        values = [runtime_map[name] for name in names]
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
