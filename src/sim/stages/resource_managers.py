from __future__ import annotations

import time
from typing import Any

from src.models.model import Model
from src.models.resource_manager import create_resource_manager
from src.sim.config import Factory6GConfig

from .common import (
    append_point_metrics,
    initialize_stage_metrics,
    mc_stop_reason,
    resolve_kwargs,
    extract_error_stats,
    transmitted_ut_mask,
)


def _new_rm_point_state() -> dict[str, Any]:
    return {
        "errors": 0,
        "bits": 0,
        "block_errors": 0,
        "blocks": 0,
        "throughput": 0.0,
        "latency_sec": 0.0,
        "energy_joules": 0.0,
        "runtime_sec": 0.0,
        "num_batches": 0,
        "done": False,
    }


def run_resource_manager_stage(config: Factory6GConfig) -> dict[str, Any]:
    system_config = config.system_runtime_config
    methods = config.resource_managers.enabled
    ebno_db_range = [float(v) for v in config.monte_carlo.ebno_db_range]

    num_ut = int(system_config.get("num_ut", 8))
    if config.resource_managers.num_active_users > num_ut:
        raise ValueError("resource_managers.num_active_users must be <= system.num_ut.")

    aggregated = initialize_stage_metrics(methods)
    runtime_totals_sec = {method: 0.0 for method in methods}

    adaptive_kwargs = resolve_kwargs(config.estimators.kwargs, "adaptive")
    shared_model = Model(
        config=system_config,
        estimator_type="adaptive",
        estimator_kwargs=adaptive_kwargs,
        perfect_csi=False,
    )
    managers = {
        method: create_resource_manager(
            method,
            num_ut=num_ut,
            num_active=config.resource_managers.num_active_users,
            cnn_model_path=config.resource_managers.cnn_model_path,
            drl_model_path=config.resource_managers.drl_model_path,
            manager_kwargs=resolve_kwargs(config.resource_managers.kwargs, method),
        )
        for method in methods
    }
    point_state = {
        method: [_new_rm_point_state() for _ in ebno_db_range]
        for method in methods
    }

    total_points = len(methods) * len(ebno_db_range)
    for batch_index in range(config.monte_carlo.max_batches):
        remaining = sum(
            1
            for method in methods
            for state in point_state[method]
            if not state["done"]
        )
        if remaining == 0:
            break

        batch_start = time.time()
        print(
            f"Batch {batch_index + 1}/{config.monte_carlo.max_batches} "
            f"(active points {remaining}/{total_points}) ... ",
            end="",
            flush=True,
        )

        for ebno_index, ebno_db in enumerate(ebno_db_range):
            if all(point_state[method][ebno_index]["done"] for method in methods):
                continue
            context = shared_model.prepare_batch_context(
                batch_size=config.monte_carlo.batch_size,
                ebno_db=ebno_db,
                include_feedback=True,
            )
            for method, manager in managers.items():
                stats = point_state[method][ebno_index]
                if stats["done"]:
                    continue

                method_start = time.perf_counter()
                feedback = context.feedback if manager.needs_channel_feedback else None
                directives = manager.get_runtime_directives(system_config, ebno_db, feedback=feedback)
                result = shared_model.run_batch(context, directives=directives, include_details=True)
                elapsed = time.perf_counter() - method_start
                runtime_totals_sec[method] += elapsed
                stats["runtime_sec"] += elapsed

                active_ut_mask = transmitted_ut_mask(directives, num_ut=num_ut)
                batch_stats = extract_error_stats(
                    result["bits"],
                    result["bits_hat"],
                    ut_mask=active_ut_mask,
                )
                stats["errors"] += batch_stats["bit_errors"]
                stats["bits"] += batch_stats["total_bits"]
                stats["block_errors"] += batch_stats["block_errors"]
                stats["blocks"] += batch_stats["total_blocks"]
                stats["throughput"] += max(0.0, batch_stats["total_bits"] - batch_stats["bit_errors"])
                stats["latency_sec"] += float(result.get("runtime_latency_sec", 0.0))
                stats["energy_joules"] += float(result.get("energy_joules", 0.0))
                stats["num_batches"] += 1

                stop_reason = mc_stop_reason(
                    num_batches=stats["num_batches"],
                    total_bits=stats["bits"],
                    total_block_errors=stats["block_errors"],
                    target_block_errors=config.monte_carlo.target_block_errors,
                    total_bit_errors=stats["errors"],
                    target_ber=config.monte_carlo.target_ber,
                    confidence_level=config.monte_carlo.confidence_level,
                    min_batches=config.monte_carlo.min_batches,
                    min_total_bits=config.monte_carlo.min_total_bits,
                )
                if stop_reason is not None:
                    stats["done"] = True
        print(f"Done ({time.time() - batch_start:.2f}s)")

    for method in methods:
        for stats in point_state[method]:
            safe_batches = max(1, int(stats["num_batches"]))
            avg_latency_sec = float(stats["latency_sec"]) / safe_batches
            avg_energy = float(stats["energy_joules"]) / safe_batches
            avg_power = avg_energy / max(avg_latency_sec, 1e-12)
            point = {
                "ber": (float(stats["errors"]) / float(stats["bits"])) if stats["bits"] > 0 else 0.0,
                "latency_ms": avg_latency_sec * 1000.0,
                "throughput_bits_per_batch": float(stats["throughput"]) / safe_batches,
                "energy_joules_per_batch": avg_energy,
                "avg_power_w": avg_power,
                "runtime_sec": float(stats["runtime_sec"]),
                "bit_errors": float(stats["errors"]),
                "total_bits": float(stats["bits"]),
                "block_errors": float(stats["block_errors"]),
                "total_blocks": float(stats["blocks"]),
                "num_batches": float(stats["num_batches"]),
            }
            append_point_metrics(
                aggregated[method],
                confidence_level=config.monte_carlo.confidence_level,
                point=point,
            )

    return {
        "stage": "resource_managers",
        "ebno_db_range": ebno_db_range,
        "methods": aggregated,
        "runtime_totals_sec": runtime_totals_sec,
    }
