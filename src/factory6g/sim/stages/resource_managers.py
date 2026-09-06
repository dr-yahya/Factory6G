from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from factory6g.models.model import Model
from factory6g.models.resource_manager import create_resource_manager
from factory6g.sim.config import Factory6GConfig

from .checkpoint import delete_checkpoint, load_checkpoint, save_checkpoint
from .common import (
    PointAccumulator,
    append_point_metrics,
    compare_methods_paired,
    derive_seed,
    fmt_elapsed,
    initialize_stage_metrics,
    mc_stop_reason,
    resolve_kwargs,
    seed_global_rngs,
    slot_duration_seconds,
    transmitted_ut_mask,
)


def run_resource_manager_stage(
    config: Factory6GConfig,
    *,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    system_config = config.system_runtime_config
    methods = config.resource_managers.enabled
    ebno_db_range = [float(v) for v in config.monte_carlo.ebno_db_range]

    num_ut = int(system_config.get("num_ut", 8))
    if config.resource_managers.num_active_users > num_ut:
        raise ValueError("resource_managers.num_active_users must be <= system.num_ut.")

    base_seed = int(config.simulation.seed)
    max_harq_rounds = int(system_config.get("harq_max_rounds", 1))

    aggregated = initialize_stage_metrics(methods)
    runtime_totals_sec = {method: 0.0 for method in methods}

    adaptive_kwargs = resolve_kwargs(config.estimators.kwargs, "adaptive")
    shared_model = Model(
        config=system_config,
        estimator_type="adaptive",
        estimator_kwargs=adaptive_kwargs,
        perfect_csi=False,
    )
    slot_duration_sec = slot_duration_seconds(system_config)
    managers = {
        method: create_resource_manager(
            method,
            num_ut=num_ut,
            num_active=config.resource_managers.num_active_users,
            cnn_model_path=config.resource_managers.cnn_model_path,
            drl_model_path=config.resource_managers.drl_model_path,
            manager_kwargs=resolve_kwargs(config.resource_managers.kwargs, method),
            strict_policy_loading=config.resource_managers.strict_policy_loading,
            model_root=config.resource_managers.model_root,
        )
        for method in methods
    }
    # Record what actually ran, so a learned curve can never be silently backed
    # by the heuristic fallback in the published artifacts.
    manager_provenance = {
        method: manager.provenance()
        for method, manager in managers.items()
        if hasattr(manager, "provenance")
    }
    point_state = {
        method: [PointAccumulator(num_ut) for _ in ebno_db_range] for method in methods
    }

    start_batch_index = 0
    if checkpoint_dir is not None:
        ckpt = load_checkpoint(checkpoint_dir)
        if ckpt is not None:
            point_state = {
                method: [PointAccumulator.from_dict(entry) for entry in entries]
                for method, entries in ckpt["point_state"].items()
            }
            runtime_totals_sec = ckpt["runtime_totals_sec"]
            # Scheduler state must be restored too: a resumed run that restarts
            # every scheduler cold while its statistics keep accumulating is not
            # the same experiment as an uninterrupted one.
            for method, state in (ckpt.get("manager_state") or {}).items():
                if method in managers:
                    managers[method].load_state(state)
            start_batch_index = int(ckpt["batch_index"]) + 1
            print(
                f"[checkpoint] Resuming resource_managers from batch "
                f"{start_batch_index + 1}/{config.monte_carlo.max_batches}"
            )

    total_points = len(methods) * len(ebno_db_range)
    stage_start = time.perf_counter()
    for batch_index in range(start_batch_index, config.monte_carlo.max_batches):
        remaining = sum(
            1 for method in methods for state in point_state[method] if not state.done
        )
        if remaining == 0:
            break

        batch_start = time.time()
        print(
            f"Batch {batch_index + 1}/{config.monte_carlo.max_batches} "
            f"(active points {remaining}/{total_points}) "
            f"[{fmt_elapsed(time.perf_counter() - stage_start)} elapsed] ... ",
            end="",
            flush=True,
        )

        for ebno_index, ebno_db in enumerate(ebno_db_range):
            if all(point_state[method][ebno_index].done for method in methods):
                continue
            print(f"  snr={ebno_db:+.1f}dB ", end="", flush=True)
            seed_global_rngs(
                derive_seed(base_seed, "resource_managers", ebno_db, batch_index)
            )
            context = shared_model.prepare_batch_context(
                batch_size=config.monte_carlo.batch_size,
                ebno_db=ebno_db,
                include_feedback=True,
            )
            for method, manager in managers.items():
                stats_acc = point_state[method][ebno_index]
                if stats_acc.done:
                    continue

                print(f"[{method}]", end="", flush=True)
                method_start = time.perf_counter()
                feedback = context.feedback if manager.needs_channel_feedback else None
                directives = manager.get_runtime_directives(
                    system_config, ebno_db, feedback=feedback
                )
                result = shared_model.run_batch(
                    context,
                    directives=directives,
                    include_details=True,
                    harq_max_rounds=max_harq_rounds,
                )
                elapsed = time.perf_counter() - method_start
                runtime_totals_sec[method] += elapsed

                stats_acc.add_batch(
                    result,
                    ut_mask=transmitted_ut_mask(directives, num_ut=num_ut),
                    elapsed_sec=elapsed,
                    num_ut=num_ut,
                )

                stop_reason = mc_stop_reason(
                    num_batches=stats_acc.num_batches,
                    total_bits=stats_acc.bits,
                    total_block_errors=stats_acc.block_errors,
                    target_block_errors=config.monte_carlo.target_block_errors,
                    total_bit_errors=stats_acc.errors,
                    target_ber=config.monte_carlo.target_ber,
                    stop_policy=config.monte_carlo.stop_policy,
                    confidence_level=config.monte_carlo.confidence_level,
                    min_batches=config.monte_carlo.min_batches,
                    min_total_bits=config.monte_carlo.min_total_bits,
                )
                if stop_reason is not None:
                    stats_acc.done = True
                    stats_acc.stop_reason = stop_reason
        print(f"Done ({time.time() - batch_start:.2f}s)")
        if checkpoint_dir is not None:
            save_checkpoint(
                checkpoint_dir,
                {
                    "batch_index": batch_index,
                    "point_state": {
                        method: [entry.to_dict() for entry in entries]
                        for method, entries in point_state.items()
                    },
                    "runtime_totals_sec": runtime_totals_sec,
                    "manager_state": {
                        method: manager.export_state()
                        for method, manager in managers.items()
                    },
                },
            )

    if checkpoint_dir is not None:
        delete_checkpoint(checkpoint_dir)

    finalized: dict[str, list[dict[str, Any]]] = {}
    for method in methods:
        finalized[method] = []
        for stats_acc in point_state[method]:
            point = stats_acc.finalize(
                confidence_level=config.monte_carlo.confidence_level,
                slot_duration_sec=slot_duration_sec,
                max_harq_rounds=max_harq_rounds,
            )
            finalized[method].append(point)
            append_point_metrics(
                aggregated[method],
                confidence_level=config.monte_carlo.confidence_level,
                point=point,
            )

    # Paired comparison against the configured reference baseline.
    reference = config.resource_managers.paired_reference
    if reference not in finalized and methods:
        reference = methods[0]

    return {
        "stage": "resource_managers",
        "ebno_db_range": ebno_db_range,
        "methods": aggregated,
        "runtime_totals_sec": runtime_totals_sec,
        "manager_provenance": manager_provenance,
        "paired_reference": reference,
        "paired_comparisons": compare_methods_paired(
            finalized,
            reference=reference,
            confidence_level=config.monte_carlo.confidence_level,
        ),
    }
