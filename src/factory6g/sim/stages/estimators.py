from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from factory6g.models.model import Model
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
)


def run_estimator_stage(
    config: Factory6GConfig,
    *,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    system_config = config.system_runtime_config
    methods = config.estimators.enabled
    ebno_db_range = [float(v) for v in config.monte_carlo.ebno_db_range]
    num_ut = int(system_config.get("num_ut", 8))
    base_seed = int(config.simulation.seed)
    max_harq_rounds = int(system_config.get("harq_max_rounds", 1))

    paired_reference = config.estimators.paired_reference or (methods[0] if methods else "")

    aggregated = initialize_stage_metrics(methods)
    runtime_totals_sec = {method: 0.0 for method in methods}
    shared_context_model = Model(
        config=system_config,
        estimator_type="ls",
        estimator_kwargs=resolve_kwargs(config.estimators.kwargs, "ls"),
        perfect_csi=False,
    )
    models = {
        method: Model(
            config=system_config,
            estimator_type=method,
            estimator_kwargs=resolve_kwargs(config.estimators.kwargs, method),
            perfect_csi=(method.lower() == "perfect"),
        )
        for method in methods
    }
    slot_duration_sec = slot_duration_seconds(system_config)
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
            start_batch_index = int(ckpt["batch_index"]) + 1
            print(
                f"[checkpoint] Resuming estimator from batch "
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
            f"Estimator batch {batch_index + 1}/{config.monte_carlo.max_batches} "
            f"(active points {remaining}/{total_points}) "
            f"[{fmt_elapsed(time.perf_counter() - stage_start)} elapsed] ... ",
            end="",
            flush=True,
        )

        for ebno_index, ebno_db in enumerate(ebno_db_range):
            if all(point_state[method][ebno_index].done for method in methods):
                continue
            print(f"  snr={ebno_db:+.1f}dB ", end="", flush=True)
            # Seed derived from the point identity, not from one global stream, so
            # a point is reproducible on its own and unaffected by which other
            # methods are enabled or by where a resumed run picked up.
            seed_global_rngs(derive_seed(base_seed, "estimators", ebno_db, batch_index))
            context = shared_context_model.prepare_batch_context(
                batch_size=config.monte_carlo.batch_size,
                ebno_db=ebno_db,
                include_feedback=False,
            )
            for method, model in models.items():
                stats_acc = point_state[method][ebno_index]
                if stats_acc.done:
                    continue

                print(f"[{method}]", end="", flush=True)
                method_start = time.perf_counter()
                result = model.run_batch(
                    context, include_details=True, harq_max_rounds=max_harq_rounds
                )
                elapsed = time.perf_counter() - method_start
                runtime_totals_sec[method] += elapsed

                stats_acc.add_batch(
                    result, ut_mask=None, elapsed_sec=elapsed, num_ut=num_ut
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
                },
            )

    if checkpoint_dir is not None:
        delete_checkpoint(checkpoint_dir)

    finalized: dict[str, list[dict[str, Any]]] = {}
    for method in methods:
        print(f"Estimator: {method}")
        finalized[method] = []
        for ebno_db, stats_acc in zip(ebno_db_range, point_state[method]):
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
            print(
                f"  Eb/No={ebno_db:4.1f} dB"
                f" | BLER={point['bler']:.3e}"
                f" | BLERub={point['bler_upper_confidence']:.3e}"
                f" | BER={point['ber']:.3e}"
                f" | NMSE={point['nmse_db']:.2f} dB"
                f" | Status={point['point_status']}"
                f" | Stop={point['stop_reason']}"
            )

    return {
        "stage": "estimators",
        "ebno_db_range": ebno_db_range,
        "methods": aggregated,
        "runtime_totals_sec": runtime_totals_sec,
        "paired_reference": paired_reference,
        # The per-batch samples the paired analysis is built from. Kept in the
        # results so a different reference can be compared later without
        # re-running the sweep -- the checkpoint that also holds them is deleted
        # on completion, and these runs cost about an hour each.
        "paired_samples": {
            method: [
                {
                    "batch_block_errors": point.get("batch_block_errors", []),
                    "batch_blocks": point.get("batch_blocks", []),
                }
                for point in points
            ]
            for method, points in finalized.items()
        },
        "paired_comparisons": compare_methods_paired(
            finalized,
            reference=paired_reference,
            confidence_level=config.monte_carlo.confidence_level,
        ),
    }
