from __future__ import annotations

from typing import Any

from src.models.model import Model
from src.sim.config import Factory6GConfig

from .common import append_point_metrics, initialize_stage_metrics, resolve_kwargs, run_monte_carlo_point


def run_estimator_stage(config: Factory6GConfig) -> dict[str, Any]:
    system_config = config.system_runtime_config
    methods = config.estimators.enabled
    ebno_db_range = [float(v) for v in config.monte_carlo.ebno_db_range]

    aggregated = initialize_stage_metrics(methods)
    runtime_totals_sec = {method: 0.0 for method in methods}
    models = {
        method: Model(
            config=system_config,
            estimator_type=method,
            estimator_kwargs=resolve_kwargs(config.estimators.kwargs, method),
            perfect_csi=(method.lower() == "perfect"),
        )
        for method in methods
    }

    for method in methods:
        model = models[method]
        print(f"Estimator: {method}")
        for ebno_db in ebno_db_range:
            point = run_monte_carlo_point(
                model=model,
                batch_size=config.monte_carlo.batch_size,
                ebno_db=ebno_db,
                min_batches=config.monte_carlo.min_batches,
                max_mc_batches=config.monte_carlo.max_batches,
                target_block_errors=config.monte_carlo.target_block_errors,
                target_ber=config.monte_carlo.target_ber,
                confidence_level=config.monte_carlo.confidence_level,
                min_total_bits=config.monte_carlo.min_total_bits,
                include_feedback=False,
            )
            runtime_totals_sec[method] += float(point["runtime_sec"])
            append_point_metrics(
                aggregated[method],
                confidence_level=config.monte_carlo.confidence_level,
                point=point,
            )
            print(
                f"  Eb/No={ebno_db:4.1f} dB"
                f" | BER={point['ber']:.3e}"
                f" | BERub={aggregated[method]['ber_upper_confidence'][-1]:.3e}"
                f" | Latency={point['latency_ms']:.3f} ms"
                f" | Throughput={point['throughput_bits_per_batch']:.2f}"
                f" | Power={point['avg_power_w']:.4f} W"
            )

    return {
        "stage": "estimators",
        "ebno_db_range": ebno_db_range,
        "methods": aggregated,
        "runtime_totals_sec": runtime_totals_sec,
    }
