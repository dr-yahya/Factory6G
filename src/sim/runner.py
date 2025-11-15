"""
Core simulation runner for 6G simulations.

This module provides the main simulation execution logic.
"""

import time
import gc
import numpy as np
from typing import Optional, Any
from pathlib import Path

from src.sim.metrics import MetricsAccumulator
from src.sim.results import save_simulation_results
from src.sim.plotting import plot_simulation_results


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
) -> dict:
    """
    Run Monte Carlo simulation collecting extended diagnostics.
    
    Args:
        scenario: Channel scenario ("umi", "uma", "rma")
        perfect_csi_list: List of CSI conditions to test [True, False]
        ebno_db_range: Array of Eb/No values in dB
        batch_size: Batch size for simulation
        max_mc_iter: Maximum Monte Carlo iterations
        num_target_block_errors: Target number of block errors
        target_bler: Target BLER for early stopping
        config: SystemConfig instance
        save_results: Whether to save results to disk
        plot_results: Whether to generate plots
        output_dir: Output directory for results
        estimator_type: Channel estimator type
        estimator_weights: Path to neural estimator weights (if applicable)
        estimator_kwargs: Additional estimator arguments
        resource_manager: Resource manager instance
        resource_manager_config: Resource manager configuration dict
        profile_name: Profile name for results
        
    Returns:
        Dictionary containing simulation results
    """
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

