#!/usr/bin/env python3
from __future__ import annotations
"""
Main simulation script for 6G Smart Factory Physical Layer System.

This script runs simulations based on the configuration in config.json.
It supports two primary simulation types:
1. 'estimators': Standard BER/BLER simulation comparing channel estimators.
2. 'resource_managers': Benchmark comparing different resource management strategies.

Usage:
    python main.py
    (Configure parameters in config.json)
"""

import os
import re
import sys
import time
import json
import random
import logging
import warnings
from pathlib import Path
from typing import Any, Optional

# Add src to path before importing project modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.sim.env import configure_env, setup_gpu


class _StreamFilter:
    """Filter noisy runtime warnings from stdout/stderr."""

    _XLA_PATTERNS = [
        re.compile(r'.*Allocation.*exceeds.*free system memory.*'),
        re.compile(r'.*external/local_xla/xla/tsl/framework/cpu_allocator_impl.*'),
        re.compile(r'.*No supported GPU was found.*'),
        re.compile(r'.*Matplotlib created a temporary cache directory.*'),
        re.compile(r'.*Matplotlib is building the font cache; this may take a moment.*'),
        re.compile(r'.*\.matplotlib is not a writable directory.*'),
    ]

    def __init__(self, original_stream):
        self.original_stream = original_stream

    def write(self, message):
        if not any(p.search(message) for p in self._XLA_PATTERNS):
            self.original_stream.write(message)

    def flush(self):
        self.original_stream.flush()


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file and flatten for system use."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    def _strip_json_comments(raw_text: str) -> str:
        """Strip // and /* */ comments while preserving string literals."""
        result = []
        i = 0
        in_string = False
        in_line_comment = False
        in_block_comment = False
        escape = False
        length = len(raw_text)

        while i < length:
            ch = raw_text[i]
            nxt = raw_text[i + 1] if i + 1 < length else ""

            if in_line_comment:
                if ch == "\n":
                    in_line_comment = False
                    result.append(ch)
                i += 1
                continue

            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    i += 2
                else:
                    i += 1
                continue

            if in_string:
                result.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
                i += 1
                continue

            if ch == "\"":
                in_string = True
                result.append(ch)
                i += 1
                continue

            if ch == "/" and nxt == "/":
                in_line_comment = True
                i += 2
                continue

            if ch == "/" and nxt == "*":
                in_block_comment = True
                i += 2
                continue

            result.append(ch)
            i += 1

        return "".join(result)

    with open(config_path, 'r', encoding="utf-8") as f:
        config_data = json.loads(_strip_json_comments(f.read()))

    # Flatten configuration for Model consumption
    # Base is system_params
    flat_config = config_data.get("system_params", {}).copy()
    
    # Merge relevant scenario_params if not present (priority to system_params)
    scenario_params = config_data.get("scenario_params", {})
    for k, v in scenario_params.items():
        if k not in flat_config and k in ["scenario", "target_bler", "min_ut_velocity", "max_ut_velocity"]:
             flat_config[k] = v

    # Merge resource_manager_params placeholders
    rm_params = config_data.get("resource_manager_params", {})
    for k in ["active_ut_mask", "per_ut_power", "pilot_reuse_factor"]:
        if k in rm_params and rm_params[k] is not None:
            flat_config[k] = rm_params[k]

    # Merge transceiver parameters that affect the Sionna antenna arrays.
    transceiver_params = config_data.get("transceiver_params", {})
    for k in [
        "antenna_spacing",
        "tx_pattern",
        "tx_polarization",
        "rx_pattern",
        "rx_polarization",
    ]:
        if k in transceiver_params and transceiver_params[k] is not None:
            flat_config[k] = transceiver_params[k]
            
    # Apply defaults for optional fields if missing
    defaults = {
        "pilot_ofdm_symbol_indices": [2, 11],
        "active_ut_mask": [1] * flat_config.get("num_ut", 8),
        "per_ut_power": [1.0] * flat_config.get("num_ut", 8),
        "pilot_reuse_factor": 1,
        "target_bler": 1e-3
    }
    
    for k, v in defaults.items():
        if k not in flat_config or flat_config[k] is None:
            flat_config[k] = v
            
    # Inject the flattened config back into config_data for passing to runners
    config_data["system_config"] = flat_config
    return config_data


def main():
    """Main entry point"""
    # Load configuration
    try:
        config_data = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    sim_config = config_data.get("simulation", {})
    scenario_params = config_data.get("scenario_params", {})
    rm_params = config_data.get("resource_manager_params", {})
    system_config = config_data.get("system_config", {})
    
    # Environment Setup
    gpu_id = sim_config.get("gpu_id", 0)
    force_cpu = sim_config.get("force_cpu", False)
    log_level_str = sim_config.get("log_level", "INFO")
    seed = sim_config.get("seed", 42)

    # Configure environment BEFORE importing TensorFlow/Sionna
    configure_env(force_cpu=force_cpu, gpu_num=gpu_id)
    sys.stdout = _StreamFilter(sys.stdout)
    sys.stderr = _StreamFilter(sys.stderr)

    # Import TensorFlow-dependent modules only after runtime env is configured
    import numpy as np
    import tensorflow as tf
    import sionna.phy
    from src.sim.simulation import run_simulation_loop

    # Logging Setup
    logging.basicConfig(level=getattr(logging, log_level_str.upper(), logging.INFO))
    
    # Configure logging levels
    _level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.getLogger().setLevel(_level)
    tf.get_logger().setLevel('ERROR' if _level > logging.DEBUG else 'INFO')
    
    # Suppress Python warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    logging.getLogger('tensorflow').setLevel(logging.ERROR if _level > logging.DEBUG else logging.INFO)
    logging.getLogger('absl').setLevel(logging.ERROR if _level > logging.DEBUG else logging.INFO)
    
    if not force_cpu:
        setup_gpu(gpu_id, force_cpu=force_cpu)
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    sionna.phy.config.seed = seed
    
    run_mode = sim_config.get("run_mode", "single")
    single_run_target = sim_config.get("single_run_target", sim_config.get("type", "estimators"))
    if isinstance(single_run_target, str):
        single_run_target = single_run_target.lower()
    if isinstance(run_mode, str):
        run_mode = run_mode.lower()

    if run_mode == "both":
        run_sequence = ["estimators", "resource_managers"]
    elif run_mode == "single":
        run_sequence = [single_run_target]
    else:
        print("Unknown simulation.run_mode. Supported: 'single', 'both'")
        return

    print(f"Loaded configuration for scenario: {system_config.get('scenario')}")
    print(f"Starting execution with run mode: {run_mode}")
    print(f"Run sequence: {run_sequence}")

    # Construct the unified configuration for run_simulation_loop
    loop_config = {
        "system_config": system_config,
        "output_dir": sim_config.get("output_dir", "results"),
        "batch_size": scenario_params.get("batch_size", 32),
        "total_batches": scenario_params.get("total_batches", 10),
        "max_mc_batches": scenario_params.get(
            "max_mc_batches",
            scenario_params.get("confidence_max_batches", 20000),
        ),
        "plot_results": sim_config.get("plot_results", True),
        "target_block_errors": scenario_params.get("target_block_errors", 1000),
        "target_ber": scenario_params.get("target_ber"),
        "confidence_level": scenario_params.get("confidence_level", 0.95),
        "confidence_max_batches": scenario_params.get("confidence_max_batches", 20000),
        "min_total_bits": scenario_params.get("min_total_bits", 0),
    }

    # Eb/No Range
    ebno_min = scenario_params.get("ebno_min", 0.0)
    ebno_max = scenario_params.get("ebno_max", 20.0)
    ebno_step = scenario_params.get("ebno_step", 5.0)
    loop_config["ebno_db_range"] = np.arange(ebno_min, ebno_max + ebno_step, ebno_step).tolist()

    for sim_type in run_sequence:
        active_loop_config = dict(loop_config)

        if sim_type == "estimators":
            estimators = scenario_params.get("estimators", ["ls", "dft", "lmmse", "pso", "perfect"])
            active_loop_config["estimators"] = estimators
            active_loop_config["estimator_kwargs"] = scenario_params.get("estimator_kwargs", {})
            print("\n=== Running estimator comparison ===")
            run_simulation_loop(active_loop_config)

        elif sim_type == "resource_managers":
            managers = rm_params.get("resource_managers", ["static", "round_robin", "max_throughput", "pf", "cnn"])
            active_loop_config["resource_managers"] = managers
            active_loop_config["num_active_users"] = rm_params.get("num_active_users", 2)
            active_loop_config["cnn_model_path"] = rm_params.get("cnn_model_path", "models/cnn_resource_manager.h5")
            print("\n=== Running resource manager comparison ===")
            run_simulation_loop(active_loop_config)

        else:
            print(f"Unknown simulation target: {sim_type}. Supported: 'estimators', 'resource_managers'")
            return


if __name__ == "__main__":
    main()
