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
import logging
import warnings
from pathlib import Path
from typing import Any, Optional

# Add src to path before importing project modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now safe to import other modules directly
import numpy as np
import tensorflow as tf
from src.sim.simulation import run_simulation_campaign
from src.sim.env import configure_env, setup_gpu


class _StderrFilter:
    """Filter XLA allocator warnings from stderr."""

    _XLA_PATTERNS = [
        re.compile(r'.*Allocation.*exceeds.*free system memory.*'),
        re.compile(r'.*external/local_xla/xla/tsl/framework/cpu_allocator_impl.*'),
    ]

    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, message):
        if not any(p.search(message) for p in self._XLA_PATTERNS):
            self.original_stderr.write(message)

    def flush(self):
        self.original_stderr.flush()


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file and flatten for system use."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)

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

    # Logging Setup
    logging.basicConfig(level=getattr(logging, log_level_str.upper(), logging.INFO))
    sys.stderr = _StderrFilter(sys.stderr)
    
    # Configure logging levels
    _level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.getLogger().setLevel(_level)
    tf.get_logger().setLevel('ERROR' if _level > logging.DEBUG else 'INFO')
    
    # Suppress Python warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    logging.getLogger('tensorflow').setLevel(logging.ERROR if _level > logging.DEBUG else logging.INFO)
    logging.getLogger('absl').setLevel(logging.ERROR if _level > logging.DEBUG else logging.INFO)
    
    setup_gpu(gpu_id, force_cpu=force_cpu)
    
    # Set random seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Determine what to run based on simulation type
    sim_type = sim_config.get("type", "estimators")
    
    print(f"Loaded configuration for scenario: {system_config.get('scenario')}")
    print(f"Starting execution with type: {sim_type}")

    # Construct the unified configuration for run_simulation_campaign
    campaign_config = {
        "system_config": system_config,
        "output_dir": os.path.join(sim_config.get("output_dir", "results"), "campaign"),
        "batch_size": scenario_params.get("batch_size", 32),
        "total_batches": scenario_params.get("total_batches", 10),
        "plot_results": sim_config.get("plot_results", True),
    }

    # Eb/No Range
    ebno_min = scenario_params.get("ebno_min", 0.0)
    ebno_max = scenario_params.get("ebno_max", 20.0)
    ebno_step = scenario_params.get("ebno_step", 5.0)
    campaign_config["ebno_db_range"] = np.arange(ebno_min, ebno_max + ebno_step, ebno_step).tolist()

    if sim_type == "estimators":
        # Check for estimators list in config, else default
        estimators = scenario_params.get("estimators", ["ls", "dft", "lmmse", "pso", "perfect"])
        campaign_config["estimators"] = estimators
        
    elif sim_type == "resource_managers":
        # Check for resource_managers list in config, else default
        managers = rm_params.get("resource_managers", ["static", "round_robin", "max_throughput", "pf", "cnn"])
        campaign_config["resource_managers"] = managers
        campaign_config["num_active_users"] = rm_params.get("num_active_users", 2) # Pass specific RM param
        campaign_config["cnn_model_path"] = rm_params.get("cnn_model_path", "models/cnn_resource_manager.h5") # Pass specific RM param
        
    else:
        print(f"Unknown simulation type: {sim_type}. Supported: 'estimators', 'resource_managers'")
        return

    # Run the campaign
    run_simulation_campaign(campaign_config)


if __name__ == "__main__":
    main()
