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
import matplotlib.pyplot as plt
from src.models.resource_manager import StaticResourceManager
from src.models.cnn_resource_manager import CNNResourceManager
from src.sim.metrics import MetricsAccumulator
from src.sim.runner import run_simulation
from src.sim.plotting import plot_simulation_results
from src.sim.results import save_simulation_results
from src.sim.env import configure_env, setup_gpu
from src.components.config import SystemConfig


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
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        # Fallback to older config name if primary not found
        fallback = "min_6g_params_config.json"
        if os.path.exists(fallback):
            print(f"⚠ config.json not found, falling back to {fallback}")
            with open(fallback, 'r') as f:
                return json.load(f)
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def run_estimators_simulation(config_data: dict):
    """Run standard channel estimator comparison simulation/benchmark."""
    sim_config = config_data.get("simulation", {})
    scenario_params = config_data.get("scenario_params", {})
    
    output_dir = os.path.join(sim_config.get("output_dir", "results"), "benchmarks")
    batch_size = scenario_params.get("batch_size", 32)
    estimators = scenario_params.get("estimators", ["ls", "dft", "lmmse", "pso", "perfect"])
    
    # Eb/No Range from config
    ebno_min = scenario_params.get("ebno_min", 0.0)
    ebno_max = scenario_params.get("ebno_max", 20.0)
    ebno_step = scenario_params.get("ebno_step", 5.0)
    ebno_db_range = np.arange(ebno_min, ebno_max + ebno_step, ebno_step)

    # Use the benchmark function which handles running and plotting comparison
    from src.sim.benchmarks import run_estimator_benchmark
    
    return run_estimator_benchmark(
        estimators=estimators,
        ebno_db_range=ebno_db_range,
        batch_size=batch_size,
        output_dir=output_dir,
    )


def run_resource_managers_benchmark(config_data: dict):
    """Run resource manager benchmark."""
    from src.sim.benchmarks import run_resource_manager_benchmark
    
    sim_config = config_data.get("simulation", {})
    scenario_params = config_data.get("scenario_params", {})
    rm_params = config_data.get("resource_manager_params", {})
    
    output_dir = os.path.join(sim_config.get("output_dir", "results"), "benchmarks")
    batch_size = scenario_params.get("batch_size", 32)
    managers_list = rm_params.get("resource_managers", ["static", "round_robin", "max_throughput", "pf", "cnn"])
    cnn_model_path = rm_params.get("cnn_model_path", "models/cnn_resource_manager.h5")
    
    # Eb/No Range from config (or default)
    ebno_min = scenario_params.get("ebno_min", 0.0)
    ebno_max = scenario_params.get("ebno_max", 20.0)
    ebno_step = scenario_params.get("ebno_step", 10.0)
    ebno_db_range = np.arange(ebno_min, ebno_max + ebno_step, ebno_step)
    
    return run_resource_manager_benchmark(
        managers_list=managers_list,
        ebno_db_range=ebno_db_range,
        batch_size=batch_size,
        cnn_model_path=cnn_model_path,
        output_dir=output_dir,
    )


def main():
    """Main entry point"""
    # Load configuration
    try:
        config_data = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    sim_config = config_data.get("simulation", {})
    sim_type = sim_config.get("type", "estimators")
    
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

    # Now safe to import TensorFlow/Sionna
    import tensorflow as tf
    import sionna
    
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

    print(f"Starting execution with type: {sim_type}")
    
    if sim_type == "estimators":
        run_estimators_simulation(config_data)
    elif sim_type == "resource_managers":
        run_resource_managers_benchmark(config_data)
    else:
        print(f"Unknown simulation type: {sim_type}. Supported: 'estimators', 'resource_managers'")


if __name__ == "__main__":
    main()
