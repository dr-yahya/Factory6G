#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import random
import re
import sys
import warnings
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.sim.config import ConfigError, load_config
from src.sim.env import configure_env, setup_gpu


class _StreamFilter:
    """Filter noisy runtime warnings from stdout/stderr."""

    _XLA_PATTERNS = [
        re.compile(r".*Allocation.*exceeds.*free system memory.*"),
        re.compile(r".*external/local_xla/xla/tsl/framework/cpu_allocator_impl.*"),
        re.compile(r".*No supported GPU was found.*"),
        re.compile(r".*Matplotlib created a temporary cache directory.*"),
        re.compile(r".*Matplotlib is building the font cache; this may take a moment.*"),
        re.compile(r".*\.matplotlib is not a writable directory.*"),
    ]

    def __init__(self, original_stream):
        self.original_stream = original_stream

    def write(self, message):
        if not any(pattern.search(message) for pattern in self._XLA_PATTERNS):
            self.original_stream.write(message)

    def flush(self):
        self.original_stream.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Factory6G simulations.")
    parser.add_argument("--config", default="config.json", help="Path to the simulation config JSON file.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        config = load_config(args.config)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"Error loading configuration: {exc}")
        return 1

    sim_config = config.simulation
    configure_env(force_cpu=sim_config.force_cpu, gpu_num=sim_config.gpu_id)
    sys.stdout = _StreamFilter(sys.stdout)
    sys.stderr = _StreamFilter(sys.stderr)

    import numpy as np
    import sionna.phy
    import tensorflow as tf

    from src.sim.flow import run_simulation_flow

    logging.basicConfig(level=getattr(logging, sim_config.log_level, logging.INFO))
    level = getattr(logging, sim_config.log_level, logging.INFO)
    logging.getLogger().setLevel(level)
    tf.get_logger().setLevel("ERROR" if level > logging.DEBUG else "INFO")
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
    logging.getLogger("tensorflow").setLevel(logging.ERROR if level > logging.DEBUG else logging.INFO)
    logging.getLogger("absl").setLevel(logging.ERROR if level > logging.DEBUG else logging.INFO)

    if not sim_config.force_cpu:
        setup_gpu(sim_config.gpu_id, force_cpu=sim_config.force_cpu)

    random.seed(sim_config.seed)
    np.random.seed(sim_config.seed)
    tf.random.set_seed(sim_config.seed)
    sionna.phy.config.seed = sim_config.seed

    print(f"Loaded configuration for scenario: {config.system.scenario}")
    print("Starting execution for fixed flow: estimators -> resource_managers")
    run_simulation_flow(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
