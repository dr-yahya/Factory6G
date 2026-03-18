#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import logging
import random
import re
import sys
import warnings
from pathlib import Path
from typing import TextIO

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.sim.config import ConfigError, load_config
from src.sim.env import configure_env, setup_gpu
from src.sim.run_context import create_run_context


class _FilteredTeeStream:
    """Mirror stream output to console + log file while filtering noisy lines."""

    _NOISY_PATTERNS = [
        re.compile(r".*Allocation.*exceeds.*free system memory.*"),
        re.compile(r".*external/local_xla/xla/tsl/framework/cpu_allocator_impl.*"),
        re.compile(r".*No supported GPU was found.*"),
        re.compile(r".*Matplotlib created a temporary cache directory.*"),
        re.compile(r".*Matplotlib is building the font cache; this may take a moment.*"),
        re.compile(r".*\.matplotlib is not a writable directory.*"),
    ]

    def __init__(self, console_stream: TextIO, log_stream: TextIO):
        self.console_stream = console_stream
        self.log_stream = log_stream

    def write(self, message):
        if not message:
            return 0
        if any(pattern.search(message) for pattern in self._NOISY_PATTERNS):
            return len(message)
        self.console_stream.write(message)
        self.log_stream.write(message)
        return len(message)

    def flush(self):
        self.console_stream.flush()
        self.log_stream.flush()

    @property
    def encoding(self):
        return getattr(self.console_stream, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.console_stream, "errors", "strict")

    def isatty(self):
        return bool(getattr(self.console_stream, "isatty", lambda: False)())

    def fileno(self):
        return self.console_stream.fileno()

    def __getattr__(self, name: str):
        return getattr(self.console_stream, name)


def _configure_root_logging(level: int, console_stream: TextIO, log_path: Path) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler(console_stream)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)


def _build_run_suffix(config, args: argparse.Namespace) -> str:
    all_methods = list(config.estimators.enabled) + list(config.resource_managers.enabled)
    methods_part = "_".join(all_methods) if all_methods else "run"

    cmt = config.system.channel_model_type
    channel_part = config.system.scenario if cmt == "tr38901" else cmt

    modulation_map = {1: "bpsk", 2: "qpsk", 4: "16qam", 6: "64qam"}
    modulation_part = modulation_map.get(
        config.system.num_bits_per_symbol,
        f"{config.system.num_bits_per_symbol}bps",
    )

    return f"{methods_part}_{channel_part}_{modulation_part}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Factory6G simulations.")
    parser.add_argument("--config", default="config.json", help="Path to the simulation config JSON file.")
    parser.add_argument(
        "--resume",
        metavar="RUN_DIR",
        default=None,
        help="Resume an interrupted run from an existing run directory (e.g. results/20260315_092517_simulation).",
    )
    parser.add_argument(
        "--estimators",
        metavar="METHODS",
        default=None,
        help="Comma-separated estimator methods to run, e.g. --estimators ls,pso. Overrides config. Skips resource-manager stage unless --resource-managers is also given.",
    )
    parser.add_argument(
        "--resource-managers",
        metavar="METHODS",
        default=None,
        help="Comma-separated resource-manager methods to run, e.g. --resource-managers wmmse,drl. Overrides config. Skips estimator stage unless --estimators is also given.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle: TextIO | None = None

    try:
        config = load_config(args.config)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"Error loading configuration: {exc}")
        return 1

    # CLI method overrides: --estimators / --resource-managers filter which methods run.
    # If only one flag is given, the other stage is skipped (enabled=[]).
    if args.estimators is not None or args.resource_managers is not None:
        est_enabled = (
            [m.strip().lower() for m in args.estimators.split(",") if m.strip()]
            if args.estimators is not None
            else []
        )
        rm_enabled = (
            [m.strip().lower() for m in args.resource_managers.split(",") if m.strip()]
            if args.resource_managers is not None
            else []
        )
        config = dataclasses.replace(
            config,
            estimators=dataclasses.replace(config.estimators, enabled=est_enabled),
            resource_managers=dataclasses.replace(config.resource_managers, enabled=rm_enabled),
        )

    try:
        sim_config = config.simulation
        if args.resume:
            run_dir = Path(args.resume)
            # Extract run_id as the timestamp prefix (everything before the first _ after date)
            parts = run_dir.name.split("_")
            run_id = "_".join(parts[:2]) if len(parts) >= 2 else run_dir.name
            log_mode = "a"
        else:
            run_id, run_dir = create_run_context(
                sim_config.output_dir,
                suffix=_build_run_suffix(config, args),
            )
            log_mode = "w"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "simulation.log"

        try:
            log_handle = log_path.open(log_mode, encoding="utf-8")
        except OSError as exc:
            original_stderr.write(f"Error opening log file '{log_path}': {exc}\n")
            original_stderr.flush()
            return 1

        sys.stdout = _FilteredTeeStream(original_stdout, log_handle)
        sys.stderr = _FilteredTeeStream(original_stderr, log_handle)
        print(f"Log file: {log_path.resolve()}")

        configure_env(force_cpu=sim_config.force_cpu, gpu_num=sim_config.gpu_id)

        import numpy as np
        import sionna.phy
        import tensorflow as tf

        from src.sim.flow import run_simulation_flow

        level = getattr(logging, sim_config.log_level, logging.INFO)
        _configure_root_logging(level, original_stdout, log_path)
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
        run_simulation_flow(config, run_id=run_id, run_dir=run_dir)
        return 0
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if log_handle is not None:
            log_handle.flush()
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
