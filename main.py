#!/usr/bin/env python3
from __future__ import annotations
"""
Main simulation script for 6G Smart Factory Physical Layer System.

This script runs BER/BLER simulations for the complete OFDM-MIMO system with
different configurations (scenarios, CSI conditions, channel estimators, etc.).
It provides a comprehensive command-line interface for system evaluation and
performance analysis.

Theory:
    BER/BLER Simulation:
    
    The script performs Monte Carlo simulations to estimate:
    - Bit Error Rate (BER): P(b̂ ≠ b) = E[I(b̂ ≠ b)]
    - Block Error Rate (BLER): P(∃ i: b̂[i] ≠ b[i]) = 1 - (1 - BER)^n
    
    Simulation Process:
    1. Generate random information bits
    2. Transmit through system (encoder → modulator → channel → receiver → decoder)
    3. Compare received bits with transmitted bits
    4. Compute error rates: BER = (# bit errors) / (# total bits)
    5. Repeat for multiple channel realizations (Monte Carlo)
    
    Stopping Criteria:
    - Maximum iterations: Stop after max_mc_iter channel realizations
    - Target block errors: Stop after num_target_block_errors errors
    - Target BLER: Stop if BLER < target_bler (early stopping)
    
    Eb/No Range:
    - Eb/No: Energy per bit to noise power spectral density ratio
    - Eb/No (dB) = 10·log10(Eb/N0)
    - Higher Eb/No: Better performance (lower BER/BLER)
    - Typical range: -5 dB to 20 dB for evaluation
    
    Performance Analysis:
    - Compare perfect vs imperfect CSI
    - Compare different channel estimators
    - Compare different scenarios (UMi, UMa, RMa)
    - Generate performance curves (BER/BLER vs Eb/No)

Usage:
    python main.py [options]

Examples:
    # Run default 6G simulation (6g_baseline profile)
    python main.py

    # Run specific 6G scenario profile
    python main.py --scenario-profile 6g_baseline_perfect

    # Run multiple 6G scenario profiles
    python main.py --scenario-profile 6g_baseline 6g_static_rm

    # Run with manual parameters (bypass scenario profiles)
    python main.py --scenario-profile "" --scenario uma --estimator ls_lin

    # Run only perfect CSI
    python main.py --scenario-profile 6g_baseline --perfect-csi-only
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Any, Optional

# Add src to path before importing setup_venv
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Virtual environment setup - must be done before other imports
from src.utils.setup_venv import setup_venv
setup_venv()

# Now safe to import other modules
import numpy as np
import matplotlib.pyplot as plt
from src.models.resource_manager import StaticResourceManager
from src.sim.scenarios import SCENARIO_PRESETS, ScenarioSpec
from src.sim.metrics import MetricsAccumulator
from src.sim.runner import run_simulation
from src.sim.plotting import plot_simulation_results
from src.sim.results import save_simulation_results
from src.utils.env import configure_env, setup_gpu


def print_results_summary(results: dict):
    """Print concise summary tables for the collected metrics."""
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Scenario : {results.get('scenario', 'unknown')}")
    print(f"Estimator: {results.get('estimator', 'N/A').upper()}")
    if results.get("profile"):
        print(f"Profile : {results['profile']}")

    runs = results.get("runs", [])
    if not runs:
        print("No metrics available.")
        return

    for run in runs:
        csi_str = "Perfect" if run.get("perfect_csi") else "Imperfect"
        print(f"\n[{csi_str} CSI]")
        print("-" * 80)
        header = f"{'Eb/No [dB]':>10} | {'BER':>12} | {'BLER':>12} | {'SINR (dB)':>10} | {'NMSE (dB)':>10} | {'Avg Iter':>9}"
        print(header)
        print("-" * len(header))
        for metric in run.get("metrics", []):
            ebno = metric.get("ebno_db")
            overall = metric.get("overall", {})
            ber = overall.get("ber")
            bler = overall.get("bler")
            sinr_db = overall.get("sinr_db")
            nmse_db = overall.get("nmse_db")
            iter_avg = overall.get("decoder_iter_avg")
            print(
                f"{ebno:10.2f} | "
                f"{(ber if ber is not None else float('nan')):>12.3e} | "
                f"{(bler if bler is not None else float('nan')):>12.3e} | "
                f"{(sinr_db if sinr_db is not None else float('nan')):>10.3f} | "
                f"{(nmse_db if nmse_db is not None else float('nan')):>10.3f} | "
                f"{(iter_avg if iter_avg is not None else float('nan')):>9.3f}"
            )
    print("=" * 80)


def main():
    """Main entry point"""
    # Build CLI without importing TensorFlow yet
    parser = argparse.ArgumentParser(
        description="6G Smart Factory Physical Layer Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        default='umi',
        choices=['umi', 'uma', 'rma'],
        help='Channel scenario (default: umi)'
    )

    parser.add_argument(
        '--scenario-profile',
        nargs='+',
        default=['6g_baseline'],
        help=f"Run one or more predefined scenario presets ({', '.join(SCENARIO_PRESETS.keys())}). Default: 6g_baseline"
    )

    parser.add_argument(
        '--list-scenarios',
        action='store_true',
        help='List available scenario presets and exit.'
    )

    parser.add_argument(
        '--estimator',
        nargs='+',
        default=['ls_nn'],
        choices=['ls', 'ls_nn', 'ls_lin', 'neural'],
        help='Channel estimator(s): ls, ls_nn, ls_lin, neural (multiple allowed)'
    )

    parser.add_argument(
        '--neural-weights',
        type=str,
        default=None,
        help='Path to pretrained neural estimator weights (required when using --estimator neural).'
    )

    parser.add_argument(
        '--neural-hidden-units',
        type=int,
        nargs='+',
        default=[32, 32],
        help='Hidden layer sizes for the neural estimator (default: 32 32).'
    )
    
    parser.add_argument(
        '--perfect-csi-only',
        action='store_true',
        help='Run only perfect CSI simulation'
    )
    
    parser.add_argument(
        '--imperfect-csi-only',
        action='store_true',
        help='Run only imperfect CSI simulation'
    )
    
    parser.add_argument(
        '--ebno-min',
        type=float,
        default=-5.0,
        help='Minimum Eb/No in dB (default: -5.0)'
    )
    
    parser.add_argument(
        '--ebno-max',
        type=float,
        default=15.0,
        help='Maximum Eb/No in dB (default: 15.0)'
    )
    
    parser.add_argument(
        '--ebno-step',
        type=float,
        default=2.0,
        help='Eb/No step size in dB (default: 2.0)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for simulation (default: 128)'
    )
    
    parser.add_argument(
        '--max-iter',
        type=int,
        default=1000,
        help='Maximum Monte Carlo iterations (default: 1000)'
    )
    
    parser.add_argument(
        '--target-block-errors',
        type=int,
        default=1000,
        help='Target number of block errors (default: 1000)'
    )
    
    parser.add_argument(
        '--target-bler',
        type=float,
        default=1e-3,
        help='Target BLER for early stopping (default: 1e-3)'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device number (default: 0)'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU execution and silence CUDA library errors'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip generating plots'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving results'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Resource Management (Static) flags
    parser.add_argument(
        '--use-static-rm',
        action='store_true',
        help='Enable Static Resource Manager (scheduling/power control)'
    )
    parser.add_argument(
        '--active-ut-mask',
        type=int,
        nargs='+',
        default=None,
        help='Active UT mask, e.g., 1 0 1 0'
    )
    parser.add_argument(
        '--per-ut-power',
        type=float,
        nargs='+',
        default=None,
        help='Per-UT power scaling (linear), e.g., 1.0 0.5 2.0 0.5'
    )
    parser.add_argument(
        '--pilot-reuse-factor',
        type=int,
        default=None,
        help='Pilot reuse factor placeholder (integer)'
    )
    
    args = parser.parse_args()

    if args.list_scenarios:
        print("Available scenario profiles:")
        for key, spec in SCENARIO_PRESETS.items():
            print(f"  - {key}: {spec.description}")
        return
    
    # Configure environment BEFORE importing TensorFlow/Sionna
    configure_env(force_cpu=args.cpu, gpu_num=args.gpu)

    # Now safe to import TensorFlow/Sionna and configure runtime
    import tensorflow as tf
    import sionna
    tf.get_logger().setLevel('ERROR')
    setup_gpu(args.gpu)
    
    # Set random seed (Sionna >=0.16 has no global phy seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    results_all = []

    # Filter out empty strings from scenario_profile if provided
    if args.scenario_profile:
        args.scenario_profile = [p for p in args.scenario_profile if p and p.strip()]
        if not args.scenario_profile:
            args.scenario_profile = None

    if args.scenario_profile:
        profile_names = [name.lower() for name in args.scenario_profile]
        for profile_name in profile_names:
            if profile_name not in SCENARIO_PRESETS:
                print(f"⚠ Unknown scenario profile '{profile_name}'. Available: {', '.join(SCENARIO_PRESETS.keys())}")
                continue
            spec = SCENARIO_PRESETS[profile_name]
            scenario_name = spec.channel_scenario or args.scenario
            ebno_db_range = np.arange(spec.ebno_min, spec.ebno_max + spec.ebno_step, spec.ebno_step)

            rm = None
            if spec.resource_manager:
                rm = StaticResourceManager(**spec.resource_manager)

            estimators = spec.estimators or args.estimator
            for estimator in estimators:
                estimator_kwargs = {}
                if spec.estimator_kwargs:
                    if estimator in spec.estimator_kwargs and isinstance(spec.estimator_kwargs[estimator], dict):
                        estimator_kwargs = dict(spec.estimator_kwargs[estimator])
                    elif isinstance(spec.estimator_kwargs, dict):
                        estimator_kwargs = dict(spec.estimator_kwargs)
                estimator_weights = spec.estimator_weights
                if estimator == 'neural':
                    estimator_kwargs.setdefault('hidden_units', args.neural_hidden_units)
                    if estimator_weights is None:
                        estimator_weights = args.neural_weights

                profile_output_dir = Path(args.output_dir) / spec.name
                results = run_simulation(
                    scenario=scenario_name,
                    perfect_csi_list=spec.perfect_csi,
                    ebno_db_range=ebno_db_range,
                    batch_size=spec.batch_size,
                    max_mc_iter=spec.max_iter,
                    num_target_block_errors=spec.target_block_errors,
                    target_bler=spec.target_bler,
                    save_results=not args.no_save,
                    plot_results=not args.no_plot,
                    output_dir=str(profile_output_dir),
                    estimator_type=estimator,
                    estimator_weights=estimator_weights,
                    estimator_kwargs=estimator_kwargs,
                    resource_manager=rm,
                    resource_manager_config=spec.resource_manager,
                    profile_name=spec.name,
                )
                if spec.description:
                    print(f"\nDescription: {spec.description}")
                print_results_summary(results)
                results_all.append(results)
    else:
        # Determine CSI conditions from CLI switches
        if args.perfect_csi_only:
            perfect_csi_list = [True]
        elif args.imperfect_csi_only:
            perfect_csi_list = [False]
        else:
            perfect_csi_list = [True, False]

        ebno_db_range = np.arange(args.ebno_min, args.ebno_max + args.ebno_step, args.ebno_step)

        rm_config = None
        resource_manager = None
        if args.use_static_rm:
            rm_config = {
                "active_ut_mask": args.active_ut_mask,
                "per_ut_power": args.per_ut_power,
                "pilot_reuse_factor": args.pilot_reuse_factor,
            }
            rm_config = {k: v for k, v in rm_config.items() if v is not None}
            resource_manager = StaticResourceManager(**rm_config) if rm_config else None
        
        for estimator in args.estimator:
            estimator_kwargs = {}
            estimator_weights = None
            if estimator == 'neural':
                estimator_kwargs['hidden_units'] = args.neural_hidden_units
                estimator_weights = args.neural_weights
                if estimator_weights is None:
                    print("⚠ Neural estimator selected without --neural-weights. Provide weights before running.")

            results = run_simulation(
                scenario=args.scenario,
                perfect_csi_list=perfect_csi_list,
                ebno_db_range=ebno_db_range,
                batch_size=args.batch_size,
                max_mc_iter=args.max_iter,
                num_target_block_errors=args.target_block_errors,
                target_bler=args.target_bler,
                save_results=not args.no_save,
                plot_results=not args.no_plot,
                output_dir=args.output_dir,
                estimator_type=estimator,
                estimator_weights=estimator_weights,
                estimator_kwargs=estimator_kwargs,
                resource_manager=resource_manager,
                resource_manager_config=rm_config,
            )

            print_results_summary(results)
            results_all.append(results)

    if not results_all:
        return None

    return results_all if len(results_all) > 1 else results_all[0]


if __name__ == "__main__":
    main()

