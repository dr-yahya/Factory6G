#!/usr/bin/env python3
"""
Main simulation script for 6G Smart Factory Physical Layer System

This script runs BER/BLER simulations for the complete system with different
configurations (scenarios, CSI conditions, etc.).

Usage:
    python main.py [options]

Examples:
    # Run default simulation (UMi, perfect and imperfect CSI)
    python main.py

    # Run with specific scenario
    python main.py --scenario uma

    # Run only perfect CSI
    python main.py --perfect-csi-only

    # Custom Eb/No range
    python main.py --ebno-min -5 --ebno-max 15 --ebno-step 2
"""

import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.model import Model
from src.components.config import SystemConfig
from sionna.phy.utils import sim_ber
import sionna.phy

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def setup_gpu(gpu_num: int = 0):
    """Configure GPU settings"""
    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"✓ Using GPU: {gpus[0]}")
        except RuntimeError as e:
            print(f"Warning: {e}")
    else:
        print("⚠ No GPU found, using CPU")


def run_simulation(
    scenario: str = "umi",
    perfect_csi_list: list = None,
    ebno_db_range: np.ndarray = None,
    batch_size: int = 128,
    max_mc_iter: int = 1000,
    num_target_block_errors: int = 1000,
    target_bler: float = 1e-3,
    config: SystemConfig = None,
    save_results: bool = True,
    plot_results: bool = True,
    output_dir: str = "results",
    estimator_type: str = "ls",
    estimator_weights: str | None = None,
    estimator_kwargs: dict | None = None,
):
    """
    Run BER/BLER simulation for the system.
    
    Args:
        scenario: Channel scenario ("umi", "uma", "rma")
        perfect_csi_list: List of CSI conditions to test [True, False]
        ebno_db_range: Eb/No range in dB (default: -5 to 15 dB, step 2)
        batch_size: Batch size for simulation
        max_mc_iter: Maximum Monte Carlo iterations
        num_target_block_errors: Target number of block errors
        target_bler: Target BLER for early stopping
        config: Optional custom system configuration
        save_results: Whether to save results to file
        plot_results: Whether to generate plots
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing simulation results
    """
    # Default Eb/No range
    if ebno_db_range is None:
        ebno_db_range = np.arange(-5, 17, 2.0)
    
    # Create output directory
    if save_results or plot_results:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize results dictionary
    results = {
        "scenario": scenario,
        "ebno_db": list(ebno_db_range),
        "perfect_csi": [],
        "ber": [],
        "bler": [],
        "duration": None,
        "config": config.__dict__ if config else None,
        "estimator": estimator_type,
    }
    
    print("=" * 80)
    print("6G Smart Factory Physical Layer Simulation")
    print("=" * 80)
    print(f"Scenario: {scenario.upper()}")
    print(f"Eb/No range: {ebno_db_range[0]:.1f} to {ebno_db_range[-1]:.1f} dB")
    print(f"Batch size: {batch_size}")
    print(f"CSI conditions: {perfect_csi_list}")
    print(f"Estimator: {estimator_type.upper()}")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run simulation for each CSI condition
    for perfect_csi in perfect_csi_list:
        csi_str = "Perfect" if perfect_csi else "Imperfect"
        print(f"\n[{csi_str} CSI] Running simulation...")
        
        model_kwargs = dict(
            scenario=scenario,
            perfect_csi=perfect_csi,
            estimator_type=estimator_type,
            estimator_weights=estimator_weights,
            estimator_kwargs=estimator_kwargs,
        )
        if config is not None:
            model_kwargs["config"] = config
        model = Model(**model_kwargs)
        
        # Run BER simulation
        try:
            ber, bler = sim_ber(
                model,
                ebno_db_range,
                batch_size=batch_size,
                max_mc_iter=max_mc_iter,
                num_target_block_errors=num_target_block_errors,
                target_bler=target_bler
            )
            
            # Store results
            results["perfect_csi"].append(perfect_csi)
            results["ber"].append(list(ber.numpy()))
            results["bler"].append(list(bler.numpy()))
            
            print(f"[{csi_str} CSI] ✓ Simulation completed")
            
        except Exception as e:
            print(f"[{csi_str} CSI] ✗ Error: {e}")
            results["perfect_csi"].append(perfect_csi)
            results["ber"].append(None)
            results["bler"].append(None)
    
    # Calculate total duration
    results["duration"] = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"Simulation completed in {results['duration']:.2f} seconds")
    print("=" * 80)
    
    # Save results
    if save_results:
        save_simulation_results(results, output_dir, scenario, estimator_type)
    
    # Generate plots
    if plot_results:
        plot_simulation_results(results, output_dir, scenario, estimator_type)
    
    return results


def save_simulation_results(results: dict, output_dir: str, scenario: str, estimator: str):
    """Save simulation results to file"""
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/simulation_results_{scenario}_{estimator}_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    results_copy = results.copy()
    results_copy["ebno_db"] = [float(x) for x in results_copy["ebno_db"]]
    
    for i in range(len(results_copy["ber"])):
        if results_copy["ber"][i] is not None:
            results_copy["ber"][i] = [float(x) for x in results_copy["ber"][i]]
            results_copy["bler"][i] = [float(x) for x in results_copy["bler"][i]]
    
    with open(filename, 'w') as f:
        json.dump(results_copy, f, indent=2)
    
    print(f"✓ Results saved to: {filename}")


def plot_simulation_results(results: dict, output_dir: str, scenario: str, estimator: str):
    """Generate and save plots for simulation results"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot BER
    ax1.set_xlabel(r"$E_b/N_0$ (dB)")
    ax1.set_ylabel("BER")
    ax1.set_yscale('log')
    ax1.grid(which="both", alpha=0.3)
    ax1.set_title(f"Bit Error Rate - {scenario.upper()} ({estimator.upper()})")
    
    # Plot BLER
    ax2.set_xlabel(r"$E_b/N_0$ (dB)")
    ax2.set_ylabel("BLER")
    ax2.set_yscale('log')
    ax2.grid(which="both", alpha=0.3)
    ax2.set_title(f"Block Error Rate - {scenario.upper()} ({estimator.upper()})")
    
    # Plot data for each CSI condition
    colors = ['r', 'b', 'g', 'm', 'c']
    linestyles = ['-', '--', '-.', ':']
    
    for i, perfect_csi in enumerate(results["perfect_csi"]):
        if results["ber"][i] is None:
            continue
        
        csi_str = "Perfect" if perfect_csi else "Imperfect"
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        
        # BER plot
        ax1.semilogy(
            results["ebno_db"],
            results["ber"][i],
            color=color,
            linestyle=linestyle,
            marker='o',
            label=f"{csi_str} CSI",
            linewidth=2,
            markersize=6
        )
        
        # BLER plot
        ax2.semilogy(
            results["ebno_db"],
            results["bler"][i],
            color=color,
            linestyle=linestyle,
            marker='s',
            label=f"{csi_str} CSI",
            linewidth=2,
            markersize=6
        )
    
    # Set y-axis limits
    ax1.set_ylim([1e-5, 1])
    ax2.set_ylim([1e-4, 1])
    
    # Add legends
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"{output_dir}/simulation_plot_{scenario}_{estimator}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_filename}")
    
    # Also save as PDF
    pdf_filename = f"{output_dir}/simulation_plot_{scenario}_{estimator}_{timestamp}.pdf"
    plt.savefig(pdf_filename, bbox_inches='tight')
    
    plt.close()


def print_results_summary(results: dict):
    """Print a summary of simulation results"""
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 80)
    if "estimator" in results:
        print(f"Estimator: {results['estimator'].upper()}")

    for i, perfect_csi in enumerate(results["perfect_csi"]):
        if results["ber"][i] is None:
            continue
        
        csi_str = "Perfect" if perfect_csi else "Imperfect"
        print(f"\n[{csi_str} CSI]")
        print("-" * 80)
        print(f"{'Eb/No [dB]':<12} {'BER':<15} {'BLER':<15}")
        print("-" * 80)
        
        for j, ebno in enumerate(results["ebno_db"]):
            ber = results["ber"][i][j]
            bler = results["bler"][i][j]
            print(f"{ebno:>8.1f}    {ber:>12.6e}  {bler:>12.6e}")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point"""
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
        '--estimator',
        nargs='+',
        default=['ls'],
        choices=['ls', 'neural'],
        help='Channel estimator(s) to evaluate (default: ls). Provide multiple to compare.'
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
    
    args = parser.parse_args()
    
    # Setup GPU
    setup_gpu(args.gpu)
    
    # Set random seed
    sionna.phy.config.seed = args.seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Determine CSI conditions
    if args.perfect_csi_only:
        perfect_csi_list = [True]
    elif args.imperfect_csi_only:
        perfect_csi_list = [False]
    else:
        perfect_csi_list = [True, False]
    
    # Create Eb/No range
    ebno_db_range = np.arange(args.ebno_min, args.ebno_max + args.ebno_step, args.ebno_step)
    
    results_all = []
    for estimator in args.estimator:
        estimator_kwargs = {}
        estimator_weights = None
        if estimator == 'neural':
            estimator_kwargs['hidden_units'] = args.neural_hidden_units
            estimator_weights = args.neural_weights
            if estimator_weights is None:
                print("⚠ Neural estimator selected without --neural-weights. Training script should be run beforehand or weights path provided.")

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
        )

        print_results_summary(results)
        results_all.append(results)

    return results_all if len(results_all) > 1 else results_all[0]


if __name__ == "__main__":
    main()

