#!/usr/bin/env python3
"""
Diagnostic script to check LLR (Log-Likelihood Ratio) quality.

This script analyzes LLR values from the demapper to diagnose decoder issues.
It checks LLR statistics, distributions, and quality metrics to determine
if poor LLR quality is causing decoder failures.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Virtual environment setup
from src.utils.setup_venv import setup_venv
setup_venv()

# Configure environment
from src.utils.env import configure_env
configure_env(force_cpu=True, gpu_num=0)

# Now safe to import TensorFlow/Sionna
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from src.models.model import Model
from src.components.config import SystemConfig


def analyze_llr_quality(model: Model, ebno_db: float, batch_size: int = 32):
    """
    Analyze LLR quality for a given Eb/No.
    
    Args:
        model: System model instance
        ebno_db: Eb/No in dB
        batch_size: Batch size for simulation
        
    Returns:
        Dictionary with LLR statistics and diagnostics
    """
    print(f"\n{'='*80}")
    print(f"Analyzing LLR Quality at Eb/No = {ebno_db:.1f} dB")
    print(f"{'='*80}")
    
    # Run a batch to get LLRs
    results = model.run_batch(batch_size, ebno_db, include_details=True)
    
    # Extract LLRs from the results
    # The model doesn't directly return LLRs, so we need to compute them
    # Let's manually run the receiver chain to get LLRs
    
    # Generate test signal
    x_rg, b = model.get_transmitter().call(batch_size)
    
    # Convert Eb/No to noise variance
    from sionna.phy.utils import ebnodb2no
    coderate = model.get_config().coderate
    num_bits_per_symbol = model.get_config().num_bits_per_symbol
    no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
    
    # Apply channel
    y, h = model.get_channel()(x_rg, no)
    
    # Get receiver
    receiver = model.get_receiver()
    
    # Estimate channel and get LLRs
    h_hat, err_var = receiver.estimate_channel(y, no)
    x_hat, no_eff = receiver.equalize(y, h_hat, err_var, no)
    llr = receiver.demap(x_hat, no_eff)
    
    # Convert to numpy for analysis
    llr_np = llr.numpy()
    
    # Compute statistics
    llr_flat = llr_np.flatten()
    
    stats = {
        'ebno_db': ebno_db,
        'mean': float(np.mean(llr_flat)),
        'std': float(np.std(llr_flat)),
        'min': float(np.min(llr_flat)),
        'max': float(np.max(llr_flat)),
        'median': float(np.median(llr_flat)),
        'q25': float(np.percentile(llr_flat, 25)),
        'q75': float(np.percentile(llr_flat, 75)),
        'abs_mean': float(np.mean(np.abs(llr_flat))),
        'abs_std': float(np.std(np.abs(llr_flat))),
        'num_zeros': int(np.sum(np.abs(llr_flat) < 1e-10)),
        'num_extreme': int(np.sum(np.abs(llr_flat) > 50)),
        'num_nan': int(np.sum(np.isnan(llr_flat))),
        'num_inf': int(np.sum(np.isinf(llr_flat))),
        'shape': llr_np.shape,
        'total_llrs': int(np.prod(llr_np.shape)),
    }
    
    # Check decoder output
    decoded, decoder_iter = receiver.decode(llr)
    decoder_iter_np = decoder_iter.numpy()
    
    stats['decoder_iter_mean'] = float(np.mean(decoder_iter_np))
    stats['decoder_iter_max'] = int(np.max(decoder_iter_np))
    stats['decoder_iter_min'] = int(np.min(decoder_iter_np))
    
    # Compute BER for reference
    decoded_np = decoded.numpy()
    b_np = b.numpy()
    
    # Reshape to compare
    b_flat = b_np.flatten()
    decoded_flat = decoded_np.flatten()
    
    if len(b_flat) == len(decoded_flat):
        bit_errors = np.sum(b_flat != decoded_flat)
        stats['ber'] = float(bit_errors / len(b_flat))
        stats['bit_errors'] = int(bit_errors)
        stats['total_bits'] = int(len(b_flat))
    else:
        stats['ber'] = None
        stats['bit_errors'] = None
        stats['total_bits'] = None
    
    # Print statistics
    print(f"\nLLR Statistics:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Total LLRs: {stats['total_llrs']:,}")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std: {stats['std']:.4f}")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    print(f"  Median: {stats['median']:.4f}")
    print(f"  Mean |LLR|: {stats['abs_mean']:.4f}")
    print(f"  Std |LLR|: {stats['abs_std']:.4f}")
    print(f"  25th percentile: {stats['q25']:.4f}")
    print(f"  75th percentile: {stats['q75']:.4f}")
    
    print(f"\nLLR Quality Checks:")
    print(f"  Near-zero LLRs (|LLR| < 1e-10): {stats['num_zeros']:,} ({100*stats['num_zeros']/stats['total_llrs']:.2f}%)")
    print(f"  Extreme LLRs (|LLR| > 50): {stats['num_extreme']:,} ({100*stats['num_extreme']/stats['total_llrs']:.2f}%)")
    print(f"  NaN values: {stats['num_nan']:,}")
    print(f"  Inf values: {stats['num_inf']:,}")
    
    print(f"\nDecoder Performance:")
    print(f"  Avg iterations: {stats['decoder_iter_mean']:.2f}")
    print(f"  Min iterations: {stats['decoder_iter_min']}")
    print(f"  Max iterations: {stats['decoder_iter_max']}")
    if stats['ber'] is not None:
        print(f"  BER: {stats['ber']:.6e}")
        print(f"  Bit errors: {stats['bit_errors']:,} / {stats['total_bits']:,}")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    issues = []
    
    if stats['abs_mean'] < 0.1:
        issues.append("⚠ LLR magnitudes are very small (mean |LLR| < 0.1) - decoder may struggle")
    elif stats['abs_mean'] < 0.5:
        issues.append("⚠ LLR magnitudes are small (mean |LLR| < 0.5) - decoder may have difficulty")
    
    if stats['num_zeros'] > stats['total_llrs'] * 0.1:
        issues.append(f"⚠ Too many near-zero LLRs ({100*stats['num_zeros']/stats['total_llrs']:.1f}%) - poor signal quality")
    
    if stats['num_extreme'] > stats['total_llrs'] * 0.1:
        issues.append(f"⚠ Too many extreme LLRs ({100*stats['num_extreme']/stats['total_llrs']:.1f}%) - possible numerical issues")
    
    if stats['num_nan'] > 0:
        issues.append(f"⚠ Found {stats['num_nan']} NaN values - system error!")
    
    if stats['num_inf'] > 0:
        issues.append(f"⚠ Found {stats['num_inf']} Inf values - numerical overflow!")
    
    if stats['decoder_iter_mean'] < 0.1:
        issues.append("⚠ Decoder iterations are near zero - decoder may not be running")
    
    if stats['ber'] is not None and stats['ber'] > 0.1:
        issues.append(f"⚠ BER is very high ({stats['ber']:.4f}) - system not working properly")
    
    if not issues:
        print("  ✓ LLR quality appears reasonable")
    else:
        for issue in issues:
            print(f"  {issue}")
    
    return stats, llr_np


def plot_llr_distribution(llr_data: np.ndarray, ebno_db: float, output_dir: Path):
    """Plot LLR distribution histogram."""
    llr_flat = llr_data.flatten()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'LLR Quality Analysis at Eb/No = {ebno_db:.1f} dB', fontsize=14)
    
    # Histogram of LLR values
    ax = axes[0, 0]
    ax.hist(llr_flat, bins=100, density=True, alpha=0.7, edgecolor='black')
    ax.set_xlabel('LLR Value')
    ax.set_ylabel('Density')
    ax.set_title('LLR Distribution')
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Histogram of |LLR| values
    ax = axes[0, 1]
    ax.hist(np.abs(llr_flat), bins=100, density=True, alpha=0.7, edgecolor='black', color='orange')
    ax.set_xlabel('|LLR| Value')
    ax.set_ylabel('Density')
    ax.set_title('|LLR| Distribution')
    ax.axvline(np.mean(np.abs(llr_flat)), color='r', linestyle='--', linewidth=2, 
               label=f'Mean = {np.mean(np.abs(llr_flat)):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # CDF of |LLR|
    ax = axes[1, 0]
    sorted_abs_llr = np.sort(np.abs(llr_flat))
    p = np.arange(1, len(sorted_abs_llr) + 1) / len(sorted_abs_llr)
    ax.plot(sorted_abs_llr, p, linewidth=2)
    ax.set_xlabel('|LLR| Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF of |LLR|')
    ax.grid(True, alpha=0.3)
    ax.axvline(0.1, color='r', linestyle='--', alpha=0.5, label='|LLR| = 0.1')
    ax.axvline(1.0, color='g', linestyle='--', alpha=0.5, label='|LLR| = 1.0')
    ax.legend()
    
    # Box plot
    ax = axes[1, 1]
    ax.boxplot(llr_flat, vert=True)
    ax.set_ylabel('LLR Value')
    ax.set_title('LLR Box Plot')
    ax.axhline(0, color='r', linestyle='--', linewidth=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / f'llr_quality_ebno_{ebno_db:.1f}dB.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_file}")
    plt.close()


def main():
    """Main diagnostic function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose LLR quality issues')
    parser.add_argument('--ebno', type=float, nargs='+', default=[10.0, 15.0, 20.0, 25.0, 30.0],
                        help='Eb/No values to test (dB)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for simulation')
    parser.add_argument('--output-dir', type=str, default='diagnostics',
                        help='Output directory for plots')
    parser.add_argument('--scenario', type=str, default='umi',
                        choices=['umi', 'uma', 'rma'],
                        help='Channel scenario')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("LLR Quality Diagnostic Tool")
    print("="*80)
    print(f"Scenario: {args.scenario}")
    print(f"Eb/No values: {args.ebno} dB")
    print(f"Batch size: {args.batch_size}")
    print("="*80)
    
    # Create model
    config = SystemConfig(scenario=args.scenario)
    model = Model(scenario=args.scenario, perfect_csi=False, config=config)
    
    # Analyze at different Eb/No values
    all_stats = []
    
    for ebno_db in args.ebno:
        try:
            stats, llr_data = analyze_llr_quality(model, ebno_db, args.batch_size)
            all_stats.append(stats)
            
            # Plot distribution
            plot_llr_distribution(llr_data, ebno_db, output_dir)
            
        except Exception as e:
            print(f"\n❌ Error at Eb/No = {ebno_db:.1f} dB: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    if len(all_stats) > 1:
        print(f"\n{'='*80}")
        print("Summary Comparison Across Eb/No Values")
        print(f"{'='*80}")
        print(f"{'Eb/No (dB)':>10} | {'Mean |LLR|':>12} | {'Std |LLR|':>12} | {'Dec Iter':>9} | {'BER':>12}")
        print("-" * 80)
        for stats in all_stats:
            print(f"{stats['ebno_db']:10.1f} | "
                  f"{stats['abs_mean']:12.4f} | "
                  f"{stats['abs_std']:12.4f} | "
                  f"{stats['decoder_iter_mean']:9.2f} | "
                  f"{(stats['ber'] if stats['ber'] is not None else float('nan')):12.6e}")
    
    print(f"\n{'='*80}")
    print("Diagnostic complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

