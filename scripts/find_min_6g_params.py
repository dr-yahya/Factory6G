#!/usr/bin/env python3
"""
Find minimum 6G simulation parameters for channel estimation.

This script starts with high 6G-compliant parameters and decrementally reduces them
until finding the minimum viable parameters that still work correctly,
then saves the configuration.

Usage:
    python scripts/find_min_6g_params.py
"""

import os
import sys
import json
import traceback
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.sim.env import configure_env

# Configure environment before importing TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def test_configuration(
    batch_size: int,
    fft_size: int,
    num_bs_ant: int,
    num_ut: int,
    num_ut_ant: int,
    num_ofdm_symbols: int,
    timeout_seconds: int = 30,
    verbose: bool = True,
) -> tuple[bool, Optional[str]]:
    """
    Test a configuration by running a minimal simulation.
    
    Returns:
        (success: bool, error_message: Optional[str])
    """
    if verbose:
        print(f"    [TEST] Starting configuration test...")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        if verbose:
            print(f"    [TEST] Importing modules...")
        from src.components.config import SystemConfig
        from src.models.model import Model
        
        if verbose:
            print(f"    [TEST] Creating SystemConfig...")
        # Create configuration
        config = SystemConfig(
            fft_size=fft_size,
            num_bs_ant=num_bs_ant,
            num_ut=num_ut,
            num_ut_ant=num_ut_ant,
            num_ofdm_symbols=num_ofdm_symbols,
        )
        
        if verbose:
            print(f"    [TEST] Creating Model (scenario=umi, estimator=ls)...")
        # Create model with minimal settings
        model = Model(
            scenario="umi",
            perfect_csi=False,
            config=config,
            estimator_type="ls",  # Simplest estimator
        )
        
        if verbose:
            print(f"    [TEST] Model created successfully")
        
        # Try to run a small batch
        # Use a single Eb/No point for speed
        ebno_db = 5.0
        
        if verbose:
            print(f"    [TEST] Running batch (batch_size={batch_size}, ebno_db={ebno_db})...")
        
        # Run batch (timeout handled at higher level if needed)
        # Note: SIGALRM is Unix-only, so we'll rely on Python's timeout or manual checking
        result = model.run_batch(batch_size, ebno_db, include_details=False)
        
        if verbose:
            print(f"    [TEST] Batch completed, validating results...")
        
        # Check if result is valid
        if result is None:
            if verbose:
                print(f"    [TEST] ERROR: Model returned None")
            return False, "Model returned None"
        
        # Check if we got expected keys
        required_keys = ["bits", "bits_hat"]
        if not all(key in result for key in required_keys):
            missing = [k for k in required_keys if k not in result]
            if verbose:
                print(f"    [TEST] ERROR: Missing keys: {missing}, got: {list(result.keys())}")
            return False, f"Missing required keys in result: {list(result.keys())}"
        
        if verbose:
            print(f"    [TEST] ✓ Configuration test PASSED")
        return True, None
            
    except MemoryError as e:
        error_msg = f"Out of memory: {str(e)}"
        if verbose:
            print(f"    [TEST] ERROR: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_type = type(e).__name__
        error_msg = f"{error_type}: {str(e)}"
        if verbose:
            print(f"    [TEST] ERROR: {error_msg}")
            import traceback
            if verbose:
                print(f"    [TEST] Traceback:")
                for line in traceback.format_exc().split('\n')[:5]:
                    if line.strip():
                        print(f"      {line}")
        return False, error_msg


def find_min_6g_params(
    start_batch_size: int = 64,
    start_fft_size: int = 16384,
    start_num_bs_ant: int = 4096,
    start_num_ut: int = 256,
    start_num_ut_ant: int = 8,
    start_num_ofdm_symbols: int = 28,
    decrement_strategy: str = "balanced",
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """
    Find minimum 6G parameters by decrementally reducing them.
    
    Args:
        start_*: Starting (high) values for each parameter (6G maximums)
        decrement_strategy: "balanced" (decrease all) or "individual" (one at a time)
        timeout_seconds: Maximum time per test
    
    Returns:
        Dictionary with min_6g_params and test_history
    """
    print("=" * 80)
    print("Finding Minimum 6G Simulation Parameters (6G 3GPP Compliant)")
    print("=" * 80)
    print(f"Starting parameters (6G maximums):")
    print(f"  batch_size: {start_batch_size}")
    print(f"  fft_size: {start_fft_size} (6G: 512-16384)")
    print(f"  num_bs_ant: {start_num_bs_ant} (6G massive MIMO: 32-4096)")
    print(f"  num_ut: {start_num_ut} (6G: 8-256)")
    print(f"  num_ut_ant: {start_num_ut_ant} (6G: 2-8)")
    print(f"  num_ofdm_symbols: {start_num_ofdm_symbols} (3GPP standard: 14)")
    print(f"Decrement strategy: {decrement_strategy}")
    print("=" * 80)
    
    # Current parameters (ensure FFT is power of 2 and within 6G bounds)
    if start_fft_size > 16384:
        start_fft_size = 16384
    # Round to nearest power of 2
    log_val = np.log2(start_fft_size)
    start_fft_size = 2 ** int(np.floor(log_val))
    start_fft_size = min(start_fft_size, 16384)  # Ensure 6G maximum
    
    current = {
        "batch_size": start_batch_size,
        "fft_size": start_fft_size,
        "num_bs_ant": start_num_bs_ant,
        "num_ut": start_num_ut,
        "num_ut_ant": start_num_ut_ant,
        "num_ofdm_symbols": start_num_ofdm_symbols,
    }
    
    # Last known working parameters (starts as current)
    last_working = current.copy()
    
    # Test history
    history = []
    
    # Decrement divisors (conservative for 6G large parameters)
    if decrement_strategy == "balanced":
        # Decrease all parameters proportionally
        divisors = {
            "batch_size": 1.25,  # Divide by 1.25
            "fft_size": 1.26,  # ~1.26^3 ≈ 2 (halves every 3 steps)
            "num_bs_ant": 1.26,
            "num_ut": 1.26,
            "num_ut_ant": 1.26,
            "num_ofdm_symbols": 1.15,  # Smaller decrements for symbols
        }
    else:  # individual
        # Test one parameter at a time
        divisors = {
            "batch_size": 1.5,
            "fft_size": 1.414,  # sqrt(2)
            "num_bs_ant": 1.414,
            "num_ut": 1.414,
            "num_ut_ant": 1.414,
            "num_ofdm_symbols": 1.2,
        }
    
    iteration = 0
    max_iterations = 50  # Safety limit
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n[Iteration {iteration}] Testing configuration:")
        print(f"  batch_size: {current['batch_size']}")
        print(f"  fft_size: {current['fft_size']}")
        print(f"  num_bs_ant: {current['num_bs_ant']}")
        print(f"  num_ut: {current['num_ut']}")
        print(f"  num_ut_ant: {current['num_ut_ant']}")
        print(f"  num_ofdm_symbols: {current['num_ofdm_symbols']}")
        
        # Calculate memory estimate (rough)
        memory_estimate = (
            current['batch_size'] *
            current['fft_size'] *
            current['num_ofdm_symbols'] *
            current['num_bs_ant'] *
            current['num_ut'] *
            current['num_ut_ant'] *
            8 * 2  # complex64 = 8 bytes, 2 for Re/Im
        ) / (1024**3)  # GB
        print(f"  Estimated memory: {memory_estimate:.2f} GB")
        
        # Test configuration
        import time
        test_start = time.time()
        print(f"  [TEST START] {time.strftime('%H:%M:%S')}")
        
        success, error_msg = test_configuration(
            batch_size=int(current['batch_size']),
            fft_size=int(current['fft_size']),
            num_bs_ant=int(current['num_bs_ant']),
            num_ut=int(current['num_ut']),
            num_ut_ant=int(current['num_ut_ant']),
            num_ofdm_symbols=int(current['num_ofdm_symbols']),
            timeout_seconds=timeout_seconds,
            verbose=True,
        )
        
        test_duration = time.time() - test_start
        print(f"  [TEST END] Duration: {test_duration:.2f}s")
        
        test_result = {
            "iteration": iteration,
            "params": current.copy(),
            "success": success,
            "error": error_msg,
            "memory_estimate_gb": memory_estimate,
        }
        history.append(test_result)
        
        if success:
            print(f"  ✓ SUCCESS")
            last_working = current.copy()
            
            # Decrement parameters (with 6G 3GPP minimum bounds)
            if decrement_strategy == "balanced":
                # Decrease all parameters
                print(f"  [DECREMENT] Decreasing all parameters:")
                for key in current:
                    old_val = current[key]
                    new_val = int(current[key] / divisors[key])
                    
                    # Apply 6G 3GPP minimum bounds
                    if key == "fft_size":
                        # 6G: 512-16384, must be power of 2
                        new_val = max(new_val, 512)
                        # Round down to nearest power of 2
                        if new_val < 512:
                            new_val = 512  # 6G minimum
                        else:
                            # Round down to previous power of 2
                            log_val = np.log2(new_val)
                            new_val = 2 ** int(np.floor(log_val))
                            # Ensure it's at least 512
                            new_val = max(new_val, 512)
                    elif key == "num_bs_ant":
                        # 6G massive MIMO: minimum 32
                        new_val = max(new_val, 32)
                    elif key == "num_ut":
                        # 6G: minimum 8
                        new_val = max(new_val, 8)
                    elif key == "num_ut_ant":
                        # 6G: minimum 2
                        new_val = max(new_val, 2)
                    elif key == "num_ofdm_symbols":
                        # 3GPP: minimum 14
                        new_val = max(new_val, 14)
                    
                    current[key] = new_val
                    print(f"    {key}: {old_val} -> {current[key]} (÷{divisors[key]:.3f})")
                    
                    # Check if we've hit minimum bounds
                    if old_val == new_val:
                        print(f"      [MIN BOUND] {key} reached 6G minimum")
            else:
                # Individual strategy: cycle through parameters
                param_order = ["batch_size", "fft_size", "num_bs_ant", "num_ut", "num_ut_ant", "num_ofdm_symbols"]
                param_idx = (iteration - 1) % len(param_order)
                param_name = param_order[param_idx]
                old_val = current[param_name]
                new_val = int(current[param_name] / divisors[param_name])
                
                # Apply 6G constraints for individual strategy too
                if param_name == "fft_size":
                    new_val = max(new_val, 512)
                    if new_val < 512:
                        new_val = 512
                    else:
                        log_val = np.log2(new_val)
                        new_val = 2 ** int(np.floor(log_val))
                        new_val = max(new_val, 512)
                elif param_name == "num_bs_ant":
                    new_val = max(new_val, 32)
                elif param_name == "num_ut":
                    new_val = max(new_val, 8)
                elif param_name == "num_ut_ant":
                    new_val = max(new_val, 2)
                elif param_name == "num_ofdm_symbols":
                    new_val = max(new_val, 14)
                
                current[param_name] = new_val
                print(f"  [DECREMENT] Decreasing {param_name}: {old_val} -> {current[param_name]} (÷{divisors[param_name]:.3f})")
                
                # Check if we've hit minimum bounds
                if old_val == new_val:
                    print(f"      [MIN BOUND] {param_name} reached 6G minimum")
        else:
            print(f"  ✗ FAILED: {error_msg}")
            print(f"  [STATUS] Last working config is the minimum, will attempt binary search...")
            
            # If we're using balanced strategy and it failed, try binary search
            if decrement_strategy == "balanced":
                # Binary search between last_working and current
                print(f"\n  [BINARY SEARCH] Finding minimum between working and failing config...")
                low = {k: float(v) for k, v in current.items()}  # current failed, so it's lower
                high = {k: float(v) for k, v in last_working.items()}  # last_working succeeded, so it's higher
                
                # Binary search for each parameter
                print(f"  [BINARY SEARCH] Testing individual parameters...")
                for key in current:
                    print(f"  [BINARY SEARCH] Parameter: {key}")
                    # Find min value for this parameter
                    low_val = low[key]
                    high_val = high[key]
                    print(f"    Range: {low_val} -> {high_val}")
                    
                    # Binary search
                    for step in range(5):  # Max 5 binary search steps
                        mid_val = (low_val + high_val) / 2
                        test_params = last_working.copy()
                        
                        # Apply 6G constraints during binary search
                        if key == "fft_size":
                            # Round to nearest power of 2
                            mid_val_int = int(mid_val)
                            if mid_val_int < 512:
                                mid_val_int = 512
                            else:
                                log_val = np.log2(mid_val_int)
                                mid_val_int = 2 ** int(np.floor(log_val))
                                mid_val_int = max(mid_val_int, 512)
                            test_params[key] = mid_val_int
                        else:
                            test_params[key] = int(mid_val)
                        
                        print(f"    Step {step+1}/5: Testing {key}={test_params[key]}")
                        test_success, test_error = test_configuration(
                            batch_size=int(test_params['batch_size']),
                            fft_size=int(test_params['fft_size']),
                            num_bs_ant=int(test_params['num_bs_ant']),
                            num_ut=int(test_params['num_ut']),
                            num_ut_ant=int(test_params['num_ut_ant']),
                            num_ofdm_symbols=int(test_params['num_ofdm_symbols']),
                            timeout_seconds=timeout_seconds,
                            verbose=False,  # Less verbose during binary search
                        )
                        
                        if test_success:
                            print(f"      ✓ PASSED")
                            high_val = test_params[key]
                            last_working[key] = test_params[key]
                        else:
                            print(f"      ✗ FAILED: {test_error}")
                            low_val = test_params[key]
                    
                    # Final value is the last working one (minimum)
                    last_working[key] = int(high_val)
                    print(f"    Final minimum value for {key}: {last_working[key]}")
                
                break  # Exit main loop after binary search
            else:
                # Individual strategy: this parameter failed, we found minimum for previous
                break
        
        # Check if all parameters are at minimum bounds
        all_at_minimum = (
            current['fft_size'] == 512 and
            current['num_bs_ant'] == 32 and
            current['num_ut'] == 8 and
            current['num_ut_ant'] == 2 and
            current['num_ofdm_symbols'] == 14
        )
        if all_at_minimum and success:
            print(f"\n  [MINIMUM REACHED] All parameters at 6G minimum bounds")
            break
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print("Minimum working configuration:")
    for key, value in last_working.items():
        print(f"  {key}: {value}")
    
    final_memory = (
        last_working['batch_size'] *
        last_working['fft_size'] *
        last_working['num_ofdm_symbols'] *
        last_working['num_bs_ant'] *
        last_working['num_ut'] *
        last_working['num_ut_ant'] *
        8 * 2
    ) / (1024**3)
    print(f"  Estimated memory: {final_memory:.2f} GB")
    
    return {
        "min_6g_params": last_working,
        "test_history": history,
        "final_memory_gb": final_memory,
    }


def save_results(results: Dict[str, Any], output_file: str):
    """Save results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Also save a simple config file
    config_file = output_path.parent / "min_6g_params_config.json"
    with open(config_file, 'w') as f:
        json.dump(results["min_6g_params"], f, indent=2)
    print(f"✓ Config saved to: {config_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Find minimum 6G simulation parameters"
    )
    parser.add_argument(
        '--start-batch-size',
        type=int,
        default=64,
        help='Starting batch size (default: 64, high value for 6G)'
    )
    parser.add_argument(
        '--start-fft-size',
        type=int,
        default=16384,
        help='Starting FFT size (default: 16384 for 6G maximum, 3GPP: 512-16384, must be power of 2)'
    )
    parser.add_argument(
        '--start-num-bs-ant',
        type=int,
        default=4096,
        help='Starting number of BS antennas (default: 4096, 6G massive MIMO maximum)'
    )
    parser.add_argument(
        '--start-num-ut',
        type=int,
        default=256,
        help='Starting number of UTs (default: 256, 6G maximum)'
    )
    parser.add_argument(
        '--start-num-ut-ant',
        type=int,
        default=8,
        help='Starting number of UT antennas (default: 8, 6G maximum)'
    )
    parser.add_argument(
        '--start-num-ofdm-symbols',
        type=int,
        default=28,
        help='Starting number of OFDM symbols (default: 28, higher than 3GPP standard 14)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='balanced',
        choices=['balanced', 'individual'],
        help='Decrement strategy: balanced (all params) or individual (one at a time)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Timeout per test in seconds (default: 30)'
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
        help='Force CPU execution'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/min_6g_params.json',
        help='Output file path (default: results/min_6g_params.json)'
    )
    
    args = parser.parse_args()
    
    # Configure environment
    configure_env(force_cpu=args.cpu, gpu_num=args.gpu)
    
    # Import TensorFlow after environment is configured
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    if not args.cpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print(f"✓ Using GPU: {gpus[0]}")
            except RuntimeError as e:
                print(f"Warning: {e}")
        else:
            print("⚠ No GPU found, using CPU")
    
    # Run parameter finding
    try:
        results = find_min_6g_params(
            start_batch_size=args.start_batch_size,
            start_fft_size=args.start_fft_size,
            start_num_bs_ant=args.start_num_bs_ant,
            start_num_ut=args.start_num_ut,
            start_num_ut_ant=args.start_num_ut_ant,
            start_num_ofdm_symbols=args.start_num_ofdm_symbols,
            decrement_strategy=args.strategy,
            timeout_seconds=args.timeout,
        )
        
        # Save results
        save_results(results, args.output)
        
        print("\n" + "=" * 80)
        print("SUCCESS: Minimum 6G parameters found and saved")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

