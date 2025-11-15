#!/usr/bin/env python3
"""
Find maximum simulation parameters for channel estimation before system breaks.

This script incrementally increases simulation parameters (batch_size, fft_size,
num_bs_ant, num_ut, etc.) until the system runs out of memory or crashes,
then saves the last working configuration.

Usage:
    python scripts/find_max_params.py
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

# Configure environment before importing TensorFlow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def configure_env(force_cpu: bool, gpu_num: Optional[int]):
    """Configure environment variables BEFORE importing TensorFlow/Sionna."""
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif gpu_num is not None and os.getenv("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"


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


def find_max_params(
    start_batch_size: int = 8,
    start_fft_size: int = 512,
    start_num_bs_ant: int = 16,
    start_num_ut: int = 4,
    start_num_ut_ant: int = 1,
    start_num_ofdm_symbols: int = 14,
    increment_strategy: str = "balanced",
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """
    Find maximum parameters by incrementally increasing them.
    
    Args:
        start_*: Starting values for each parameter
        increment_strategy: "balanced" (increase all) or "individual" (one at a time)
        timeout_seconds: Maximum time per test
    
    Returns:
        Dictionary with max_params and test_history
    """
    print("=" * 80)
    print("Finding Maximum Simulation Parameters (6G 3GPP Compliant)")
    print("=" * 80)
    print(f"Starting parameters (6G standards):")
    print(f"  batch_size: {start_batch_size}")
    print(f"  fft_size: {start_fft_size} (6G: 512-16384)")
    print(f"  num_bs_ant: {start_num_bs_ant} (6G massive MIMO: 32-4096)")
    print(f"  num_ut: {start_num_ut} (6G: 8-256)")
    print(f"  num_ut_ant: {start_num_ut_ant} (6G: 2-8)")
    print(f"  num_ofdm_symbols: {start_num_ofdm_symbols} (3GPP standard: 14)")
    print(f"Increment strategy: {increment_strategy}")
    print("=" * 80)
    
    # Current working parameters (ensure FFT is power of 2 and >= 512)
    if start_fft_size < 512:
        start_fft_size = 512
    # Round to nearest power of 2
    log_val = np.log2(start_fft_size)
    start_fft_size = 2 ** int(np.ceil(log_val))
    start_fft_size = max(start_fft_size, 512)  # Ensure 6G minimum
    
    current = {
        "batch_size": start_batch_size,
        "fft_size": start_fft_size,
        "num_bs_ant": start_num_bs_ant,
        "num_ut": start_num_ut,
        "num_ut_ant": start_num_ut_ant,
        "num_ofdm_symbols": start_num_ofdm_symbols,
    }
    
    # Last known working parameters
    last_working = current.copy()
    
    # Test history
    history = []
    
    # Increment multipliers (conservative for 6G large parameters)
    if increment_strategy == "balanced":
        # Increase all parameters proportionally (smaller increments for 6G)
        multipliers = {
            "batch_size": 1.25,  # Smaller increments for 6G
            "fft_size": 1.26,  # ~1.26^3 ≈ 2 (doubles every 3 steps)
            "num_bs_ant": 1.26,
            "num_ut": 1.26,
            "num_ut_ant": 1.26,
            "num_ofdm_symbols": 1.15,  # Smaller increments for symbols
        }
    else:  # individual
        # Test one parameter at a time
        multipliers = {
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
            
            # Increment parameters (with 6G 3GPP bounds)
            if increment_strategy == "balanced":
                # Increase all parameters
                print(f"  [INCREMENT] Increasing all parameters:")
                for key in current:
                    old_val = current[key]
                    new_val = int(current[key] * multipliers[key])
                    
                    # Apply 6G 3GPP bounds
                    if key == "fft_size":
                        # 6G: 512-16384, must be power of 2
                        new_val = min(new_val, 16384)
                        # Round to nearest power of 2 (always round up to next power of 2)
                        if new_val < 512:
                            new_val = 512  # 6G minimum
                        else:
                            # Round up to next power of 2
                            log_val = np.log2(new_val)
                            new_val = 2 ** int(np.ceil(log_val))
                            # Ensure it's at least 512
                            new_val = max(new_val, 512)
                    elif key == "num_bs_ant":
                        # 6G massive MIMO: up to 4096
                        new_val = min(new_val, 4096)
                    elif key == "num_ut":
                        # 6G: reasonable upper bound
                        new_val = min(new_val, 256)
                    elif key == "num_ut_ant":
                        # 6G: typically 2-8
                        new_val = min(new_val, 8)
                    elif key == "num_ofdm_symbols":
                        # 3GPP: typically 14, but can go higher
                        new_val = min(new_val, 28)
                    
                    current[key] = new_val
                    print(f"    {key}: {old_val} -> {current[key]} (×{multipliers[key]:.3f})")
            else:
                # Individual strategy: cycle through parameters
                param_order = ["batch_size", "fft_size", "num_bs_ant", "num_ut", "num_ut_ant", "num_ofdm_symbols"]
                param_idx = (iteration - 1) % len(param_order)
                param_name = param_order[param_idx]
                old_val = current[param_name]
                new_val = int(current[param_name] * multipliers[param_name])
                
                # Apply 6G constraints for individual strategy too
                if param_name == "fft_size":
                    new_val = min(new_val, 16384)
                    if new_val < 512:
                        new_val = 512
                    else:
                        log_val = np.log2(new_val)
                        new_val = 2 ** int(np.ceil(log_val))
                        new_val = max(new_val, 512)
                elif param_name == "num_bs_ant":
                    new_val = min(new_val, 4096)
                elif param_name == "num_ut":
                    new_val = min(new_val, 256)
                elif param_name == "num_ut_ant":
                    new_val = min(new_val, 8)
                elif param_name == "num_ofdm_symbols":
                    new_val = min(new_val, 28)
                
                current[param_name] = new_val
                print(f"  [INCREMENT] Increasing {param_name}: {old_val} -> {current[param_name]} (×{multipliers[param_name]:.3f})")
        else:
            print(f"  ✗ FAILED: {error_msg}")
            print(f"  [STATUS] Last working config saved, will attempt binary search...")
            
            # If we're using balanced strategy and it failed, try binary search
            if increment_strategy == "balanced":
                # Binary search between last_working and current
                print(f"\n  [BINARY SEARCH] Finding limit between working and failing config...")
                low = {k: float(v) for k, v in last_working.items()}
                high = {k: float(v) for k, v in current.items()}
                
                # Binary search for each parameter
                print(f"  [BINARY SEARCH] Testing individual parameters...")
                for key in current:
                    print(f"  [BINARY SEARCH] Parameter: {key}")
                    # Find max value for this parameter
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
                                mid_val_int = 2 ** int(np.ceil(log_val))
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
                            low_val = test_params[key]
                            last_working[key] = test_params[key]
                        else:
                            print(f"      ✗ FAILED: {test_error}")
                            high_val = test_params[key]
                    
                    # Final value is the last working one
                    last_working[key] = int(low_val)
                    print(f"    Final value for {key}: {last_working[key]}")
                
                break  # Exit main loop after binary search
            else:
                # Individual strategy: this parameter failed, move to next
                break
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print("Last working configuration:")
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
        "max_params": last_working,
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
    config_file = output_path.parent / "max_params_config.json"
    with open(config_file, 'w') as f:
        json.dump(results["max_params"], f, indent=2)
    print(f"✓ Config saved to: {config_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Find maximum simulation parameters before system breaks"
    )
    parser.add_argument(
        '--start-batch-size',
        type=int,
        default=8,
        help='Starting batch size (default: 8 for 6G, conservative)'
    )
    parser.add_argument(
        '--start-fft-size',
        type=int,
        default=512,
        help='Starting FFT size (default: 512 for 6G minimum, 3GPP: 512-16384, must be power of 2)'
    )
    parser.add_argument(
        '--start-num-bs-ant',
        type=int,
        default=16,
        help='Starting number of BS antennas (default: 16, will increase to 32+ for 6G massive MIMO)'
    )
    parser.add_argument(
        '--start-num-ut',
        type=int,
        default=4,
        help='Starting number of UTs (default: 4, will increase to 8+ for 6G)'
    )
    parser.add_argument(
        '--start-num-ut-ant',
        type=int,
        default=1,
        help='Starting number of UT antennas (default: 1, will increase to 2+ for 6G)'
    )
    parser.add_argument(
        '--start-num-ofdm-symbols',
        type=int,
        default=14,
        help='Starting number of OFDM symbols (default: 14, 3GPP standard)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='balanced',
        choices=['balanced', 'individual'],
        help='Increment strategy: balanced (all params) or individual (one at a time)'
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
        default='results/max_params.json',
        help='Output file path (default: results/max_params.json)'
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
        results = find_max_params(
            start_batch_size=args.start_batch_size,
            start_fft_size=args.start_fft_size,
            start_num_bs_ant=args.start_num_bs_ant,
            start_num_ut=args.start_num_ut,
            start_num_ut_ant=args.start_num_ut_ant,
            start_num_ofdm_symbols=args.start_num_ofdm_symbols,
            increment_strategy=args.strategy,
            timeout_seconds=args.timeout,
        )
        
        # Save results
        save_results(results, args.output)
        
        print("\n" + "=" * 80)
        print("SUCCESS: Maximum parameters found and saved")
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

