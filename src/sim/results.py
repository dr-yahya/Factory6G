"""
Result saving and loading utilities for 6G simulations.

This module provides functions to save and load simulation results,
including baseline comparison functionality.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def save_simulation_results(results: dict, output_dir: str) -> str:
    """
    Persist simulation results to disk and return the path.
    
    Args:
        results: Simulation results dictionary
        output_dir: Output directory path
        
    Returns:
        Path to saved results file
    """
    scenario = results.get("scenario", "unknown")
    estimator = results.get("estimator", "est")
    profile = results.get("profile")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_suffix = f"_{profile}" if profile else ""
    filename = f"{output_dir}/simulation_results_{scenario}_{estimator}{profile_suffix}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {filename}")
    return filename


def load_baseline_results(baseline_path: Optional[Path] = None, project_root: Optional[Path] = None) -> Optional[dict]:
    """
    Load baseline results for comparison.
    
    Args:
        baseline_path: Path to baseline results JSON file. If None, uses default location.
        project_root: Project root directory. If None, tries to infer from baseline_path.
    
    Returns:
        Baseline results dictionary or None if file not found.
    """
    if baseline_path is None:
        if project_root is None:
            # Try to infer from common locations
            baseline_path = Path("results") / "3gpp_release19_baseline" / "simulation_results.json"
        else:
            baseline_path = project_root / "results" / "3gpp_release19_baseline" / "simulation_results.json"
    
    if not baseline_path.exists():
        print(f"⚠ Baseline results not found at: {baseline_path}")
        print("  Comparison plots will show only 6G simulation results.")
        return None
    
    try:
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        print(f"✓ Loaded baseline results from: {baseline_path}")
        baseline_type = baseline.get("baseline_type", "Baseline")
        print(f"  Baseline type: {baseline_type}")
        return baseline
    except Exception as e:
        print(f"⚠ Error loading baseline results: {e}")
        return None

