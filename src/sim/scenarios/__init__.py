"""
Scenario definitions for 6G indoor smart factory simulations.

This package contains individual scenario files that define simulation
configurations for indoor smart factory environments.
"""

from __future__ import annotations

from typing import Dict
import importlib
import pkgutil
from pathlib import Path

from ..scenario_spec import ScenarioSpec

# Re-export ScenarioSpec for convenience
__all__ = ["ScenarioSpec", "SCENARIO_PRESETS"]


def _load_scenarios() -> Dict[str, ScenarioSpec]:
    """
    Dynamically load all scenarios from the scenarios folder.
    
    Scans the scenarios subdirectory and imports all scenario modules,
    collecting their SCENARIO objects into a dictionary.
    
    Returns:
        Dictionary mapping scenario names to ScenarioSpec objects.
    """
    scenarios = {}
    scenarios_dir = Path(__file__).parent
    
    # Import all modules in the scenarios directory
    for importer, modname, ispkg in pkgutil.iter_modules([str(scenarios_dir)]):
        if not ispkg and modname != "__init__":
            try:
                # Import the scenario module
                module = importlib.import_module(f"src.sim.scenarios.{modname}")
                # Get the SCENARIO object from the module
                if hasattr(module, "SCENARIO"):
                    scenario = module.SCENARIO
                    scenarios[scenario.name] = scenario
            except Exception as e:
                print(f"Warning: Failed to load scenario from {modname}: {e}")
    
    return scenarios


# Load scenarios dynamically from the scenarios folder
SCENARIO_PRESETS: Dict[str, ScenarioSpec] = _load_scenarios()

