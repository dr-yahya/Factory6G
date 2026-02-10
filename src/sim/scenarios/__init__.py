"""
Scenario definitions for 6G indoor smart factory simulations.

This package contains individual scenario files that define simulation
configurations for indoor smart factory environments.
"""

from __future__ import annotations

import importlib

from .spec import ScenarioSpec

# Re-export ScenarioSpec for convenience
__all__ = ["ScenarioSpec", "SCENARIO_PRESETS"]


def _import_scenario(name):
    module = importlib.import_module(f".{name}", package=__package__)
    return module.SCENARIO


SCENARIO_PRESETS = {
    "6g_smart_factory_sionna_baseline": _import_scenario("6g_smart_factory_sionna_baseline"),
    "6g_pso_enhanced": _import_scenario("6g_pso_enhanced"),
}
