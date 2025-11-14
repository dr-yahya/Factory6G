from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScenarioSpec:
    """Predefined simulation scenario including runtime parameters."""

    name: str
    description: str
    estimators: List[str] = field(default_factory=lambda: ["ls_lin"])
    perfect_csi: List[bool] = field(default_factory=lambda: [False])
    channel_scenario: Optional[str] = None
    ebno_min: float = 0.0
    ebno_max: float = 10.0
    ebno_step: float = 1.0
    batch_size: int = 64
    max_iter: int = 20
    target_block_errors: int = 200
    target_bler: float = 1e-3
    resource_manager: Optional[Dict[str, Any]] = None
    estimator_kwargs: Dict[str, Any] = field(default_factory=dict)
    estimator_weights: Optional[str] = None
    notes: Optional[str] = None


SCENARIO_PRESETS: Dict[str, ScenarioSpec] = {
    "baseline": ScenarioSpec(
        name="baseline",
        description="Baseline simulation without resource management",
        estimators=["ls_lin"],
        perfect_csi=[False],
        ebno_min=0.0,
        ebno_max=10.0,
        ebno_step=1.0,
        batch_size=64,
        max_iter=30,
        target_block_errors=300,
        target_bler=5e-4,
    ),
    "baseline_perfect": ScenarioSpec(
        name="baseline_perfect",
        description="Baseline with both perfect and imperfect CSI runs",
        estimators=["ls_lin"],
        perfect_csi=[True, False],
        ebno_min=0.0,
        ebno_max=10.0,
        ebno_step=1.0,
        batch_size=64,
        max_iter=30,
        target_block_errors=300,
        target_bler=5e-4,
    ),
    "static_rm": ScenarioSpec(
        name="static_rm",
        description="Static resource manager with gentle per-UT power shaping",
        estimators=["ls_lin"],
        perfect_csi=[False],
        resource_manager={
            "active_ut_mask": [1, 1, 1, 1],
            "per_ut_power": [1.10, 0.95, 1.05, 0.90],
        },
        ebno_min=0.0,
        ebno_max=10.0,
        ebno_step=1.0,
        batch_size=64,
        max_iter=30,
        target_block_errors=300,
        target_bler=5e-4,
        notes="All users scheduled; modest power offsets emphasize edge UTs without starving others.",
    ),
}

__all__ = ["ScenarioSpec", "SCENARIO_PRESETS"]
