"""Scenario specification dataclass definition."""

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

