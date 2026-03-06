from __future__ import annotations

from typing import Any

from src.models.model import Model
from src.sim.config import Factory6GConfig
from src.sim.flow import run_simulation_flow


def run_simulation_loop(config: Factory6GConfig) -> dict[str, Any]:
    """Backward-compatible shim to the canonical fixed-order flow."""
    return run_simulation_flow(config)


__all__ = ["Model", "run_simulation_loop"]
