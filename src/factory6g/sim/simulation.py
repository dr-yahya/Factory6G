from __future__ import annotations

from typing import Any

from factory6g.models.model import Model
from factory6g.sim.config import Factory6GConfig
from factory6g.sim.flow import run_simulation_flow


def run_simulation_loop(config: Factory6GConfig) -> dict[str, Any]:
    """Backward-compatible shim to the canonical fixed-order flow."""
    return run_simulation_flow(config)


__all__ = ["Model", "run_simulation_loop"]
