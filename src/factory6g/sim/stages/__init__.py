from __future__ import annotations

__all__ = ["run_estimator_stage", "run_resource_manager_stage"]


def __getattr__(name: str):
    if name == "run_estimator_stage":
        from .estimators import run_estimator_stage

        return run_estimator_stage
    if name == "run_resource_manager_stage":
        from .resource_managers import run_resource_manager_stage

        return run_resource_manager_stage
    raise AttributeError(name)
