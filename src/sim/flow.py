from __future__ import annotations

from pathlib import Path
from typing import Any

from src.sim.config import Factory6GConfig
from src.sim.output import write_stage_outputs, write_summary_outputs
from src.sim.run_context import build_run_dir, create_run_context, generate_run_id
from src.sim.stages.estimators import run_estimator_stage
from src.sim.stages.resource_managers import run_resource_manager_stage


def run_simulation_flow(
    config: Factory6GConfig,
    *,
    run_id: str | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    if run_id is None and run_dir is None:
        run_id, run_dir = create_run_context(
            config.simulation.output_dir,
            run_id=run_id,
        )
    elif run_id is not None and run_dir is None:
        run_dir = build_run_dir(config.simulation.output_dir, run_id)
    elif run_id is None and run_dir is not None:
        run_dir = Path(run_dir)
        suffix = "_simulation"
        if run_dir.name.endswith(suffix) and len(run_dir.name) > len(suffix):
            run_id = run_dir.name[: -len(suffix)]
        else:
            run_id = generate_run_id()
    else:
        run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  6G Smart Factory Simulation Flow")
    print("=" * 70)
    print("Stage order: estimators -> resource_managers")
    print(f"Scenario: {config.system.scenario}")
    print(f"Eb/No Range: {config.monte_carlo.ebno_db_range}")
    print(f"Run directory: {run_dir}")
    print("-" * 70)

    stage_order = ["estimators", "resource_managers"]
    stage_payloads: dict[str, dict[str, Any]] = {}
    stage_paths: dict[str, dict[str, str]] = {}

    estimator_result = run_estimator_stage(config)
    estimator_paths = write_stage_outputs(
        run_id=run_id,
        stage_name="estimators",
        stage_result=estimator_result,
        stage_dir=run_dir / "estimators",
        config_snapshot=config.to_dict(),
        confidence_level=config.monte_carlo.confidence_level,
        plot_results=config.simulation.plot_results,
    )
    stage_payloads["estimators"] = {
        "methods": estimator_result["methods"],
        "runtime_totals_sec": estimator_result["runtime_totals_sec"],
    }
    stage_paths["estimators"] = estimator_paths

    rm_result = run_resource_manager_stage(config)
    rm_paths = write_stage_outputs(
        run_id=run_id,
        stage_name="resource_managers",
        stage_result=rm_result,
        stage_dir=run_dir / "resource_managers",
        config_snapshot=config.to_dict(),
        confidence_level=config.monte_carlo.confidence_level,
        plot_results=config.simulation.plot_results,
    )
    stage_payloads["resource_managers"] = {
        "methods": rm_result["methods"],
        "runtime_totals_sec": rm_result["runtime_totals_sec"],
    }
    stage_paths["resource_managers"] = rm_paths

    summary = write_summary_outputs(
        run_id=run_id,
        run_dir=run_dir,
        stage_order=stage_order,
        stage_paths=stage_paths,
        stage_payloads=stage_payloads,
    )

    return {
        "schema_version": "2.0",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stage_order": stage_order,
        "stage_paths": stage_paths,
        "summary": summary["summary_payload"],
        "summary_paths": {
            "json": summary["summary_json"],
            "csv": summary["summary_csv"],
        },
    }
