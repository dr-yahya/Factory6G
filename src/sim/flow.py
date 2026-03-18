from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from src.sim.config import Factory6GConfig
from src.sim.output import write_combined_modulation_plot, write_stage_outputs, write_summary_outputs
from src.sim.run_context import build_run_dir, create_run_context, generate_run_id
from src.sim.stages.common import fmt_elapsed
from src.sim.stages.estimators import run_estimator_stage
from src.sim.stages.resource_managers import run_resource_manager_stage

_MODULATION_DISPLAY = {2: "QPSK", 4: "16-QAM", 6: "64-QAM", 1: "BPSK"}


def _load_completed_stage(stage_dir: Path) -> dict[str, Any] | None:
    """Return saved stage result dict if the stage already finished, else None."""
    result_path = stage_dir / "stage_results_v2.json"
    if not result_path.exists():
        return None
    with result_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def run_simulation_flow(
    config: Factory6GConfig,
    *,
    run_id: str | None = None,
    run_dir: Path | None = None,
    modulations: list[tuple[str, int]] | None = None,
    channels: list[str] | None = None,
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

    # Defaults
    if not modulations:
        modulations = [("default", config.system.num_bits_per_symbol)]
    if not channels:
        channels = [config.system.channel_model_type]

    run_estimators = bool(config.estimators.enabled)
    run_rms = bool(config.resource_managers.enabled)

    active_stages = []
    if run_estimators:
        active_stages.append("estimators")
    if run_rms:
        active_stages.append("resource_managers")

    multi_mod = len(modulations) > 1
    multi_ch = len(channels) > 1

    mod_display_names = [_MODULATION_DISPLAY.get(bits, f"{bits}bps") for _, bits in modulations]

    print("=" * 70)
    print("  6G Smart Factory Simulation Flow")
    print("=" * 70)
    print(f"Stages: {' -> '.join(active_stages) if active_stages else '(none)'}")
    if run_estimators:
        print(f"Estimators: {config.estimators.enabled}")
    if run_rms:
        print(f"Resource managers: {config.resource_managers.enabled}")
    print(f"Modulations: {' | '.join(mod_display_names)}")
    print(f"Channels: {' | '.join(channels)}")
    print(f"Scenario: {config.system.scenario}")
    print(f"Eb/No Range: {config.monte_carlo.ebno_db_range}")
    print(f"Run directory: {run_dir}")
    print("-" * 70)

    flow_start = time.perf_counter()

    combined_plot_entries: list[tuple[str, str, dict[str, Any], list[float]]] = []
    all_stage_payloads: dict[str, dict[str, Any]] = {}
    all_stage_paths: dict[str, dict[str, str]] = {}

    for mod_label, bits_per_symbol in modulations:
        mod_display = _MODULATION_DISPLAY.get(bits_per_symbol, f"{bits_per_symbol}bps")
        mod_config = dataclasses.replace(
            config,
            system=dataclasses.replace(config.system, num_bits_per_symbol=bits_per_symbol),
        )

        for channel in channels:
            combo_config = dataclasses.replace(
                mod_config,
                system=dataclasses.replace(mod_config.system, channel_model_type=channel),
            )

            # Subdir naming based on how many dimensions vary
            if multi_mod and multi_ch:
                combo_key = f"{mod_label}_{channel}"
            elif multi_mod:
                combo_key = mod_label
            elif multi_ch:
                combo_key = channel
            else:
                combo_key = ""  # single combo → no subdir

            combo_dir = run_dir / combo_key if combo_key else run_dir
            combo_dir.mkdir(parents=True, exist_ok=True)

            # Label for combined plot
            if multi_mod and multi_ch:
                combo_label = f"{mod_display} – {channel}"
            elif multi_mod:
                combo_label = mod_display
            else:
                combo_label = channel

            print(f"\n{'=' * 70}")
            print(f"  {combo_label}")
            print(f"{'=' * 70}")

            stage_payloads: dict[str, dict[str, Any]] = {}
            stage_paths: dict[str, dict[str, str]] = {}

            if run_estimators:
                estimator_stage_dir = combo_dir / "estimators"
                saved_estimators = _load_completed_stage(estimator_stage_dir)
                if saved_estimators is not None:
                    print("[checkpoint] Estimator stage already complete — skipping.")
                    estimator_result = {
                        "stage": "estimators",
                        "ebno_db_range": saved_estimators["ebno_db_range"],
                        "methods": saved_estimators["methods"],
                        "runtime_totals_sec": saved_estimators["runtime_totals_sec"],
                    }
                    estimator_paths = {
                        "dir": str(estimator_stage_dir),
                        "json": str(estimator_stage_dir / "stage_results_v2.json"),
                        "csv": str(estimator_stage_dir / "stage_results_v2.csv"),
                    }
                else:
                    print("[estimators] Starting...")
                    _t = time.perf_counter()
                    estimator_result = run_estimator_stage(combo_config, checkpoint_dir=estimator_stage_dir)
                    print(f"[estimators] Done in {fmt_elapsed(time.perf_counter() - _t)}")
                    estimator_paths = write_stage_outputs(
                        run_id=run_id,
                        stage_name="estimators",
                        stage_result=estimator_result,
                        stage_dir=estimator_stage_dir,
                        config_snapshot=combo_config.to_dict(),
                        confidence_level=combo_config.monte_carlo.confidence_level,
                        plot_results=combo_config.simulation.plot_results,
                    )
                stage_payloads["estimators"] = {
                    "methods": estimator_result["methods"],
                    "runtime_totals_sec": estimator_result["runtime_totals_sec"],
                }
                stage_paths["estimators"] = estimator_paths
                combined_plot_entries.append((
                    combo_label, "estimators",
                    estimator_result["methods"],
                    estimator_result["ebno_db_range"],
                ))

            if run_rms:
                rm_stage_dir = combo_dir / "resource_managers"
                saved_rm = _load_completed_stage(rm_stage_dir)
                if saved_rm is not None:
                    print("[checkpoint] Resource-manager stage already complete — skipping.")
                    rm_result = {
                        "stage": "resource_managers",
                        "ebno_db_range": saved_rm["ebno_db_range"],
                        "methods": saved_rm["methods"],
                        "runtime_totals_sec": saved_rm["runtime_totals_sec"],
                    }
                    rm_paths = {
                        "dir": str(rm_stage_dir),
                        "json": str(rm_stage_dir / "stage_results_v2.json"),
                        "csv": str(rm_stage_dir / "stage_results_v2.csv"),
                    }
                else:
                    print("[resource_managers] Starting...")
                    _t = time.perf_counter()
                    rm_result = run_resource_manager_stage(combo_config, checkpoint_dir=rm_stage_dir)
                    print(f"[resource_managers] Done in {fmt_elapsed(time.perf_counter() - _t)}")
                    rm_paths = write_stage_outputs(
                        run_id=run_id,
                        stage_name="resource_managers",
                        stage_result=rm_result,
                        stage_dir=rm_stage_dir,
                        config_snapshot=combo_config.to_dict(),
                        confidence_level=combo_config.monte_carlo.confidence_level,
                        plot_results=combo_config.simulation.plot_results,
                    )
                stage_payloads["resource_managers"] = {
                    "methods": rm_result["methods"],
                    "runtime_totals_sec": rm_result["runtime_totals_sec"],
                }
                stage_paths["resource_managers"] = rm_paths
                combined_plot_entries.append((
                    combo_label, "resource_managers",
                    rm_result["methods"],
                    rm_result["ebno_db_range"],
                ))

            # Merge into top-level dicts keyed by "{combo_key}/{stage}" or just "{stage}"
            for stage_key, payload in stage_payloads.items():
                dict_key = f"{combo_key}/{stage_key}" if combo_key else stage_key
                all_stage_payloads[dict_key] = payload
            for stage_key, paths in stage_paths.items():
                dict_key = f"{combo_key}/{stage_key}" if combo_key else stage_key
                all_stage_paths[dict_key] = paths

    # Combined BER plot when multiple combinations exist
    if config.simulation.plot_results and len(combined_plot_entries) > 1:
        write_combined_modulation_plot(run_dir, combined_plot_entries)

    print(f"\nSimulation complete. Total time: {fmt_elapsed(time.perf_counter() - flow_start)}")
    summary = write_summary_outputs(
        run_id=run_id,
        run_dir=run_dir,
        stage_order=list(all_stage_paths.keys()),
        stage_paths=all_stage_paths,
        stage_payloads=all_stage_payloads,
    )

    return {
        "schema_version": "2.0",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stage_order": active_stages,
        "stage_paths": all_stage_paths,
        "summary": summary["summary_payload"],
        "summary_paths": {
            "json": summary["summary_json"],
            "csv": summary["summary_csv"],
        },
    }
