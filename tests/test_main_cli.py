from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import make_tiny_config, write_config


@pytest.mark.slow
def test_main_cli_runs_fixed_flow_without_plots(tmp_path):
    output_dir = tmp_path / "results"
    config_data = make_tiny_config(str(output_dir))
    config_data["estimators"]["enabled"] = ["ls"]
    config_data["resource_managers"]["enabled"] = ["static"]
    config_path = write_config(tmp_path, config_data)

    result = subprocess.run(
        [sys.executable, "-m", "factory6g.cli.run", "--config", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    summary_json_files = list(output_dir.rglob("summary_v2.json"))
    summary_csv_files = list(output_dir.rglob("summary_v2.csv"))
    stage_json_files = list(output_dir.rglob("stage_results_v2.json"))
    stage_csv_files = list(output_dir.rglob("stage_results_v2.csv"))
    log_files = list(output_dir.rglob("simulation.log"))
    png_files = list(output_dir.rglob("*.png"))

    assert summary_json_files, "Expected summary JSON output."
    assert summary_csv_files, "Expected summary CSV output."
    assert len(stage_json_files) == 2, "Expected per-stage v2 JSON outputs."
    assert len(stage_csv_files) == 2, "Expected per-stage v2 CSV outputs."
    assert len(log_files) == 1, "Expected exactly one run log file."
    assert not png_files, "plot_results=False should not generate PNG plots."
    log_text = log_files[0].read_text(encoding="utf-8")
    assert log_text.strip(), "Run log file must not be empty."
    assert "Starting execution for fixed flow: estimators -> resource_managers" in log_text
    assert "Stage order: estimators -> resource_managers" in log_text

    stage_payload = json.loads(stage_json_files[0].read_text(encoding="utf-8"))
    assert {
        "schema_version",
        "run_id",
        "stage",
        "ebno_db_range",
        "confidence_level",
        "config_snapshot",
        "methods",
        "runtime_totals_sec",
    } <= set(stage_payload)

    payload = json.loads(summary_json_files[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0"
    assert payload["stage_order"] == ["estimators", "resource_managers"]
    assert {"estimators", "resource_managers"} <= set(payload["stage_paths"])


@pytest.mark.slow
def test_main_cli_generates_required_stage_plots(tmp_path):
    output_dir = tmp_path / "results"
    config_data = make_tiny_config(str(output_dir))
    config_data["simulation"]["plot_results"] = True
    config_data["estimators"]["enabled"] = ["ls"]
    config_data["resource_managers"]["enabled"] = ["static"]
    config_path = write_config(tmp_path, config_data)

    result = subprocess.run(
        [sys.executable, "-m", "factory6g.cli.run", "--config", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    required = [
        "ber_vs_ebno.png",
        "ber_raw_vs_ebno.png",
        "latency_vs_ebno.png",
        "throughput_vs_ebno.png",
        "power_vs_ebno.png",
        "runtime_by_method.png",
    ]
    # Plots are written twice on purpose: once inside each stage directory and
    # once under `overview/<stage>/`. Scope the check to the stage directories
    # so adding an overview view does not look like a duplicate.
    stage_dirs = [
        path
        for path in output_dir.rglob("stage_results_v2.json")
    ]
    assert len(stage_dirs) == 2, "Expected two stage directories."
    for filename in required:
        stage_plots = [path.parent / filename for path in stage_dirs]
        for plot_path in stage_plots:
            assert plot_path.exists(), f"Missing stage plot {plot_path}."
        overview_plots = list((output_dir.rglob(f"overview/*/{filename}")))
        assert overview_plots, f"Expected overview plots for {filename}."
