from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from .conftest import make_tiny_config, write_config


@pytest.mark.slow
def test_main_cli_runs_one_point_estimator_mode_without_plots(tmp_path):
    output_dir = tmp_path / "results"
    config_data = make_tiny_config(str(output_dir))
    config_data["simulation"]["targets"] = ["estimators"]
    config_data["estimators"]["enabled"] = ["ls"]
    config_path = write_config(tmp_path, config_data)

    result = subprocess.run(
        [sys.executable, "main.py", "--config", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    json_files = list(output_dir.rglob("*.json"))
    csv_files = list(output_dir.rglob("*.csv"))
    png_files = list(output_dir.rglob("*.png"))

    assert json_files, "Expected simulation JSON output."
    assert csv_files, "Expected simulation CSV output."
    assert not png_files, "plot_results=False should not generate PNG plots."

    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert {"config", "results", "ebno_db_range", "timestamp", "run_label", "mode"} <= set(payload)
