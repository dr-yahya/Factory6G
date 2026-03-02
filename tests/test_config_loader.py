from __future__ import annotations

import pytest

from src.sim.config import ConfigError, load_config

from .conftest import make_tiny_config, write_config


def test_valid_config_parses_into_normalized_object(tmp_path):
    config_path = write_config(tmp_path, make_tiny_config(str(tmp_path / "results")))
    config = load_config(config_path)
    assert config.system.scenario == "umi"
    assert config.monte_carlo.ebno_db_range == [0.0]
    assert config.system_runtime_config["tx_pattern"] == "tr38901"


def test_missing_required_keys_fail_fast(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    del config_data["system"]["scenario"]
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_unknown_keys_fail_fast(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["simulation"]["legacy"] = True
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_system_scenario_survives_normalization(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"]["scenario"] = "rma"
    config_path = write_config(tmp_path, config_data)
    config = load_config(config_path)
    assert config.system.scenario == "rma"
