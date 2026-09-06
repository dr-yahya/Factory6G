from __future__ import annotations

import pytest

from factory6g.sim.config import ConfigError, load_config

from .conftest import make_tiny_config, write_config


def test_valid_config_parses_into_normalized_object(tmp_path):
    config_path = write_config(tmp_path, make_tiny_config(str(tmp_path / "results")))
    config = load_config(config_path)
    assert config.system.scenario == "umi"
    assert config.monte_carlo.ebno_db_range == [0.0]
    assert config.monte_carlo.stop_policy == "sweep"
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


def test_locked_5g_scenario_rejects_other_values(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"]["scenario"] = "rma"
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_simulation_targets_are_rejected(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["simulation"]["targets"] = ["estimators"]
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_non_5g_radio_preset_values_are_rejected(tmp_path):
    # The 5G lock guards the numerology, not the fading model: `rayleigh` is a
    # supported and documented channel choice (`--channel rayleigh`).
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"]["carrier_frequency"] = 28.0e9
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_rayleigh_channel_model_is_accepted_under_the_5g_profile(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"]["channel_model_type"] = "rayleigh"
    config_path = write_config(tmp_path, config_data)
    assert load_config(config_path).system.channel_model_type == "rayleigh"


def test_unknown_channel_model_is_rejected(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"]["channel_model_type"] = "not_a_model"
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_default_profile_still_locks_the_5g_numerology(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"]["num_ofdm_symbols"] = 7
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_6g_profile_allows_fr3_carrier_and_mini_slots(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"].update(
        {
            "radio_profile": "6g_fr3",
            "carrier_frequency": 13.0e9,
            "subcarrier_spacing": 120000.0,
            "num_ofdm_symbols": 4,
            "pilot_ofdm_symbol_indices": [0, 2],
        }
    )
    config_path = write_config(tmp_path, config_data)
    system = load_config(config_path).system
    assert system.radio_profile == "6g_fr3"
    assert system.num_ofdm_symbols == 4


def test_6g_profile_rejects_out_of_band_carrier(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"].update({"radio_profile": "6g_fr3", "carrier_frequency": 3.5e9})
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_6g_profile_rejects_pilots_outside_the_mini_slot(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"].update(
        {
            "radio_profile": "6g_fr3",
            "carrier_frequency": 13.0e9,
            "subcarrier_spacing": 120000.0,
            "num_ofdm_symbols": 4,
            "pilot_ofdm_symbol_indices": [2, 11],
        }
    )
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_indoor_factory_scenarios_are_accepted(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"].update({"radio_profile": "custom", "scenario": "inf_dh"})
    config_path = write_config(tmp_path, config_data)
    assert load_config(config_path).system.scenario == "inf_dh"


def test_harq_rounds_must_be_at_least_one(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"]["harq_max_rounds"] = 0
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_non_5g_tx_pattern_is_rejected(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["transceiver"]["tx_pattern"] = "dipole"
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_resource_manager_kwargs_and_model_paths_parse(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["resource_managers"]["drl_model_path"] = "models/drl_policy.h5"
    config_data["resource_managers"]["kwargs"] = {
        "wmmse": {"iterations": 8},
        "queue_aware": {"arrival_rate": 0.4},
    }
    config_path = write_config(tmp_path, config_data)
    config = load_config(config_path)
    assert config.resource_managers.drl_model_path == "models/drl_policy.h5"
    assert config.resource_managers.kwargs["wmmse"]["iterations"] == 8
    assert config.resource_managers.kwargs["queue_aware"]["arrival_rate"] == 0.4


def test_threshold_stop_policy_requires_target_ber(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["monte_carlo"]["stop_policy"] = "threshold"
    config_data["monte_carlo"]["target_ber"] = None
    config_path = write_config(tmp_path, config_data)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_threshold_stop_policy_with_target_ber_parses(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["monte_carlo"]["stop_policy"] = "threshold"
    config_data["monte_carlo"]["target_ber"] = 1e-5
    config_path = write_config(tmp_path, config_data)
    config = load_config(config_path)
    assert config.monte_carlo.stop_policy == "threshold"
    assert config.monte_carlo.target_ber == 1e-5


def test_estimators_paired_reference_defaults_to_the_first_method(tmp_path):
    """Unset, the reference is the first enabled estimator, as it always was."""
    from factory6g.sim.config import EstimatorsConfig

    config = EstimatorsConfig.from_dict({"enabled": ["LS", "DFT"]})
    assert config.paired_reference is None
    assert config.enabled == ["ls", "dft"]


def test_estimators_paired_reference_is_lowercased_and_validated():
    """A comparison against a method that is not running is a silent wrong answer.

    "Beats LS" says nothing about beating fixed DFT, so the reference has to be
    nameable -- and naming one that was never run must fail loudly rather than
    fall back to the default.
    """
    from factory6g.sim.config import ConfigError, EstimatorsConfig

    config = EstimatorsConfig.from_dict(
        {"enabled": ["ls", "dft", "adaptive_window"], "paired_reference": "DFT"}
    )
    assert config.paired_reference == "dft"

    with pytest.raises(ConfigError, match="paired_reference"):
        EstimatorsConfig.from_dict({"enabled": ["ls", "dft"], "paired_reference": "lmmse"})
