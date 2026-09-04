from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from factory6g.models.resource_manager import (
    create_resource_manager,
    resolve_resource_manager_name,
)
from factory6g.sim.config import load_config
from factory6g.sim.types import ResourceManagerFeedback


def _make_feedback() -> ResourceManagerFeedback:
    h_hat = np.zeros((1, 1, 1, 4, 1, 1, 4), dtype=np.complex64)
    user_vectors = np.array(
        [
            [2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.5 + 0.0j, 0.2 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.1 + 0.0j],
        ],
        dtype=np.complex64,
    )
    for ut_idx in range(4):
        h_hat[0, 0, 0, ut_idx, 0, 0, :] = user_vectors[ut_idx]
    err_var = np.zeros_like(h_hat, dtype=np.float32)
    return ResourceManagerFeedback(h_hat=tf.constant(h_hat), err_var=tf.constant(err_var))


def test_resource_manager_config_parses_new_fields_without_conftest():
    config_data = {
        "simulation": {
            "gpu_id": 0,
            "force_cpu": True,
            "log_level": "INFO",
            "seed": 1,
            "output_dir": "results",
            "plot_results": False,
        },
        "monte_carlo": {
            "batch_size": 1,
            "min_batches": 1,
            "max_batches": 1,
            "target_block_errors": None,
            "target_ber": None,
            "confidence_level": 0.95,
            "min_total_bits": 0,
            "ebno_min": 0.0,
            "ebno_max": 0.0,
            "ebno_step": 1.0,
        },
        "estimators": {"enabled": ["ls"], "kwargs": {}},
        "resource_managers": {
            "enabled": ["wmmse", "queue_aware", "drl"],
            "cnn_model_path": None,
            "drl_model_path": "models/policy.h5",
            "num_active_users": 2,
            "kwargs": {
                "wmmse": {"iterations": 8},
                "queue_aware": {"arrival_rate": 0.4},
            },
        },
        "factory_scenario": {
            "room_dimensions": [10.0, 10.0, 4.0],
            "num_machines": 1,
            "machine_size_range": [[1.0, 1.5], [1.0, 1.5], [1.0, 1.5]],
            "materials": {
                "metal": {
                    "name": "factory_metal",
                    "relative_permittivity": 1.0,
                    "conductivity": 1e7,
                },
                "concrete": {
                    "name": "factory_concrete",
                    "relative_permittivity": 7.0,
                    "conductivity": 0.1,
                },
            },
        },
        "system": {
            "carrier_frequency": 3.5e9,
            "fft_size": 32,
            "subcarrier_spacing": 30000.0,
            "num_ofdm_symbols": 14,
            "cyclic_prefix_length": 20,
            "pilot_ofdm_symbol_indices": [2, 11],
            "num_bs_ant": 4,
            "num_ut": 4,
            "num_ut_ant": 1,
            "num_bits_per_symbol": 2,
            "coderate": 0.5,
            "num_decoding_iter": 2,
            "channel_model_type": "tr38901",
            "scenario": "umi",
            "direction": "uplink",
            "o2i_model": "low",
            "enable_pathloss": False,
            "enable_shadow_fading": False,
            "min_ut_velocity": 0.0,
            "max_ut_velocity": 0.0,
        },
        "ray_tracing": {"max_depth": 2, "samples_per_src": 128, "max_paths": 8},
        "transceiver": {
            "tx_height_offset": 1.0,
            "rx_height": 1.0,
            "antenna_spacing": 0.5,
            "tx_pattern": "tr38901",
            "tx_polarization": "cross",
            "rx_pattern": "iso",
            "rx_polarization": "V",
            "wall_thickness": 0.2,
            "room_padding": 1.0,
            "rx_boundary_padding": 1.0,
        },
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "config.json"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        config = load_config(config_path)

    assert config.resource_managers.drl_model_path == "models/policy.h5"
    assert config.resource_managers.kwargs["wmmse"]["iterations"] == 8
    assert config.resource_managers.kwargs["queue_aware"]["arrival_rate"] == 0.4


def test_wmmse_manager_returns_two_active_users_with_power():
    manager = create_resource_manager(
        "wmmse",
        num_ut=4,
        num_active=2,
        cnn_model_path=None,
        drl_model_path=None,
        manager_kwargs={"iterations": 6},
    )
    directives = manager.get_runtime_directives({"num_ut": 4}, ebno_db=5.0, feedback=_make_feedback())

    assert directives.active_ut_mask is not None
    assert directives.per_ut_power is not None
    assert sum(directives.active_ut_mask) == 2
    assert directives.active_ut_mask[0] == 1
    assert len(directives.per_ut_power) == 4
    assert all(0.0 <= value <= 1.0 for value in directives.per_ut_power)
    assert any(value > 0.0 for value in directives.per_ut_power)


def test_queue_aware_manager_uses_virtual_arrival_pressure_without_feedback():
    manager = create_resource_manager(
        "queue_aware",
        num_ut=4,
        num_active=1,
        cnn_model_path=None,
        drl_model_path=None,
        manager_kwargs={"arrival_rate": [0.0, 1.0, 0.0, 0.0], "utility_weight": 0.0},
    )
    directives = manager.get_runtime_directives({"num_ut": 4}, ebno_db=0.0, feedback=None)

    assert directives.active_ut_mask == [0, 1, 0, 0]
    assert directives.per_ut_power == [0.0, 1.0, 0.0, 0.0]


def test_drl_manager_heuristic_fallback_returns_valid_directives():
    manager = create_resource_manager(
        "drl",
        num_ut=4,
        num_active=2,
        cnn_model_path=None,
        drl_model_path=None,
        manager_kwargs={"temperature": 0.5, "fairness_weight": 0.4},
        strict_policy_loading=False,
    )
    directives = manager.get_runtime_directives({"num_ut": 4}, ebno_db=5.0, feedback=_make_feedback())

    assert directives.active_ut_mask is not None
    assert directives.per_ut_power is not None
    assert sum(directives.active_ut_mask) == 2
    assert directives.active_ut_mask[0] == 1
    assert len(directives.per_ut_power) == 4
    assert all(0.0 <= value <= 1.0 for value in directives.per_ut_power)


def test_strict_loading_refuses_to_run_the_heuristic_under_a_learned_name():
    """A `drl` curve must never be a hand-written rule in disguise."""
    with pytest.raises((RuntimeError, ValueError)):
        create_resource_manager(
            "drl",
            num_ut=4,
            num_active=2,
            cnn_model_path=None,
            drl_model_path=None,
        )


def test_drl_provenance_records_the_heuristic_fallback():
    manager = create_resource_manager(
        "drl",
        num_ut=4,
        num_active=2,
        cnn_model_path=None,
        drl_model_path=None,
        strict_policy_loading=False,
    )
    provenance = manager.provenance()
    assert provenance["policy_loaded"] is False
    assert provenance["actor"] == "heuristic_fallback"


def test_unknown_resource_manager_name_is_rejected_with_the_valid_list():
    with pytest.raises(ValueError, match="Valid names"):
        create_resource_manager(
            "totally_made_up",
            num_ut=4,
            num_active=2,
            cnn_model_path=None,
            drl_model_path=None,
        )


def test_registry_does_not_substring_match():
    """`static_drl` used to resolve to the static manager via substring matching."""
    with pytest.raises(ValueError):
        resolve_resource_manager_name("static_drl")
    assert resolve_resource_manager_name("max_throughput") == "max_throughput"
    assert resolve_resource_manager_name("PF") == "proportional_fair"


def test_static_manager_honours_manager_kwargs():
    """Static used to silently discard its configured kwargs."""
    manager = create_resource_manager(
        "static",
        num_ut=4,
        num_active=2,
        cnn_model_path=None,
        drl_model_path=None,
        manager_kwargs={"active_ut_mask": [1, 0, 1, 0], "per_ut_power": [0.5, 0.0, 0.5, 0.0]},
    )
    directives = manager.get_runtime_directives({"num_ut": 4}, ebno_db=5.0)
    assert directives.active_ut_mask == [1, 0, 1, 0]
    assert directives.per_ut_power == [0.5, 0.0, 0.5, 0.0]


def test_static_subset_matches_the_scheduler_load():
    """Equal-load control: `static` full-load is not comparable to a k-user scheduler."""
    full = create_resource_manager(
        "static", num_ut=4, num_active=2, cnn_model_path=None, drl_model_path=None
    ).get_runtime_directives({"num_ut": 4}, ebno_db=5.0)
    subset = create_resource_manager(
        "static_subset", num_ut=4, num_active=2, cnn_model_path=None, drl_model_path=None
    ).get_runtime_directives({"num_ut": 4}, ebno_db=5.0)

    assert sum(full.active_ut_mask) == 4
    assert sum(subset.active_ut_mask) == 2


class TestPerPointSchedulerState:
    """State must not leak between Eb/No points (review 1.4)."""

    def test_round_robin_rotates_independently_per_point(self):
        manager = create_resource_manager(
            "round_robin", num_ut=4, num_active=1, cnn_model_path=None, drl_model_path=None
        )
        config = {"num_ut": 4}
        first_at_0 = manager.get_runtime_directives(config, ebno_db=0.0).active_ut_mask
        # Visiting another point must not advance the 0 dB rotation.
        manager.get_runtime_directives(config, ebno_db=10.0)
        manager.get_runtime_directives(config, ebno_db=20.0)
        second_at_0 = manager.get_runtime_directives(config, ebno_db=0.0).active_ut_mask

        assert first_at_0.index(1) == 0
        assert second_at_0.index(1) == 1

    def test_proportional_fair_memory_is_isolated_per_point(self):
        manager = create_resource_manager(
            "pf", num_ut=4, num_active=1, cnn_model_path=None, drl_model_path=None
        )
        config = {"num_ut": 4, "coderate": 0.5, "num_bits_per_symbol": 2}
        feedback = _make_feedback()
        for _ in range(5):
            manager.get_runtime_directives(config, ebno_db=0.0, feedback=feedback)

        state = manager.export_state()["avg_rates"]
        assert set(state) == {"0.0"}

    def test_scheduler_state_round_trips_through_export_and_load(self):
        original = create_resource_manager(
            "round_robin", num_ut=4, num_active=1, cnn_model_path=None, drl_model_path=None
        )
        config = {"num_ut": 4}
        for _ in range(3):
            original.get_runtime_directives(config, ebno_db=0.0)

        restored = create_resource_manager(
            "round_robin", num_ut=4, num_active=1, cnn_model_path=None, drl_model_path=None
        )
        restored.load_state(original.export_state())

        assert (
            restored.get_runtime_directives(config, ebno_db=0.0).active_ut_mask
            == original.get_runtime_directives(config, ebno_db=0.0).active_ut_mask
        )
