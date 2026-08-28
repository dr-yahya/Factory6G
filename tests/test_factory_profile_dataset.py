from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import scripts.tools.generate_factory_dataset as dataset_module


def _fake_generate_profile_samples(
    config: dict,
    profile: dataset_module.ProfileDefinition,
    *,
    profile_index: int,
    num_samples: int,
    seed: int,
    preview: bool,
) -> dataset_module.GeneratedProfileData:
    del preview
    rng = np.random.default_rng(seed)
    room_dims = list(config["factory_scenario"]["room_dimensions"])
    transceiver = config["transceiver"]
    tx_position = [0.0, 0.0, float(room_dims[2]) - float(transceiver.get("tx_height_offset", 1.0))]
    rx_height = float(transceiver.get("rx_height", 1.0))
    max_paths = int(config["ray_tracing"].get("max_paths", 12))

    rx_positions = rng.uniform(-1.0, 1.0, size=(num_samples, 3)).astype(np.float32)
    rx_positions[:, 2] = rx_height
    paths_a = (
        rng.normal(0.0, 1.0, size=(num_samples, 1, 1, 1, max_paths, 1))
        + 1j * rng.normal(0.0, 1.0, size=(num_samples, 1, 1, 1, max_paths, 1))
    ).astype(np.complex64)
    paths_tau = np.full((num_samples, 1, max_paths), -1.0, dtype=np.float32)
    for sample_idx in range(num_samples):
        valid = min(4, max_paths)
        paths_tau[sample_idx, 0, :valid] = np.linspace(1e-9, 8e-9, valid, dtype=np.float32)

    csv_rows: list[dict] = []
    for sample_idx in range(num_samples):
        csv_rows.append(
            dataset_module._build_summary_row(
                sample_idx,
                profile=profile,
                profile_index=profile_index,
                rx_pos=rx_positions[sample_idx].tolist(),
                tx_pos=tx_position,
                a_val=paths_a[sample_idx : sample_idx + 1],
                tau_val=paths_tau[sample_idx : sample_idx + 1],
            )
        )

    machine_layout = [
        {"x": 0.0, "y": 0.0, "z": 0.0, "sx": 1.0, "sy": 1.0, "sz": 1.5},
    ]
    return dataset_module.GeneratedProfileData(
        profile=profile,
        profile_index=profile_index,
        seed=seed,
        frequency_hz=float(config["system"]["carrier_frequency"]),
        room_dimensions=room_dims,
        machine_layout=machine_layout,
        tx_position=tx_position,
        rx_positions=rx_positions,
        paths_a=paths_a,
        paths_tau=paths_tau,
        csv_rows=csv_rows,
    )


def _args_for(tmp_path: Path, *, seed: int, generate_visuals: bool) -> argparse.Namespace:
    return argparse.Namespace(
        config="config/config.json",
        profile_config="config/factory_size_profiles.json",
        profiles="S,M,L",
        samples_per_profile=2,
        output_dir=str(tmp_path / "data"),
        dataset_prefix="factory_dataset",
        seed=seed,
        generate_visuals=generate_visuals,
        visuals_dir=str(tmp_path / "visuals"),
        preview=False,
    )


def test_profile_loader_fails_when_scale_missing(tmp_path):
    payload = {
        "profiles": {
            "S": {
                "factory_scale": "S",
                "scenario_label": "x",
                "real_world_archetype": "x",
                "factory_scenario_overrides": {},
                "transceiver_overrides": {},
                "ray_tracing_overrides": {},
            },
            "M": {
                "factory_scale": "M",
                "scenario_label": "x",
                "real_world_archetype": "x",
                "factory_scenario_overrides": {},
                "transceiver_overrides": {},
                "ray_tracing_overrides": {},
            },
        }
    }
    config_path = tmp_path / "profiles.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing: L"):
        dataset_module.load_profile_defs(str(config_path))


def test_profile_loader_fails_when_profile_malformed(tmp_path):
    payload = {
        "profiles": {
            "S": {
                "factory_scale": "S",
                "scenario_label": "x",
                "real_world_archetype": "x",
                "factory_scenario_overrides": {},
                "transceiver_overrides": "bad",
                "ray_tracing_overrides": {},
            },
            "M": {
                "factory_scale": "M",
                "scenario_label": "x",
                "real_world_archetype": "x",
                "factory_scenario_overrides": {},
                "transceiver_overrides": {},
                "ray_tracing_overrides": {},
            },
            "L": {
                "factory_scale": "L",
                "scenario_label": "x",
                "real_world_archetype": "x",
                "factory_scenario_overrides": {},
                "transceiver_overrides": {},
                "ray_tracing_overrides": {},
            },
        }
    }
    config_path = tmp_path / "profiles.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="transceiver_overrides"):
        dataset_module.load_profile_defs(str(config_path))


def test_profile_pipeline_smoke_schema_and_visuals(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_module, "_generate_profile_samples", _fake_generate_profile_samples)
    args = _args_for(tmp_path, seed=123, generate_visuals=True)
    combined_paths = dataset_module.run_profile_pipeline(args)

    assert combined_paths["h5_path"].exists()
    assert combined_paths["csv_path"].exists()
    assert combined_paths["manifest_path"].exists()

    for scale in ("S", "M", "L"):
        h5_path = tmp_path / "data" / f"factory_dataset_{scale}_2.h5"
        csv_path = tmp_path / "data" / f"factory_dataset_{scale}_2.csv"
        assert h5_path.exists()
        assert csv_path.exists()
        with h5py.File(h5_path, "r") as h5f:
            assert {"rx_positions", "paths_a", "paths_tau"} <= set(h5f.keys())
            assert {"profile_index", "factory_scale", "scenario_label", "real_world_archetype"} <= set(h5f.keys())
            assert h5f["rx_positions"].shape[0] == 2
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert {"factory_scale", "scenario_label", "real_world_archetype"} <= set(reader.fieldnames or [])

    combined_h5 = tmp_path / "data" / "factory_dataset_SML_6.h5"
    combined_csv = tmp_path / "data" / "factory_dataset_SML_6.csv"
    manifest_path = tmp_path / "data" / "factory_dataset_SML_6_manifest.json"
    assert combined_h5.exists()
    assert combined_csv.exists()
    assert manifest_path.exists()

    with h5py.File(combined_h5, "r") as h5f:
        assert h5f["rx_positions"].shape[0] == 6
        profile_indices = h5f["profile_index"][:]
        unique, counts = np.unique(profile_indices, return_counts=True)
        assert dict(zip(unique.tolist(), counts.tolist())) == {0: 2, 1: 2, 2: 2}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["profile_index_mapping"] == {"S": 0, "M": 1, "L": 2}
    assert len(manifest["profiles"]) == 3

    for scale in ("S", "M", "L"):
        assert (tmp_path / "visuals" / f"layout_2d_{scale}.png").exists()
        assert (tmp_path / "visuals" / f"layout_3d_{scale}.png").exists()
        assert (tmp_path / "visuals" / f"kpi_panel_{scale}.png").exists()


def test_profile_pipeline_determinism_same_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_module, "_generate_profile_samples", _fake_generate_profile_samples)

    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    args1 = _args_for(run1, seed=77, generate_visuals=False)
    args2 = _args_for(run2, seed=77, generate_visuals=False)
    dataset_module.run_profile_pipeline(args1)
    dataset_module.run_profile_pipeline(args2)

    h5_run1 = run1 / "data" / "factory_dataset_S_2.h5"
    h5_run2 = run2 / "data" / "factory_dataset_S_2.h5"
    with h5py.File(h5_run1, "r") as f1, h5py.File(h5_run2, "r") as f2:
        np.testing.assert_allclose(f1["rx_positions"][:], f2["rx_positions"][:], rtol=0.0, atol=0.0)
        assert f1["paths_a"].shape == f2["paths_a"].shape
