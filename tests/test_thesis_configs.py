"""The documented result families must exist, load, and mean what the docs say.

Two defects motivated this file, both found with the configs missing entirely:

1. `docs/THESIS_RESULTS.md` -- the document that maps every thesis claim to the
   command reproducing it -- referenced `config/thesis/resource_managers_inf.json`,
   which did not exist. Four of the six documented families had no config in the
   repo at all; one survived only as two divergent copies inside evidence bundles.

2. `system.inf_hall_volume_m3` / `inf_hall_surface_m2` were plain floats
   defaulting to the *small* hall. `asdict()` always materialised them, so the
   geometry fallback in `ChannelModel._apply_inf_large_scale` was unreachable and
   every hall size simulated one delay spread -- while the selectivity report
   derived its numbers from the room dimensions and printed a varying sweep. The
   table in `THESIS_RESULTS.md` therefore described a channel nobody simulated.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from factory6g.components.inf_channel import (
    coherence_bandwidth_hz,
    hall_volume_and_surface,
    inf_delay_spread_seconds,
)
from factory6g.sim.config import ConfigError, load_config

from .conftest import make_tiny_config, write_config

REPO_ROOT = Path(__file__).resolve().parents[1]
THESIS_CONFIG_DIR = REPO_ROOT / "config" / "thesis"

# Selectivity (signal bandwidth / coherence bandwidth) as published in the
# result-families table of docs/THESIS_RESULTS.md.
DOCUMENTED_SELECTIVITY = {
    "estimators_inf_s.json": 7.3,
    "estimators_inf_m.json": 9.2,
    "estimators_inf_l.json": 12.1,
    "estimators_inf_narrowband.json": 0.45,
    "resource_managers_inf.json": 9.2,
}


def _thesis_configs() -> list[Path]:
    return sorted(THESIS_CONFIG_DIR.glob("*.json"))


def test_thesis_config_directory_is_populated():
    names = {p.name for p in _thesis_configs()}
    assert names >= set(DOCUMENTED_SELECTIVITY) | {"estimators_umi.json"}


@pytest.mark.parametrize("config_path", _thesis_configs(), ids=lambda p: p.name)
def test_every_thesis_config_loads(config_path):
    load_config(config_path)


def test_every_config_path_named_in_docs_exists():
    """Reproduction commands in the docs must point at files that are here."""
    pattern = re.compile(r"config/[A-Za-z0-9_/]+\.json")
    referenced: dict[str, set[str]] = {}
    doc_files = list((REPO_ROOT / "docs").glob("*.md")) + [REPO_ROOT / "README.md"]
    for doc in doc_files:
        for match in pattern.findall(doc.read_text(encoding="utf-8")):
            referenced.setdefault(match, set()).add(doc.name)

    missing = {
        path: sorted(sources)
        for path, sources in referenced.items()
        if not (REPO_ROOT / path).is_file()
    }
    assert not missing, f"documented config paths that do not exist: {missing}"

    for path in referenced:
        load_config(REPO_ROOT / path)


@pytest.mark.parametrize(
    "name,expected", sorted(DOCUMENTED_SELECTIVITY.items()), ids=lambda v: str(v)
)
def test_selectivity_matches_the_published_table(name, expected):
    """The hall the docs describe is the hall the channel actually builds."""
    config = load_config(THESIS_CONFIG_DIR / name)
    runtime = config.system_runtime_config
    volume, surface = hall_volume_and_surface(runtime["room_dimensions"])
    delay_spread = float(inf_delay_spread_seconds(volume, surface)[0])
    bandwidth = config.system.fft_size * config.system.subcarrier_spacing
    selectivity = bandwidth / coherence_bandwidth_hz(delay_spread)
    assert selectivity == pytest.approx(expected, abs=0.05)


def test_hall_overrides_are_absent_from_the_runtime_config(tmp_path):
    """Unset overrides must not reach the channel, or geometry never wins."""
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"].pop("inf_hall_volume_m3", None)
    config_data["system"].pop("inf_hall_surface_m2", None)
    config = load_config(write_config(tmp_path, config_data))

    assert config.system.inf_hall_volume_m3 is None
    assert config.system.inf_hall_surface_m2 is None
    runtime = config.system_runtime_config
    assert "inf_hall_volume_m3" not in runtime
    assert "inf_hall_surface_m2" not in runtime


def test_delay_spread_tracks_hall_size():
    """A factory-size sweep has to be a sweep over a real propagation property."""
    spreads = []
    for room in ([15.0, 15.0, 5.0], [25.0, 25.0, 6.0], [40.0, 40.0, 8.0]):
        volume, surface = hall_volume_and_surface(room)
        spreads.append(float(inf_delay_spread_seconds(volume, surface)[0]))
    assert spreads == sorted(spreads)
    assert spreads[-1] > spreads[0] * 1.5


def test_half_specified_hall_override_is_rejected(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"]["inf_hall_volume_m3"] = 1125.0
    config_data["system"].pop("inf_hall_surface_m2", None)
    with pytest.raises(ConfigError, match="set together or both omitted"):
        load_config(write_config(tmp_path, config_data))


def test_both_hall_overrides_together_are_honoured(tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["system"]["inf_hall_volume_m3"] = 2000.0
    config_data["system"]["inf_hall_surface_m2"] = 1000.0
    config = load_config(write_config(tmp_path, config_data))
    runtime = config.system_runtime_config
    assert runtime["inf_hall_volume_m3"] == 2000.0
    assert runtime["inf_hall_surface_m2"] == 1000.0


def test_no_config_carries_a_hall_override():
    """Regression: a hand-entered 900 m^2 contradicted the 750 m^2 geometry."""
    offenders = []
    for path in _thesis_configs() + [REPO_ROOT / "config" / "config.json"]:
        system = json.loads(path.read_text(encoding="utf-8")).get("system", {})
        if "inf_hall_volume_m3" in system or "inf_hall_surface_m2" in system:
            offenders.append(path.name)
    assert not offenders, f"hall overrides should be derived from geometry: {offenders}"
