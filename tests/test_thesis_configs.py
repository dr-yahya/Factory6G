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
    # Only paths actually passed to --config are run configs. The docs also name
    # data files that live under config/ -- factory_size_profiles.json is a table
    # of scenario presets, not something load_config can read -- so requiring
    # every mentioned path to parse as a config fails on files that were never
    # meant to.
    runnable = re.compile(r"--config[= ]+(config/[A-Za-z0-9_/]+\.json)")
    referenced: dict[str, set[str]] = {}
    loadable: set[str] = set()
    doc_files = list((REPO_ROOT / "docs").glob("*.md")) + [REPO_ROOT / "README.md"]
    for doc in doc_files:
        text = doc.read_text(encoding="utf-8")
        for match in pattern.findall(text):
            referenced.setdefault(match, set()).add(doc.name)
        loadable.update(runnable.findall(text))

    missing = {
        path: sorted(sources)
        for path, sources in referenced.items()
        if not (REPO_ROOT / path).is_file()
    }
    assert not missing, f"documented config paths that do not exist: {missing}"

    # Every family config must parse, however the docs happen to spell its path
    # -- THESIS_RESULTS.md writes the reproduction command with a `<family>`
    # placeholder, so matching on --config alone would check almost nothing.
    family_configs = sorted((REPO_ROOT / "config" / "thesis").glob("*.json"))
    assert family_configs, "config/thesis/ is empty -- the documented families are gone"
    for path in family_configs:
        load_config(path)
    for path in sorted(loadable):
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


def test_factory_size_tables_stay_in_sync():
    """One preset table, not three.

    The presets, the short display names and the long descriptions used to live
    in three places -- `sim/factory_profiles.py`, private aliases in `sim/flow.py`,
    and a separate hard-coded dict in `cli/visualize.py` that had drifted to
    different strings. They are now one module, and must agree.
    """
    from factory6g.sim.factory_profiles import (
        FACTORY_SIZE_DESCRIPTIONS,
        FACTORY_SIZE_DISPLAY,
        FACTORY_SIZE_PRESETS,
    )

    assert set(FACTORY_SIZE_PRESETS) == set(FACTORY_SIZE_DISPLAY)
    assert set(FACTORY_SIZE_PRESETS) == set(FACTORY_SIZE_DESCRIPTIONS)
    for key, preset in FACTORY_SIZE_PRESETS.items():
        assert len(preset["room_dimensions"]) == 3, key
        assert preset["num_ut"] > 0, key
        # Kronecker pilots require the FFT size to divide by the user count.
        assert 128 % preset["num_ut"] == 0, key


def test_raw_config_loader_handles_comments(tmp_path):
    """Configs may carry `//` comments, and the raw loader must survive them.

    `cli.run` used to read the file a second time with a plain `json.load`
    wrapped in `except Exception: raw_config = {}`. A commented config parsed
    fine through `load_config` and then silently produced an empty raw config,
    dropping every section `Factory6GConfig` does not model -- the whole
    `jidd_scma` block among them.
    """
    from factory6g.sim.config import load_raw_config

    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["jidd_scma"] = {"polar_N": 256}
    config_path = write_config(tmp_path, config_data)
    commented = "// leading comment\n" + config_path.read_text(encoding="utf-8")
    config_path.write_text(commented, encoding="utf-8")

    raw = load_raw_config(config_path)
    assert raw["jidd_scma"]["polar_N"] == 256
    load_config(config_path)
