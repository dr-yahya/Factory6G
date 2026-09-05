"""TR 38.901 Indoor Factory (InF) large-scale propagation.

Sionna ships UMi/UMa/RMa but not the Rel-16 Indoor Factory scenarios, which are
the ones actually written for smart-factory deployments. Without them a
"factory" study is really a UMi study: the hall dimensions, machine count and
materials in `factory_scenario` never reach the propagation model, so a sweep
labelled "factory size" is only a sweep over the number of users.

This module supplies the InF large-scale model -- LOS probability, path loss and
shadow fading -- which is layered on top of a small-scale fading generator. All
formulas are from 3GPP TR 38.901 V17 section 7.4 (Table 7.4.1-1 for path loss,
Table 7.4.2-1 for LOS probability).

Sub-scenarios:
    inf_sl  Sparse clutter, low  BS (BS below the average clutter height)
    inf_dl  Dense  clutter, low  BS
    inf_sh  Sparse clutter, high BS (BS above the clutter)
    inf_dh  Dense  clutter, high BS
    inf_hh  High Tx, high Rx -- always LOS
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0

INF_SCENARIOS = ("inf_sl", "inf_dl", "inf_sh", "inf_dh", "inf_hh")


@dataclass(frozen=True)
class InFScenarioParameters:
    """Per-sub-scenario NLOS path-loss coefficients and shadow-fading sigma."""

    nlos_intercept: float
    nlos_distance_slope: float
    nlos_shadow_sigma_db: float
    high_bs: bool

    @staticmethod
    def for_scenario(scenario: str) -> "InFScenarioParameters":
        table = {
            # Table 7.4.1-1: PL = a + b*log10(d_3D) + 20*log10(f_c[GHz])
            "inf_sl": InFScenarioParameters(33.0, 25.5, 5.7, False),
            "inf_dl": InFScenarioParameters(18.6, 35.7, 7.2, False),
            "inf_sh": InFScenarioParameters(32.4, 23.0, 5.9, True),
            "inf_dh": InFScenarioParameters(33.63, 21.9, 4.0, True),
            # InF-HH is always LOS; the NLOS row is unused but kept coherent.
            "inf_hh": InFScenarioParameters(32.4, 23.0, 5.9, True),
        }
        key = scenario.lower()
        if key not in table:
            raise ValueError(
                f"Unknown InF scenario '{scenario}'. Valid: {sorted(table)}."
            )
        return table[key]


# LOS path loss is common to all sub-scenarios (Table 7.4.1-1).
_LOS_INTERCEPT = 31.84
_LOS_DISTANCE_SLOPE = 21.50
_LOS_FREQUENCY_SLOPE = 19.00
_LOS_SHADOW_SIGMA_DB = 4.3


def inf_los_probability(
    distance_2d_m: np.ndarray,
    *,
    scenario: str,
    clutter_density: float,
    clutter_size_m: float,
    bs_height_m: float,
    ut_height_m: float,
    clutter_height_m: float,
) -> np.ndarray:
    """LOS probability for an InF sub-scenario (TR 38.901 Table 7.4.2-1).

    The clutter density is the fraction of the floor occupied by machinery, which
    is exactly what `factory_scenario.num_machines` and the machine size range
    describe -- so hall layout finally influences propagation.
    """
    key = scenario.lower()
    if key == "inf_hh":
        return np.ones_like(np.asarray(distance_2d_m, dtype=np.float64))

    density = float(np.clip(clutter_density, 1e-6, 0.99))
    k = -float(clutter_size_m) / math.log(1.0 - density)

    if key in ("inf_sh", "inf_dh"):
        # High BS: the subscenario constant is stretched by the height ratio.
        denominator = float(clutter_height_m) - float(ut_height_m)
        if abs(denominator) < 1e-9:
            raise ValueError(
                "InF high-BS LOS probability is undefined when the clutter height "
                "equals the UT height."
            )
        k *= (float(bs_height_m) - float(ut_height_m)) / denominator

    k = max(k, 1e-6)
    return np.exp(-np.asarray(distance_2d_m, dtype=np.float64) / k)


def inf_path_loss_db(
    distance_3d_m: np.ndarray,
    *,
    scenario: str,
    carrier_frequency_hz: float,
    is_los: np.ndarray,
) -> np.ndarray:
    """InF path loss in dB (TR 38.901 Table 7.4.1-1).

    The NLOS value is floored by the LOS value (and, for InF-DL, additionally by
    the InF-SL value) exactly as the table requires.
    """
    params = InFScenarioParameters.for_scenario(scenario)
    distance = np.maximum(np.asarray(distance_3d_m, dtype=np.float64), 1.0)
    frequency_ghz = float(carrier_frequency_hz) / 1e9
    log_distance = np.log10(distance)
    log_frequency = math.log10(frequency_ghz)

    pl_los = (
        _LOS_INTERCEPT
        + _LOS_DISTANCE_SLOPE * log_distance
        + _LOS_FREQUENCY_SLOPE * log_frequency
    )
    pl_nlos = (
        params.nlos_intercept
        + params.nlos_distance_slope * log_distance
        + 20.0 * log_frequency
    )
    pl_nlos = np.maximum(pl_nlos, pl_los)
    if scenario.lower() == "inf_dl":
        sparse = InFScenarioParameters.for_scenario("inf_sl")
        pl_sl = (
            sparse.nlos_intercept
            + sparse.nlos_distance_slope * log_distance
            + 20.0 * log_frequency
        )
        pl_nlos = np.maximum(pl_nlos, pl_sl)

    return np.where(np.asarray(is_los, dtype=bool), pl_los, pl_nlos)


def inf_shadow_sigma_db(scenario: str, is_los: np.ndarray) -> np.ndarray:
    """Per-link shadow-fading standard deviation in dB."""
    params = InFScenarioParameters.for_scenario(scenario)
    return np.where(
        np.asarray(is_los, dtype=bool),
        _LOS_SHADOW_SIGMA_DB,
        params.nlos_shadow_sigma_db,
    )


def inf_delay_spread_seconds(
    hall_volume_m3: float,
    hall_surface_m2: float,
    *,
    rng: np.random.Generator | None = None,
    num_links: int = 1,
) -> np.ndarray:
    """RMS delay spread for an InF hall (TR 38.901 Table 7.5-6, Part 3).

        mu_lgDS    = log10(26 * (V / S) + 14) - 9.35
        sigma_lgDS = 0.15

    V is the hall volume and S its total interior surface area, so the delay
    spread is set by the room's geometry -- a bigger hall reverberates longer.
    This is the mechanism that makes a "factory size" sweep physically
    meaningful rather than a relabelled user-count sweep.
    """
    volume = max(float(hall_volume_m3), 1.0)
    surface = max(float(hall_surface_m2), 1.0)
    mu = math.log10(26.0 * (volume / surface) + 14.0) - 9.35
    if rng is None:
        return np.full(num_links, 10.0**mu)
    return np.power(10.0, rng.normal(mu, 0.15, num_links))


def hall_volume_and_surface(room_dimensions: list[float]) -> tuple[float, float]:
    """Interior volume and total surface area of a rectangular hall."""
    length, width, height = (float(v) for v in room_dimensions[:3])
    volume = length * width * height
    surface = 2.0 * length * width + 2.0 * height * (length + width)
    return volume, surface


def exponential_pdp(
    delay_spread_sec: float,
    num_taps: int,
    sample_duration_sec: float,
) -> np.ndarray:
    """Sampled exponential power delay profile with the requested RMS spread.

    An exponential PDP is the standard indoor model. The decay constant is set
    so the *sampled* profile has the target RMS delay spread, and the result is
    normalised to unit total power.
    """
    if delay_spread_sec <= 0.0 or sample_duration_sec <= 0.0:
        profile = np.zeros(max(num_taps, 1))
        profile[0] = 1.0
        return profile

    taps = np.arange(max(num_taps, 1)) * sample_duration_sec
    # For a continuous exponential profile the RMS spread equals the decay
    # constant, which is the right starting point for the sampled version.
    profile = np.exp(-taps / delay_spread_sec)
    total = profile.sum()
    if total <= 0.0:
        profile = np.zeros_like(profile)
        profile[0] = 1.0
        return profile
    return profile / total


def coherence_bandwidth_hz(delay_spread_sec: float) -> float:
    """Coherence bandwidth at 0.5 correlation, the usual 1 / (5 * DS) rule."""
    if delay_spread_sec <= 0.0:
        return float("inf")
    return 1.0 / (5.0 * delay_spread_sec)


def clutter_density_from_layout(
    num_machines: int,
    machine_size_range: list[list[float]],
    room_dimensions: list[float],
) -> float:
    """Fraction of the hall floor covered by machinery.

    Uses the mean machine footprint from the configured size range, so the
    factory geometry in `factory_scenario` maps onto the InF clutter density the
    LOS probability depends on.
    """
    if not room_dimensions or len(room_dimensions) < 2:
        return 0.3
    floor_area = float(room_dimensions[0]) * float(room_dimensions[1])
    if floor_area <= 0.0:
        return 0.3
    mean_x = float(np.mean(machine_size_range[0])) if machine_size_range else 1.0
    mean_y = float(np.mean(machine_size_range[1])) if len(machine_size_range) > 1 else 1.0
    covered = float(num_machines) * mean_x * mean_y
    return float(np.clip(covered / floor_area, 1e-3, 0.95))


def mean_clutter_size_m(machine_size_range: list[list[float]]) -> float:
    """Characteristic clutter size, the `d_clutter` of Table 7.4.2-1."""
    if not machine_size_range:
        return 2.0
    mean_x = float(np.mean(machine_size_range[0]))
    mean_y = float(np.mean(machine_size_range[1])) if len(machine_size_range) > 1 else mean_x
    return max(math.sqrt(max(mean_x * mean_y, 1e-6)), 0.1)


def sample_inf_large_scale_gain(
    *,
    num_links: int,
    scenario: str,
    carrier_frequency_hz: float,
    room_dimensions: list[float],
    bs_height_m: float,
    ut_height_m: float,
    clutter_density: float,
    clutter_size_m: float,
    clutter_height_m: float,
    enable_pathloss: bool,
    enable_shadow_fading: bool,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw UT positions in the hall and return per-link large-scale gains.

    Returns linear amplitude gains (not powers) plus the diagnostics a study
    wants to report: LOS flags, 2D/3D distances and the path loss in dB.
    """
    length, width = float(room_dimensions[0]), float(room_dimensions[1])
    # BS at the hall centre, UTs uniform over the floor.
    ut_xy = np.stack(
        [rng.uniform(0.0, length, num_links), rng.uniform(0.0, width, num_links)],
        axis=-1,
    )
    bs_xy = np.array([length / 2.0, width / 2.0])
    distance_2d = np.linalg.norm(ut_xy - bs_xy, axis=-1)
    height_difference = float(bs_height_m) - float(ut_height_m)
    distance_3d = np.sqrt(distance_2d**2 + height_difference**2)

    los_probability = inf_los_probability(
        distance_2d,
        scenario=scenario,
        clutter_density=clutter_density,
        clutter_size_m=clutter_size_m,
        bs_height_m=bs_height_m,
        ut_height_m=ut_height_m,
        clutter_height_m=clutter_height_m,
    )
    is_los = rng.uniform(0.0, 1.0, num_links) < los_probability

    path_loss_db = inf_path_loss_db(
        distance_3d,
        scenario=scenario,
        carrier_frequency_hz=carrier_frequency_hz,
        is_los=is_los,
    )
    total_loss_db = np.zeros(num_links) if not enable_pathloss else path_loss_db.copy()
    if enable_shadow_fading:
        sigma = inf_shadow_sigma_db(scenario, is_los)
        total_loss_db = total_loss_db + rng.normal(0.0, sigma, num_links)

    # Normalise to unit *mean power* across links. The Eb/No sweep already sets
    # the operating point, so absolute path loss would only shift the x-axis;
    # what matters is the relative spread across users, which is what a scheduler
    # exploits and which this preserves exactly.
    amplitude_gain = 10.0 ** (-total_loss_db / 20.0)
    mean_power = float(np.mean(amplitude_gain**2))
    if mean_power > 0.0:
        amplitude_gain = amplitude_gain / math.sqrt(mean_power)

    return {
        "amplitude_gain": amplitude_gain,
        "is_los": is_los,
        "distance_2d_m": distance_2d,
        "distance_3d_m": distance_3d,
        "path_loss_db": path_loss_db,
        "los_probability": los_probability,
    }
