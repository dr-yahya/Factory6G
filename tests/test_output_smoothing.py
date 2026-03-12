from __future__ import annotations

import numpy as np

from src.sim.output import _smooth_ber_curve, _weighted_isotonic


def test_weighted_isotonic_enforces_monotonic_decreasing():
    y = np.array([0.12, 0.10, 0.11, 0.06, 0.07, 0.03], dtype=float)
    w = np.array([10, 10, 10, 10, 10, 10], dtype=float)
    fitted = _weighted_isotonic(y, w, increasing=False)
    assert np.all(np.diff(fitted) <= 1e-12)


def test_ber_smoothing_uses_jeffreys_without_zero_collapse():
    metric_map = {
        "bit_errors": [40, 12, 2, 0, 0, 0],
        "total_bits": [1000, 1000, 1000, 1000, 1000, 1000],
    }
    raw = np.array([0.04, 0.012, 0.002, 0.0, 0.0, 0.0], dtype=float)
    smooth = _smooth_ber_curve(metric_map=metric_map, raw_values=raw)
    assert np.all(smooth > 0.0)
    assert np.all(np.diff(np.log10(smooth)) <= 1e-12)
