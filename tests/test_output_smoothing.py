from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from factory6g.sim.output import _plot_ber_publication, _plot_ber_raw


def test_publication_plot_uses_upper_bound_for_low_evidence_points(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured: list[dict[str, object]] = []
    original_semilogy = Axes.semilogy

    def _capture(self, x, y, *args, **kwargs):
        captured.append(
            {
                "x": np.asarray(x, dtype=float),
                "y": np.asarray(y, dtype=float),
                "label": kwargs.get("label"),
                "linestyle": kwargs.get("linestyle"),
                "markerfacecolor": kwargs.get("markerfacecolor"),
            }
        )
        return original_semilogy(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "semilogy", _capture)

    _plot_ber_publication(
        plt=plt,
        methods={
            "ls": {
                "ber": [1.0e-1, 0.0, 0.0],
                "ber_upper_confidence": [1.2e-1, 2.0e-4, 1.0e-4],
                "point_status": ["resolved", "upper_bound_only", "upper_bound_only"],
            }
        },
        ebno_range=[-1.0, 0.0, 1.0],
        title="test",
        output_path=tmp_path / "ber_publication.png",
    )

    assert len(captured) == 2
    np.testing.assert_allclose(captured[0]["x"], [-1.0])
    np.testing.assert_allclose(captured[0]["y"], [1.0e-1])
    assert captured[0]["label"] == "ls"

    np.testing.assert_allclose(captured[1]["x"], [0.0, 1.0])
    np.testing.assert_allclose(captured[1]["y"], [2.0e-4, 1.0e-4])
    assert captured[1]["linestyle"] == "--"
    assert captured[1]["markerfacecolor"] == "none"


def test_publication_plot_omits_experimental_pso_from_headline(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels: list[str] = []
    original_semilogy = Axes.semilogy

    def _capture(self, x, y, *args, **kwargs):
        label = kwargs.get("label")
        if isinstance(label, str):
            labels.append(label)
        return original_semilogy(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "semilogy", _capture)

    _plot_ber_publication(
        plt=plt,
        methods={
            "pso": {
                "ber": [6.0e-2] * 7,
                "ber_upper_confidence": [6.1e-2] * 7,
                "point_status": ["resolved"] * 7,
            },
            "ls": {
                "ber": [2.0e-1, 1.0e-1, 5.0e-2, 2.0e-2, 1.0e-2, 5.0e-3, 2.0e-3],
                "ber_upper_confidence": [2.1e-1, 1.1e-1, 5.1e-2, 2.1e-2, 1.1e-2, 5.1e-3, 2.1e-3],
                "point_status": ["resolved"] * 7,
            },
        },
        ebno_range=[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        title="test",
        output_path=tmp_path / "ber_publication_omit.png",
    )

    assert "ls" in labels
    assert "pso" not in labels


def test_raw_ber_plot_omits_zero_points_instead_of_floor_clipping(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured: dict[str, np.ndarray] = {}
    original_semilogy = Axes.semilogy

    def _capture(self, x, y, *args, **kwargs):
        label = kwargs.get("label")
        if isinstance(label, str):
            captured[label] = np.asarray(y, dtype=float)
        return original_semilogy(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "semilogy", _capture)

    _plot_ber_raw(
        plt=plt,
        methods={
            "a": {"ber": [0.3, 0.0, 0.1]},
            "b": {"ber": [0.4, 0.2, 0.0]},
        },
        ebno_range=[0.0, 1.0, 2.0],
        title="test",
        output_path=tmp_path / "ber_raw.png",
    )

    np.testing.assert_allclose(captured["a"][[0, 2]], [0.3, 0.1])
    assert np.isnan(captured["a"][1])
    np.testing.assert_allclose(captured["b"][:2], [0.4, 0.2])
    assert np.isnan(captured["b"][2])
