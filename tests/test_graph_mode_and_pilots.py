"""Tests for graph-mode execution, estimator complexity and pilot reuse."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("sionna")

from factory6g.components.estimators.complexity import (  # noqa: E402
    adaptive_complexity,
    dft_complexity,
    estimator_complexity,
    lmmse_complexity,
    ls_complexity,
)
from factory6g.models.model import Model  # noqa: E402
from factory6g.models.resource_manager import ResourceDirectives  # noqa: E402

_CONFIG = {
    "num_ut": 4,
    "num_bs_ant": 8,
    "fft_size": 32,
    "num_ofdm_symbols": 14,
    "channel_model_type": "rayleigh",
    "num_bits_per_symbol": 2,
    "coderate": 0.5,
    "num_decoding_iter": 6,
    "pilot_ofdm_symbol_indices": [2, 11],
}


class TestGraphMode:
    """Nothing in the simulation path used to be graph-compiled (review 4.1)."""

    @pytest.mark.parametrize("estimator", ["ls", "dft", "lmmse", "adaptive"])
    def test_graph_mode_matches_eager_bit_for_bit(self, estimator):
        eager = Model(config={**_CONFIG, "graph_mode": False}, estimator_type=estimator)
        graph = Model(config={**_CONFIG, "graph_mode": True}, estimator_type=estimator)
        # One shared context, so any difference is execution mode alone.
        context = eager.prepare_batch_context(
            batch_size=2, ebno_db=6.0, include_feedback=False
        )
        eager_result = eager.run_batch(context, include_details=True)
        graph_result = graph.run_batch(context, include_details=True)
        np.testing.assert_array_equal(eager_result["bits_hat"], graph_result["bits_hat"])

    def test_graph_mode_is_off_by_default(self):
        assert Model(config=_CONFIG, estimator_type="ls").graph_mode is False

    def test_compiled_function_is_built_once_and_reused(self):
        model = Model(config={**_CONFIG, "graph_mode": True}, estimator_type="ls")
        assert model._compiled_decode() is model._compiled_decode()


class TestEstimatorComplexity:
    """Complexity claims need a multiplication count, not a stopwatch."""

    def test_ordering_matches_the_algorithms(self):
        n, p = 128, 2
        assert ls_complexity(n, p) < dft_complexity(n, p) < lmmse_complexity(n, p)

    def test_eigendecomposition_beats_a_direct_inverse(self):
        n, p = 128, 2
        assert lmmse_complexity(n, p, use_eigen=True) < lmmse_complexity(
            n, p, use_eigen=False
        )

    def test_scalar_adaptive_interpolates_between_its_branches(self):
        n, p = 128, 2
        all_dft = adaptive_complexity(n, p, lmmse_fraction=0.0, selection_mode="scalar")
        all_lmmse = adaptive_complexity(n, p, lmmse_fraction=1.0, selection_mode="scalar")
        half = adaptive_complexity(n, p, lmmse_fraction=0.5, selection_mode="scalar")
        assert all_dft == pytest.approx(dft_complexity(n, p))
        assert all_lmmse == pytest.approx(lmmse_complexity(n, p))
        assert all_dft < half < all_lmmse

    def test_per_user_mode_costs_both_branches(self):
        n, p = 128, 2
        per_user = adaptive_complexity(n, p, lmmse_fraction=0.5, selection_mode="per_user")
        assert per_user > lmmse_complexity(n, p)

    def test_complexity_grows_with_fft_size(self):
        assert estimator_complexity("lmmse", 256, 2) > estimator_complexity("lmmse", 128, 2)

    def test_unknown_estimator_is_rejected(self):
        with pytest.raises(ValueError):
            estimator_complexity("not_an_estimator", 128, 2)


class TestPilotReuse:
    """`pilot_reuse_factor` was set by three managers and consumed by none."""

    @staticmethod
    def _pilot_and_data_power(reuse_factor: int):
        model = Model(config=_CONFIG, estimator_type="ls")
        directives = ResourceDirectives(
            active_ut_mask=[1] * 4,
            per_ut_power=[1.0] * 4,
            pilot_reuse_factor=reuse_factor,
        )
        x_rg, _, _ = model.get_transmitter().call(2, directives=directives)
        grid = x_rg.numpy()
        pilot = float(np.mean(np.abs(grid[:, :, :, [2, 11], :]) ** 2))
        data = float(np.mean(np.abs(grid[:, :, :, [0, 1, 3], :]) ** 2))
        return pilot, data

    def test_reuse_factor_one_is_a_no_op(self):
        pilot, data = self._pilot_and_data_power(1)
        assert pilot == pytest.approx(data)

    def test_reuse_shares_pilot_power_within_the_group(self):
        pilot_1, _ = self._pilot_and_data_power(1)
        pilot_2, _ = self._pilot_and_data_power(2)
        pilot_4, _ = self._pilot_and_data_power(4)
        assert pilot_2 == pytest.approx(pilot_1 / 2, rel=1e-5)
        assert pilot_4 == pytest.approx(pilot_1 / 4, rel=1e-5)

    def test_reuse_leaves_data_symbols_untouched(self):
        _, data_1 = self._pilot_and_data_power(1)
        _, data_4 = self._pilot_and_data_power(4)
        assert data_4 == pytest.approx(data_1, rel=1e-6)

    def test_pilot_reuse_degrades_channel_estimation(self):
        """The contamination penalty must actually show up in the estimate."""
        model = Model(config=_CONFIG, estimator_type="ls")
        context = model.prepare_batch_context(
            batch_size=4, ebno_db=0.0, include_feedback=False
        )

        def nmse(reuse_factor):
            directives = ResourceDirectives(
                active_ut_mask=[1] * 4,
                per_ut_power=[1.0] * 4,
                pilot_reuse_factor=reuse_factor,
            )
            result = model.run_batch(context, directives=directives, include_details=True)
            error = np.mean(np.abs(result["channel"] - result["channel_hat"]) ** 2)
            return error / np.mean(np.abs(result["channel"]) ** 2)

        assert nmse(4) > nmse(1)
