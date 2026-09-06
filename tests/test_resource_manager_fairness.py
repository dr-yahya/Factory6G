from __future__ import annotations

import numpy as np
import pytest

from factory6g.models.model import Model
from factory6g.models.resource_manager import StaticResourceManager
from factory6g.models.model import Model
from factory6g.sim.config import load_config
from factory6g.sim.flow import run_simulation_flow

from .conftest import make_tiny_config, write_config


@pytest.mark.slow
def test_rm_loop_prepares_one_shared_context_per_point(monkeypatch, tmp_path):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["resource_managers"]["enabled"] = ["static", "max_throughput"]
    config_path = write_config(tmp_path, config_data)
    config = load_config(config_path)

    prepare_calls = 0
    context_h_ids: list[int] = []
    context_noise_ids: list[int] = []
    feedback_ids: list[int] = []

    original_prepare = Model.prepare_batch_context
    original_run_batch = Model.run_batch

    def counted_prepare(self, *args, **kwargs):
        nonlocal prepare_calls
        include_feedback = kwargs.get("include_feedback")
        if include_feedback is None and len(args) >= 3:
            include_feedback = args[2]
        if include_feedback:
            prepare_calls += 1
        return original_prepare(self, *args, **kwargs)

    def recording_run_batch(self, batch_context, *args, **kwargs):
        if batch_context.feedback is not None:
            context_h_ids.append(id(batch_context.h_freq))
            context_noise_ids.append(id(batch_context.data_noise))
            feedback_ids.append(id(batch_context.feedback.h_hat))
        return original_run_batch(self, batch_context, *args, **kwargs)

    monkeypatch.setattr(Model, "prepare_batch_context", counted_prepare)
    monkeypatch.setattr(Model, "run_batch", recording_run_batch)

    run_simulation_flow(config)

    assert prepare_calls == 1
    assert len(set(context_h_ids)) == 1
    assert len(set(context_noise_ids)) == 1
    assert len(set(feedback_ids)) == 1


def test_repeating_same_manager_and_context_is_bit_identical():
    config_data = make_tiny_config("results")
    runtime_config = config_data["system"] | config_data["transceiver"]
    model = Model(config=runtime_config, estimator_type="lmmse", perfect_csi=False)
    context = model.prepare_batch_context(batch_size=1, ebno_db=0.0, include_feedback=True)
    manager = StaticResourceManager(active_ut_mask=[1, 1], per_ut_power=[1.0, 1.0])
    directives = manager.get_runtime_directives(runtime_config, 0.0, feedback=context.feedback)

    result_one = model.run_batch(context, directives=directives, include_details=False)
    result_two = model.run_batch(context, directives=directives, include_details=False)

    assert np.array_equal(result_one["bits"], result_two["bits"])
    assert np.array_equal(result_one["bits_hat"], result_two["bits_hat"])
