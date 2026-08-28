from __future__ import annotations

import numpy as np

from factory6g.sim.config import load_config
from factory6g.sim.stages import estimators as estimator_stage

from .conftest import make_tiny_config, write_config


class _FakeEstimatorModel:
    created_instances = 0
    prepared_tokens: list[str] = []
    run_tokens: dict[str, list[str]] = {}

    def __init__(
        self,
        *,
        config: dict,
        perfect_csi: bool = False,
        estimator_type: str = "ls",
        estimator_kwargs: dict | None = None,
    ) -> None:
        del config, perfect_csi, estimator_kwargs
        self.estimator_type = estimator_type
        self.instance_index = _FakeEstimatorModel.created_instances
        _FakeEstimatorModel.created_instances += 1
        if self.instance_index > 0:
            _FakeEstimatorModel.run_tokens.setdefault(self.estimator_type, [])

    def prepare_batch_context(self, batch_size: int, ebno_db: float, include_feedback: bool):
        assert self.instance_index == 0
        token = f"ebno={float(ebno_db):.1f}|batch={len(_FakeEstimatorModel.prepared_tokens)}"
        _FakeEstimatorModel.prepared_tokens.append(token)
        return {
            "token": token,
            "batch_size": batch_size,
            "ebno_db": float(ebno_db),
            "include_feedback": include_feedback,
        }

    def run_batch(self, context, directives=None, include_details: bool = True):
        del directives, include_details
        assert self.instance_index > 0
        _FakeEstimatorModel.run_tokens[self.estimator_type].append(context["token"])
        total_bits = 64
        bit_errors = 32 if self.estimator_type == "ls" else 8
        bits = np.zeros((1, 1, 1, total_bits), dtype=np.int32)
        bits_hat = bits.copy()
        bits_hat.reshape(-1)[:bit_errors] = 1
        return {
            "bits": bits,
            "bits_hat": bits_hat,
            "runtime_latency_sec": 1e-4,
            "energy_joules": 1e-6,
        }


def test_estimator_stage_reuses_shared_batch_context_across_methods(tmp_path, monkeypatch):
    config_data = make_tiny_config(str(tmp_path / "results"))
    config_data["estimators"]["enabled"] = ["ls", "dft"]
    config_data["resource_managers"]["enabled"] = ["static"]
    config_data["monte_carlo"]["max_batches"] = 2
    config_data["monte_carlo"]["ebno_min"] = 0.0
    config_data["monte_carlo"]["ebno_max"] = 1.0
    config_data["monte_carlo"]["ebno_step"] = 1.0

    config_path = write_config(tmp_path, config_data)
    config = load_config(config_path)

    _FakeEstimatorModel.created_instances = 0
    _FakeEstimatorModel.prepared_tokens = []
    _FakeEstimatorModel.run_tokens = {}
    monkeypatch.setattr(estimator_stage, "Model", _FakeEstimatorModel)

    result = estimator_stage.run_estimator_stage(config)

    expected_tokens = [
        "ebno=0.0|batch=0",
        "ebno=1.0|batch=1",
        "ebno=0.0|batch=2",
        "ebno=1.0|batch=3",
    ]
    assert _FakeEstimatorModel.prepared_tokens == expected_tokens
    assert _FakeEstimatorModel.run_tokens["ls"] == expected_tokens
    assert _FakeEstimatorModel.run_tokens["dft"] == expected_tokens

    assert result["methods"]["ls"]["stop_reason"] == ["max_batches", "max_batches"]
    assert result["methods"]["dft"]["stop_reason"] == ["max_batches", "max_batches"]
    assert result["methods"]["ls"]["point_status"] == ["resolved", "resolved"]
    assert result["methods"]["dft"]["point_status"] == ["upper_bound_only", "upper_bound_only"]
