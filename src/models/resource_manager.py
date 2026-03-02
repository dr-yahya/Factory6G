from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.sim.types import ResourceManagerFeedback


@dataclass(frozen=True)
class ResourceDirectives:
    active_ut_mask: list[int] | None = None
    per_ut_power: list[float] | None = None
    pilot_reuse_factor: int | None = None


class ResourceManager:
    """Base scheduling/power-control interface used by the benchmark loop."""

    needs_channel_feedback: bool = False

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        return ResourceDirectives()


class StaticResourceManager(ResourceManager):
    def __init__(
        self,
        active_ut_mask: list[int] | None = None,
        per_ut_power: list[float] | None = None,
        pilot_reuse_factor: int | None = None,
    ) -> None:
        self._active_ut_mask = active_ut_mask
        self._per_ut_power = per_ut_power
        self._pilot_reuse_factor = pilot_reuse_factor

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        return ResourceDirectives(
            active_ut_mask=self._active_ut_mask,
            per_ut_power=self._per_ut_power,
            pilot_reuse_factor=self._pilot_reuse_factor,
        )


class RoundRobinResourceManager(ResourceManager):
    def __init__(self, num_active: int = 1) -> None:
        self.num_active = num_active
        self._current_index = 0

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = int(config.get("num_ut", 8))
        mask = [0] * num_ut
        for offset in range(self.num_active):
            mask[(self._current_index + offset) % num_ut] = 1
        self._current_index = (self._current_index + self.num_active) % num_ut
        return ResourceDirectives(active_ut_mask=mask)


def _channel_power_per_user(feedback: ResourceManagerFeedback) -> np.ndarray:
    import tensorflow as tf

    h_hat = tf.cast(feedback.h_hat, tf.complex64)
    power = tf.reduce_sum(tf.abs(h_hat) ** 2, axis=[1, 2, 4, 5, 6])
    return tf.reduce_mean(power, axis=0).numpy()


class MaxThroughputResourceManager(ResourceManager):
    def __init__(self, num_active: int = 1) -> None:
        self.needs_channel_feedback = True
        self.num_active = num_active

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = int(config.get("num_ut", 8))
        if feedback is None:
            return ResourceDirectives(active_ut_mask=[1] * num_ut)
        avg_power = _channel_power_per_user(feedback)
        top_indices = np.argsort(avg_power)[::-1][: self.num_active]
        mask = [0] * num_ut
        for idx in top_indices.tolist():
            mask[int(idx)] = 1
        return ResourceDirectives(active_ut_mask=mask)


class ProportionalFairResourceManager(ResourceManager):
    def __init__(self, num_active: int = 1, alpha: float = 0.9) -> None:
        self.needs_channel_feedback = True
        self.num_active = num_active
        self.alpha = alpha
        self.avg_rates: np.ndarray | None = None

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = int(config.get("num_ut", 8))
        if self.avg_rates is None or self.avg_rates.shape[0] != num_ut:
            self.avg_rates = np.full(num_ut, 1e-3, dtype=np.float64)
        if feedback is None:
            return ResourceDirectives(active_ut_mask=[1] * num_ut)

        avg_power = _channel_power_per_user(feedback)
        inst_rates = np.log2(1.0 + avg_power * (10.0 ** (ebno_db / 10.0)))
        pf_metric = inst_rates / self.avg_rates
        top_indices = np.argsort(pf_metric)[-self.num_active :]

        mask = [0] * num_ut
        for idx in top_indices.tolist():
            mask[int(idx)] = 1

        for idx in range(num_ut):
            if mask[idx]:
                self.avg_rates[idx] = (1.0 - self.alpha) * self.avg_rates[idx] + self.alpha * inst_rates[idx]
            else:
                self.avg_rates[idx] = (1.0 - self.alpha) * self.avg_rates[idx]
        return ResourceDirectives(active_ut_mask=mask)


def create_resource_manager(
    name: str,
    *,
    num_ut: int,
    num_active: int,
    cnn_model_path: str | None,
) -> ResourceManager:
    name_lower = name.lower()
    if "static" in name_lower:
        return StaticResourceManager(active_ut_mask=[1] * num_ut, per_ut_power=[1.0] * num_ut)
    if "round" in name_lower:
        return RoundRobinResourceManager(num_active=num_active)
    if "max" in name_lower:
        return MaxThroughputResourceManager(num_active=num_active)
    if "prop" in name_lower or "pf" in name_lower:
        return ProportionalFairResourceManager(num_active=num_active)
    if "cnn" in name_lower:
        from src.models.cnn_resource_manager import CNNResourceManager

        return CNNResourceManager(model_path=cnn_model_path)
    raise ValueError(f"Unknown resource manager: {name}")
