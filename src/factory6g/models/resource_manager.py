from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from factory6g.sim.types import ResourceManagerFeedback


_EPS = 1e-9


def _resolve_model_path(model_path: str | Path, model_root: str | Path | None) -> Path:
    """Resolve a policy path against an explicit root rather than the cwd.

    Learned-manager paths in config are relative (``models/...``); resolving them
    against the process working directory meant whether the policy loaded
    depended on where the run was launched from.
    """
    path = Path(model_path)
    if path.is_absolute() or model_root is None:
        return path
    rooted = Path(model_root) / path
    return rooted if rooted.exists() else path


def _checkpoint_digest(path: Path) -> str | None:
    """SHA-256 over a checkpoint file, or over a directory's file listing."""
    try:
        digest = hashlib.sha256()
        if path.is_file():
            digest.update(path.read_bytes())
        elif path.is_dir():
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                digest.update(child.name.encode())
                digest.update(child.read_bytes())
        else:
            return None
        return digest.hexdigest()
    except OSError:
        return None


@dataclass(frozen=True)
class ResourceDirectives:
    active_ut_mask: list[int] | None = None
    per_ut_power: list[float] | None = None
    pilot_reuse_factor: int | None = None


class ResourceManager:
    """Base scheduling/power-control interface used by the benchmark loop.

    Scheduler state (round-robin pointers, proportional-fair rate averages,
    virtual queues, ...) MUST be isolated per Eb/No point. The benchmark loop
    visits every Eb/No point inside every Monte Carlo batch, so a single shared
    state object would be driven by the sweep order rather than by the time
    evolution of one link -- a manager's fairness memory at +20 dB would be
    polluted by what it saw at 0 dB. Subclasses therefore keep state in a
    ``_PerPointState`` keyed by Eb/No, and expose it through
    ``export_state``/``load_state`` so a resumed run continues from where it
    left off instead of silently restarting cold.
    """

    needs_channel_feedback: bool = False

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        return ResourceDirectives()

    def export_state(self) -> dict[str, Any]:
        """JSON-serialisable scheduler state, for checkpointing."""
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore scheduler state produced by :meth:`export_state`."""
        return None


class _PerPointState:
    """Scheduler state isolated per Eb/No point.

    Keys are rounded Eb/No values so floating-point noise in the sweep does not
    fragment the state.
    """

    def __init__(self, default_factory) -> None:
        self._default_factory = default_factory
        self._by_point: dict[float, Any] = {}

    def get(self, ebno_db: float, num_ut: int):
        key = _state_key(ebno_db)
        value = self._by_point.get(key)
        if value is None or (hasattr(value, "shape") and value.shape[0] != num_ut):
            value = self._default_factory(num_ut)
            self._by_point[key] = value
        return value

    def set(self, ebno_db: float, value) -> None:
        self._by_point[_state_key(ebno_db)] = value

    def export(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self._by_point.items():
            out[str(key)] = value.tolist() if hasattr(value, "tolist") else value
        return out

    def load(self, state: dict[str, Any], *, as_array: bool = True) -> None:
        self._by_point = {}
        for key, value in (state or {}).items():
            self._by_point[float(key)] = (
                np.asarray(value, dtype=np.float64) if as_array and isinstance(value, list) else value
            )


class StaticResourceManager(ResourceManager):
    """Fixed allocation with no channel awareness.

    Two roles, and it matters which one a comparison uses:

    * ``num_active=None`` (default) schedules every user at full power. This is
      the *full-load* reference -- "what happens with no scheduler at all".
    * ``num_active=k`` schedules the first k users. This is the
      *equal-load* control for comparing against the k-user schedulers; without
      it, "static vs DRL" compares a full-load operating point against a k-user
      one, and the resulting BER gap is a change of load, not a scheduling
      result.
    """

    def __init__(
        self,
        active_ut_mask: list[int] | None = None,
        per_ut_power: list[float] | None = None,
        pilot_reuse_factor: int | None = None,
        num_active: int | None = None,
        num_ut: int | None = None,
    ) -> None:
        if active_ut_mask is None and num_active is not None and num_ut is not None:
            active_ut_mask = _mask_from_indices(range(min(int(num_active), int(num_ut))), int(num_ut))
            per_ut_power = [1.0 if flag else 0.0 for flag in active_ut_mask]
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
        self.num_active = max(1, int(num_active))
        # The rotation pointer is per Eb/No point: a single shared pointer would
        # make the schedule depend on the sweep order and on which other methods
        # are still converging, which is not reproducible from the config.
        self._index = _PerPointState(lambda num_ut: 0)

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = _num_ut(config)
        current = int(self._index.get(ebno_db, num_ut))
        mask = [0] * num_ut
        for offset in range(min(self.num_active, num_ut)):
            mask[(current + offset) % num_ut] = 1
        self._index.set(ebno_db, (current + self.num_active) % num_ut)
        return ResourceDirectives(active_ut_mask=mask)

    def export_state(self) -> dict[str, Any]:
        return {"index": self._index.export()}

    def load_state(self, state: dict[str, Any]) -> None:
        self._index.load(state.get("index", {}), as_array=False)


def _feedback_h_hat_numpy(feedback: ResourceManagerFeedback) -> np.ndarray:
    h_hat = feedback.h_hat
    if hasattr(h_hat, "numpy"):
        h_hat = h_hat.numpy()
    return np.asarray(h_hat, dtype=np.complex64)


def _channel_power_per_user(feedback: ResourceManagerFeedback) -> np.ndarray:
    h_hat = _feedback_h_hat_numpy(feedback)
    power = np.abs(h_hat) ** 2
    reduced = power.sum(axis=(1, 2, 4, 5, 6))
    return reduced.mean(axis=0).astype(np.float64)


def _esno_linear(ebno_db: float, config: dict[str, Any] | None) -> float:
    """Convert Eb/N0 in dB to a linear Es/N0.

    Es/N0 = Eb/N0 * coderate * bits_per_symbol. Skipping this conversion (as an
    earlier revision did) leaves the metric off by that factor, which is harmless
    for pure ranking but wrong anywhere the value is read as a rate -- the
    queue-aware manager uses it as a served-bits proxy.
    """
    ebno_linear = 10.0 ** (float(ebno_db) / 10.0)
    if config is None:
        return max(ebno_linear, _EPS)
    coderate = float(config.get("coderate", 0.5))
    bits_per_symbol = float(config.get("num_bits_per_symbol", 2))
    return max(ebno_linear * coderate * bits_per_symbol, _EPS)


def _instantaneous_rate_per_user(
    feedback: ResourceManagerFeedback,
    ebno_db: float,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    avg_power = _channel_power_per_user(feedback)
    return np.log2(1.0 + avg_power * _esno_linear(ebno_db, config))


def _normalized_rate_metric(
    feedback: ResourceManagerFeedback,
    ebno_db: float,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    inst_rates = _instantaneous_rate_per_user(feedback, ebno_db, config=config)
    scale = max(float(np.max(inst_rates, initial=0.0)), _EPS)
    return inst_rates / scale


def _effective_gain_matrix(feedback: ResourceManagerFeedback) -> np.ndarray:
    h_hat = _feedback_h_hat_numpy(feedback)
    if h_hat.ndim != 7:
        raise ValueError(f"Expected h_hat with 7 dimensions, got shape {h_hat.shape}.")

    mean_h = np.mean(h_hat, axis=0)
    user_first = np.moveaxis(mean_h, 2, 0)
    vectors = user_first.reshape(user_first.shape[0], -1)
    gram = vectors @ vectors.conj().T

    num_ut = gram.shape[0]
    direct = np.maximum(np.real(np.diag(gram)), _EPS)
    gains = np.zeros((num_ut, num_ut), dtype=np.float64)
    diag_indices = np.diag_indices(num_ut)
    gains[diag_indices] = direct
    for rx_idx in range(num_ut):
        denom = max(direct[rx_idx], _EPS)
        for tx_idx in range(num_ut):
            if rx_idx == tx_idx:
                continue
            gains[rx_idx, tx_idx] = float((abs(gram[rx_idx, tx_idx]) ** 2) / denom)

    return gains / max(float(np.mean(direct)), _EPS)


def _num_ut(config: dict[str, Any]) -> int:
    return max(1, int(config.get("num_ut", 8)))


def _mask_from_indices(indices: Sequence[int], num_ut: int) -> list[int]:
    mask = [0] * num_ut
    for idx in indices:
        if 0 <= int(idx) < num_ut:
            mask[int(idx)] = 1
    return mask


def _top_indices(metric: np.ndarray, count: int) -> np.ndarray:
    if metric.size == 0:
        return np.array([], dtype=np.int64)
    active_count = max(1, min(int(count), metric.size))
    return np.argsort(metric)[::-1][:active_count]


def _normalize_power(
    power: np.ndarray,
    mask: Sequence[int],
    *,
    max_power: float = 1.0,
    min_active_power: float = 0.15,
    mode: str = "peak",
    sum_power_budget: float | None = None,
) -> list[float]:
    """Project a raw power preference onto the transmit power constraint.

    Modes:
        ``peak``      Rescale so the strongest scheduled user sits at
                      ``max_power``. Scale-free: appropriate for heuristics that
                      only express *relative* preference.
        ``absolute``  Clip to [0, max_power] without rescaling, so a manager that
                      solved for physically meaningful levels (WMMSE) keeps them.

    ``per_ut_power`` is a fraction of each UT's own maximum. That is the correct
    constraint for the uplink direction this project simulates -- every device
    has its own power amplifier, and there is no shared budget across devices.
    ``sum_power_budget`` is available for downlink or shared-budget studies: when
    set, the scheduled powers are additionally scaled so they sum to it.
    """
    power_arr = np.clip(np.asarray(power, dtype=np.float64), 0.0, None)
    mask_arr = np.asarray(mask, dtype=bool)
    out = np.zeros_like(power_arr, dtype=np.float64)
    if not np.any(mask_arr):
        return out.tolist()

    active_power = power_arr[mask_arr]
    if mode == "absolute":
        normalized = np.clip(active_power, min_active_power, max_power)
    elif mode == "peak":
        peak = max(float(np.max(active_power, initial=0.0)), _EPS)
        normalized = np.clip((active_power / peak) * max_power, min_active_power, max_power)
    else:
        raise ValueError(f"Unknown power normalization mode '{mode}'.")

    if sum_power_budget is not None:
        total = max(float(np.sum(normalized)), _EPS)
        normalized = np.clip(normalized * (float(sum_power_budget) / total), 0.0, max_power)

    out[mask_arr] = normalized
    return out.tolist()


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-3)
    shifted = np.asarray(values, dtype=np.float64) / temp
    shifted = shifted - float(np.max(shifted, initial=0.0))
    exp_values = np.exp(shifted)
    return exp_values / max(float(np.sum(exp_values)), _EPS)


def _state_key(ebno_db: float) -> float:
    return round(float(ebno_db), 6)


class MaxThroughputResourceManager(ResourceManager):
    def __init__(self, num_active: int = 1) -> None:
        self.needs_channel_feedback = True
        self.num_active = max(1, int(num_active))

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = _num_ut(config)
        if feedback is None:
            return ResourceDirectives(active_ut_mask=[1] * num_ut)
        avg_power = _channel_power_per_user(feedback)
        top_indices = _top_indices(avg_power, self.num_active)
        mask = _mask_from_indices(top_indices.tolist(), num_ut)
        return ResourceDirectives(active_ut_mask=mask)


class ProportionalFairResourceManager(ResourceManager):
    def __init__(self, num_active: int = 1, alpha: float = 0.9) -> None:
        self.needs_channel_feedback = True
        self.num_active = max(1, int(num_active))
        self.alpha = float(alpha)
        # Per Eb/No point: the fairness memory must not carry across the sweep.
        self._avg_rates = _PerPointState(
            lambda num_ut: np.full(num_ut, 1e-3, dtype=np.float64)
        )

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = _num_ut(config)
        avg_rates = np.array(self._avg_rates.get(ebno_db, num_ut), dtype=np.float64)
        if feedback is None:
            return ResourceDirectives(active_ut_mask=[1] * num_ut)

        inst_rates = _instantaneous_rate_per_user(feedback, ebno_db, config=config)
        pf_metric = inst_rates / np.maximum(avg_rates, _EPS)
        top_indices = _top_indices(pf_metric, self.num_active)
        mask = _mask_from_indices(top_indices.tolist(), num_ut)

        served = np.asarray(mask, dtype=np.float64) * inst_rates
        avg_rates = (1.0 - self.alpha) * avg_rates + self.alpha * served
        self._avg_rates.set(ebno_db, avg_rates)
        return ResourceDirectives(active_ut_mask=mask)

    def export_state(self) -> dict[str, Any]:
        return {"avg_rates": self._avg_rates.export()}

    def load_state(self, state: dict[str, Any]) -> None:
        self._avg_rates.load(state.get("avg_rates", {}))


class WMMSEResourceManager(ResourceManager):
    """Scalar effective-channel WMMSE scheduler/power allocator."""

    def __init__(
        self,
        num_active: int = 1,
        iterations: int = 12,
        damping: float = 0.5,
        max_power: float = 1.0,
        min_active_power: float = 0.15,
        user_weights: Sequence[float] | None = None,
    ) -> None:
        self.needs_channel_feedback = True
        self.num_active = max(1, int(num_active))
        self.iterations = max(1, int(iterations))
        self.damping = float(np.clip(damping, 0.0, 0.99))
        self.max_power = float(max_power)
        self.min_active_power = float(min_active_power)
        self.user_weights = None if user_weights is None else np.asarray(user_weights, dtype=np.float64)

    def _weights(self, num_ut: int) -> np.ndarray:
        if self.user_weights is None:
            return np.ones(num_ut, dtype=np.float64)
        if self.user_weights.shape[0] != num_ut:
            raise ValueError(
                f"WMMSE user_weights length {self.user_weights.shape[0]} does not match num_ut={num_ut}."
            )
        return np.maximum(self.user_weights, _EPS)

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = _num_ut(config)
        if feedback is None:
            mask = _mask_from_indices(range(min(self.num_active, num_ut)), num_ut)
            power = _normalize_power(np.ones(num_ut, dtype=np.float64), mask, max_power=self.max_power)
            return ResourceDirectives(active_ut_mask=mask, per_ut_power=power, pilot_reuse_factor=1)

        gains = _effective_gain_matrix(feedback)
        weights = self._weights(num_ut)
        diag_gain = np.maximum(np.diag(gains), _EPS)
        noise = 1.0 / _esno_linear(ebno_db, config)

        v = np.full(num_ut, np.sqrt(max(self.max_power, _EPS)) * 0.5, dtype=np.float64)
        for _ in range(self.iterations):
            total_rx = gains @ (v**2) + noise
            signal = np.sqrt(diag_gain) * v
            u = signal / np.maximum(total_rx, _EPS)
            mse = 1.0 - 2.0 * u * signal + (u**2) * total_rx
            w = weights / np.maximum(mse, _EPS)
            denom = gains.T @ (w * (u**2))
            updated_v = (w * u * np.sqrt(diag_gain)) / np.maximum(denom, _EPS)
            updated_v = np.clip(updated_v, 0.0, np.sqrt(max(self.max_power, _EPS)))
            v = self.damping * v + (1.0 - self.damping) * updated_v

        power = v**2
        interference = gains @ power - diag_gain * power
        weighted_rate = weights * np.log2(1.0 + (diag_gain * power) / np.maximum(noise + interference, _EPS))
        selected = _top_indices(weighted_rate, self.num_active)
        mask = _mask_from_indices(selected.tolist(), num_ut)
        # WMMSE solves for physically meaningful levels; peak-renormalising would
        # throw the solution away and leave only its ratios.
        power_out = _normalize_power(
            power,
            mask,
            max_power=self.max_power,
            min_active_power=self.min_active_power,
            mode="absolute",
        )
        return ResourceDirectives(active_ut_mask=mask, per_ut_power=power_out, pilot_reuse_factor=1)


class QueueAwareLyapunovResourceManager(ResourceManager):
    """
    Virtual-queue drift-plus-penalty scheduler.

    The current runtime does not expose real packet queues, so this manager
    maintains synthetic backlog state per Eb/No point to approximate
    queue-aware scheduling pressure.
    """

    def __init__(
        self,
        num_active: int = 1,
        arrival_rate: float | Sequence[float] = 0.35,
        utility_weight: float = 0.2,
        queue_cap: float = 10.0,
        initial_queue: float = 0.0,
        max_power: float = 1.0,
        min_active_power: float = 0.2,
    ) -> None:
        self.needs_channel_feedback = True
        self.num_active = max(1, int(num_active))
        self.arrival_rate = arrival_rate
        self.utility_weight = float(utility_weight)
        self.queue_cap = float(queue_cap)
        self.initial_queue = float(initial_queue)
        self.max_power = float(max_power)
        self.min_active_power = float(min_active_power)
        self._queues = _PerPointState(
            lambda num_ut: np.full(num_ut, self.initial_queue, dtype=np.float64)
        )

    def _arrivals(self, num_ut: int) -> np.ndarray:
        if np.isscalar(self.arrival_rate):
            return np.full(num_ut, float(self.arrival_rate), dtype=np.float64)
        arrivals = np.asarray(list(self.arrival_rate), dtype=np.float64)
        if arrivals.shape[0] != num_ut:
            raise ValueError(
                f"Queue-aware arrival_rate length {arrivals.shape[0]} does not match num_ut={num_ut}."
            )
        return arrivals

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = _num_ut(config)
        queues = np.array(self._queues.get(ebno_db, num_ut), dtype=np.float64)

        queues = np.minimum(queues + self._arrivals(num_ut), self.queue_cap)
        if feedback is None:
            score = queues + self.utility_weight
            rate_metric = np.ones(num_ut, dtype=np.float64)
        else:
            rate_metric = _normalized_rate_metric(feedback, ebno_db, config=config)
            score = (queues + self.utility_weight) * rate_metric

        selected = _top_indices(score, self.num_active)
        mask = _mask_from_indices(selected.tolist(), num_ut)
        mask_arr = np.asarray(mask, dtype=np.float64)
        served = mask_arr * rate_metric
        queues = np.maximum(queues - served, 0.0)
        self._queues.set(ebno_db, queues)

        power_basis = score + 0.25 * queues
        power_out = _normalize_power(
            power_basis,
            mask,
            max_power=self.max_power,
            min_active_power=self.min_active_power,
        )
        return ResourceDirectives(active_ut_mask=mask, per_ut_power=power_out, pilot_reuse_factor=1)

    def export_state(self) -> dict[str, Any]:
        return {"queues": self._queues.export()}

    def load_state(self, state: dict[str, Any]) -> None:
        self._queues.load(state.get("queues", {}))


class DRLResourceManager(ResourceManager):
    """
    Actor-style RM interface with optional Keras policy loading.

    If no trained policy artifact is supplied, the manager falls back to a
    lightweight heuristic actor so it remains runnable inside the simulator.
    """

    def __init__(
        self,
        num_active: int = 1,
        model_path: str | None = None,
        temperature: float = 0.75,
        fairness_weight: float = 0.35,
        history_alpha: float = 0.15,
        max_power: float = 1.0,
        min_active_power: float = 0.2,
        policy_score_weight: float = 1.0,
        channel_gain_weight: float = 0.0,
        strict: bool = True,
        model_root: str | Path | None = None,
    ) -> None:
        self.needs_channel_feedback = True
        self.num_active = max(1, int(num_active))
        self.temperature = float(temperature)
        self.fairness_weight = float(fairness_weight)
        self.history_alpha = float(np.clip(history_alpha, 0.01, 1.0))
        self.max_power = float(max_power)
        self.min_active_power = float(min_active_power)
        self.policy_score_weight = float(policy_score_weight)
        self.channel_gain_weight = float(channel_gain_weight)
        self.model = None
        self.model_path = model_path
        self.policy_checkpoint = None
        self.strict = bool(strict)
        # Provenance: a learned curve that silently came from the heuristic
        # fallback is indistinguishable from a real one in the output artifacts,
        # so record exactly what ran.
        self.policy_loaded = False
        self.policy_load_error: str | None = None
        self.resolved_model_path: str | None = None
        self.policy_checkpoint_digest: str | None = None
        self._avg_rates = _PerPointState(
            lambda num_ut: np.full(num_ut, 1e-3, dtype=np.float64)
        )

        if model_path:
            resolved = _resolve_model_path(model_path, model_root)
            self.resolved_model_path = str(resolved)
            try:
                from factory6g.models.drl_policy import load_policy_checkpoint

                self.policy_checkpoint = load_policy_checkpoint(str(resolved))
                self.model = self.policy_checkpoint.model
                self.policy_loaded = True
                self.policy_checkpoint_digest = _checkpoint_digest(resolved)
                print(f"Loaded DRL Resource Manager checkpoint from {resolved}")
            except Exception as exc:
                # Keras deserialisation errors embed the entire model config;
                # keep the record readable.
                detail = str(exc)
                if len(detail) > 400:
                    detail = detail[:400] + " ... [truncated]"
                self.policy_load_error = f"{type(exc).__name__}: {detail}"
                if self.strict:
                    raise RuntimeError(
                        f"Failed to load DRL resource-manager policy from '{resolved}': {detail}. "
                        "Refusing to silently fall back to the heuristic actor, which would "
                        "publish a hand-written rule under a learned method's name. Pass "
                        "strict=False (or set resource_managers.strict_policy_loading to false) "
                        "to allow the fallback."
                    ) from exc
                print(f"Failed to load DRL resource manager model from {resolved}: {exc}")
                print("DRLResourceManager will use heuristic actor fallback.")
        elif self.strict:
            raise ValueError(
                "DRLResourceManager requires a model_path when strict policy loading is on. "
                "Set resource_managers.drl_model_path, or disable strict loading to run the "
                "heuristic actor explicitly."
            )

    def provenance(self) -> dict[str, Any]:
        """What actually ran, for the stage output."""
        return {
            "policy_loaded": self.policy_loaded,
            "model_path": self.resolved_model_path,
            "checkpoint_sha256": self.policy_checkpoint_digest,
            "load_error": self.policy_load_error,
            "actor": "learned_policy" if self.policy_loaded else "heuristic_fallback",
        }

    def _predict_policy(
        self,
        channel_energy: np.ndarray,
        fairness_debt: np.ndarray,
        ebno_db: float,
        num_ut: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.policy_checkpoint is None:
            raise RuntimeError("No DRL policy model loaded.")

        from factory6g.models.drl_policy import (
            build_policy_state,
            fairness_input_for_inference,
            predict_policy_outputs,
        )

        # Feed the fairness input the checkpoint was actually trained with; a
        # policy trained under the constant regime must not be driven by a live
        # signal its weights never saw.
        fairness_input = fairness_input_for_inference(
            fairness_debt, self.policy_checkpoint.metadata
        )
        state = build_policy_state(channel_energy, ebno_db, fairness_debt=fairness_input)
        outputs = predict_policy_outputs(self.policy_checkpoint, state)
        sched_np = np.asarray(outputs["schedule_output"], dtype=np.float64).reshape(-1)
        power_np = np.asarray(outputs["power_output"], dtype=np.float64).reshape(-1)
        if sched_np.shape[0] < num_ut:
            sched_np = np.pad(sched_np, (0, num_ut - sched_np.shape[0]), constant_values=float(np.min(sched_np)))
        if power_np.shape[0] < num_ut:
            power_np = np.pad(power_np, (0, num_ut - power_np.shape[0]), constant_values=0.0)
        return sched_np[:num_ut], power_np[:num_ut]

    def get_runtime_directives(
        self,
        config: dict[str, Any],
        ebno_db: float,
        feedback: ResourceManagerFeedback | None = None,
    ) -> ResourceDirectives:
        num_ut = _num_ut(config)
        if feedback is None:
            mask = _mask_from_indices(range(min(self.num_active, num_ut)), num_ut)
            power = _normalize_power(np.ones(num_ut, dtype=np.float64), mask, max_power=self.max_power)
            return ResourceDirectives(active_ut_mask=mask, per_ut_power=power, pilot_reuse_factor=1)

        avg_rates = np.array(self._avg_rates.get(ebno_db, num_ut), dtype=np.float64)

        inst_rates = _instantaneous_rate_per_user(feedback, ebno_db, config=config)
        norm_rates = inst_rates / max(float(np.max(inst_rates, initial=0.0)), _EPS)
        fairness_debt = 1.0 / np.maximum(avg_rates, 1e-3)
        fairness_debt = fairness_debt / max(float(np.max(fairness_debt, initial=0.0)), _EPS)

        if self.policy_checkpoint is not None:
            from factory6g.models.drl_policy import channel_energy_from_h_hat, project_policy_to_directives

            channel_energy = channel_energy_from_h_hat(feedback.h_hat)[0]
            sched_scores, power_scores = self._predict_policy(channel_energy, fairness_debt, ebno_db, num_ut)
            sched_scores = (
                self.policy_score_weight * sched_scores
                + self.channel_gain_weight * norm_rates
                + self.fairness_weight * fairness_debt
            )
            power_scores = self.policy_score_weight * power_scores + self.channel_gain_weight * norm_rates
            mask, power_out = project_policy_to_directives(
                sched_scores,
                power_scores,
                num_active=self.num_active,
                max_power=self.max_power,
                min_active_power=self.min_active_power,
            )
        else:
            logits = norm_rates + self.fairness_weight * fairness_debt
            policy_probs = _softmax(logits, self.temperature)
            sched_scores = policy_probs + 0.5 * norm_rates
            power_scores = 0.5 * policy_probs + 0.5 * norm_rates + 0.25 * self.fairness_weight * fairness_debt
            selected = _top_indices(sched_scores, self.num_active)
            mask = _mask_from_indices(selected.tolist(), num_ut)
            power_out = _normalize_power(
                power_scores,
                mask,
                max_power=self.max_power,
                min_active_power=self.min_active_power,
            )

        served = np.asarray(mask, dtype=np.float64) * inst_rates
        avg_rates = (1.0 - self.history_alpha) * avg_rates + self.history_alpha * np.maximum(served, 1e-3)
        self._avg_rates.set(ebno_db, avg_rates)
        return ResourceDirectives(active_ut_mask=mask, per_ut_power=power_out, pilot_reuse_factor=1)

    def export_state(self) -> dict[str, Any]:
        return {"avg_rates": self._avg_rates.export()}

    def load_state(self, state: dict[str, Any]) -> None:
        self._avg_rates.load(state.get("avg_rates", {}))


# Exact-name registry. The previous dispatch matched substrings in order, so
# `"max"` caught anything containing "max" and a name like `"static_drl"`
# silently resolved to the static manager. Names are explicit here, and an
# unknown name fails loudly with the list of valid ones.
RESOURCE_MANAGER_ALIASES: dict[str, str] = {
    "static": "static",
    "static_full_load": "static",
    "static_subset": "static_subset",
    "round_robin": "round_robin",
    "roundrobin": "round_robin",
    "rr": "round_robin",
    "max_throughput": "max_throughput",
    "maxthroughput": "max_throughput",
    "max_snr": "max_throughput",
    "pf": "proportional_fair",
    "prop_fair": "proportional_fair",
    "proportional_fair": "proportional_fair",
    "wmmse": "wmmse",
    "queue_aware": "queue_aware",
    "lyapunov": "queue_aware",
    "backpressure": "queue_aware",
    "drl": "drl",
    "ppo": "drl",
    "actor": "drl",
    "reliability_drl": "reliability_drl",
    "reliability-drl": "reliability_drl",
    "reliabilitydrl": "reliability_drl",
    "ber_drl": "reliability_drl",
    "cnn": "cnn",
}


def resolve_resource_manager_name(name: str) -> str:
    """Map a configured resource-manager name onto its canonical key."""
    key = str(name).strip().lower()
    if key not in RESOURCE_MANAGER_ALIASES:
        raise ValueError(
            f"Unknown resource manager '{name}'. "
            f"Valid names: {sorted(RESOURCE_MANAGER_ALIASES)}."
        )
    return RESOURCE_MANAGER_ALIASES[key]


def create_resource_manager(
    name: str,
    *,
    num_ut: int,
    num_active: int,
    cnn_model_path: str | None,
    drl_model_path: str | None,
    manager_kwargs: dict[str, Any] | None = None,
    strict_policy_loading: bool = True,
    model_root: str | Path | None = None,
) -> ResourceManager:
    canonical = resolve_resource_manager_name(name)
    kwargs = dict(manager_kwargs or {})

    if canonical == "static":
        # Manager kwargs used to be dropped on the floor here.
        kwargs.setdefault("active_ut_mask", [1] * num_ut)
        kwargs.setdefault("per_ut_power", [1.0] * num_ut)
        return StaticResourceManager(**kwargs)
    if canonical == "static_subset":
        kwargs.setdefault("num_active", num_active)
        kwargs.setdefault("num_ut", num_ut)
        return StaticResourceManager(**kwargs)
    if canonical == "round_robin":
        return RoundRobinResourceManager(num_active=num_active, **kwargs)
    if canonical == "max_throughput":
        return MaxThroughputResourceManager(num_active=num_active, **kwargs)
    if canonical == "proportional_fair":
        return ProportionalFairResourceManager(num_active=num_active, **kwargs)
    if canonical == "wmmse":
        return WMMSEResourceManager(num_active=num_active, **kwargs)
    if canonical == "queue_aware":
        return QueueAwareLyapunovResourceManager(num_active=num_active, **kwargs)
    if canonical == "reliability_drl":
        kwargs.setdefault("model_path", "models/reliability_drl_resource_manager_policy")
        kwargs.setdefault("strict", strict_policy_loading)
        kwargs.setdefault("model_root", model_root)
        return DRLResourceManager(num_active=num_active, **kwargs)
    if canonical == "drl":
        kwargs.setdefault("model_path", drl_model_path)
        kwargs.setdefault("strict", strict_policy_loading)
        kwargs.setdefault("model_root", model_root)
        return DRLResourceManager(num_active=num_active, **kwargs)
    if canonical == "cnn":
        from factory6g.models.cnn_resource_manager import CNNResourceManager

        kwargs.setdefault("model_path", cnn_model_path)
        return CNNResourceManager(**kwargs)
    raise ValueError(f"Unhandled resource manager '{name}' (canonical '{canonical}').")
