from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


POLICY_CHECKPOINT_FORMAT = "factory6g_drl_policy_v1"
POLICY_MODEL_FILENAME = "policy.keras"
POLICY_METADATA_FILENAME = "metadata.json"
POLICY_NORMALIZATION_FILENAME = "normalization.npz"
POLICY_HISTORY_FILENAME = "history.json"
_EPS = 1e-6


@dataclass(frozen=True)
class PolicyNormalization:
    channel_mean: np.ndarray
    channel_std: np.ndarray
    ebno_mean: float
    ebno_std: float
    fairness_mean: float
    fairness_std: float

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "channel_mean": np.asarray(self.channel_mean, dtype=np.float32),
            "channel_std": np.asarray(self.channel_std, dtype=np.float32),
            "ebno_mean": np.asarray(self.ebno_mean, dtype=np.float32),
            "ebno_std": np.asarray(self.ebno_std, dtype=np.float32),
            "fairness_mean": np.asarray(self.fairness_mean, dtype=np.float32),
            "fairness_std": np.asarray(self.fairness_std, dtype=np.float32),
        }

    @classmethod
    def from_npz(cls, npz_path: str | Path) -> "PolicyNormalization":
        payload = np.load(npz_path)
        return cls(
            channel_mean=np.asarray(payload["channel_mean"], dtype=np.float32),
            channel_std=np.asarray(payload["channel_std"], dtype=np.float32),
            ebno_mean=float(np.asarray(payload["ebno_mean"]).item()),
            ebno_std=float(np.asarray(payload["ebno_std"]).item()),
            fairness_mean=float(np.asarray(payload["fairness_mean"]).item()),
            fairness_std=float(np.asarray(payload["fairness_std"]).item()),
        )


@dataclass(frozen=True)
class PolicyCheckpoint:
    model: Any
    metadata: dict[str, Any]
    normalization: PolicyNormalization | None
    checkpoint_dir: Path


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def load_rm_policy_dataset(data_path: str) -> dict[str, np.ndarray]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to load the RM training parquet dataset.") from exc

    df = pd.read_parquet(data_path)
    channel_energy = np.stack([np.stack(item).astype(np.float32) for item in df["channel_energy"].tolist()])
    active_mask = np.asarray(df["active_ut_mask"].tolist(), dtype=np.float32)
    power = np.asarray(df["per_ut_power"].tolist(), dtype=np.float32)
    ebno_db = df["ebno_db"].to_numpy(dtype=np.float32)
    utility = (
        df["oracle_utility"].to_numpy(dtype=np.float32)
        if "oracle_utility" in df.columns
        else np.zeros(len(df), dtype=np.float32)
    )
    return {
        "channel_energy": channel_energy,
        "active_ut_mask": active_mask,
        "per_ut_power": power,
        "ebno_db": ebno_db,
        "oracle_utility": utility,
    }


def channel_energy_from_h_hat(h_hat: Any) -> np.ndarray:
    h_hat_np = np.asarray(_to_numpy(h_hat), dtype=np.complex64)
    if h_hat_np.ndim != 7:
        raise ValueError(f"Expected h_hat with 7 dimensions, got shape {h_hat_np.shape}.")

    power = np.abs(h_hat_np) ** 2
    power = np.mean(power, axis=1)
    power = np.mean(power, axis=1)
    power = np.mean(power, axis=2)
    channel_energy = np.mean(power, axis=2)
    return channel_energy.astype(np.float32)


def build_policy_state(
    channel_energy: np.ndarray,
    ebno_db: float | np.ndarray,
    fairness_debt: float | np.ndarray | None = None,
) -> np.ndarray:
    channel_energy_np = np.asarray(channel_energy, dtype=np.float32)
    squeeze = channel_energy_np.ndim == 2
    if squeeze:
        channel_energy_np = channel_energy_np[None, ...]
    if channel_energy_np.ndim != 3:
        raise ValueError(
            f"Expected channel_energy with shape [num_ut, fft] or [batch, num_ut, fft], got {channel_energy_np.shape}."
        )

    batch_size, num_ut, _ = channel_energy_np.shape

    ebno_np = np.asarray(ebno_db, dtype=np.float32)
    if ebno_np.ndim == 0:
        ebno_np = np.full((batch_size, 1, 1), float(ebno_np), dtype=np.float32)
    elif ebno_np.ndim == 1 and ebno_np.shape[0] == batch_size:
        ebno_np = ebno_np[:, None, None]
    else:
        raise ValueError(f"Expected ebno_db scalar or [batch], got shape {ebno_np.shape}.")
    ebno_feature = np.broadcast_to(ebno_np, (batch_size, num_ut, 1))

    if fairness_debt is None:
        fairness_feature = np.ones((batch_size, num_ut, 1), dtype=np.float32)
    else:
        fairness_np = np.asarray(fairness_debt, dtype=np.float32)
        if fairness_np.ndim == 0:
            fairness_np = np.full((batch_size, num_ut, 1), float(fairness_np), dtype=np.float32)
        elif fairness_np.ndim == 1 and fairness_np.shape[0] == num_ut:
            fairness_np = fairness_np[None, :, None]
        elif fairness_np.ndim == 2 and fairness_np.shape == (batch_size, num_ut):
            fairness_np = fairness_np[..., None]
        else:
            raise ValueError(
                "Expected fairness_debt scalar, [num_ut], or [batch, num_ut], "
                f"got shape {fairness_np.shape}."
            )
        fairness_feature = np.broadcast_to(fairness_np, (batch_size, num_ut, 1))

    state = np.concatenate([channel_energy_np, fairness_feature, ebno_feature], axis=-1)
    return state[0] if squeeze else state


def compute_policy_normalization(
    channel_energy: np.ndarray,
    ebno_db: np.ndarray,
    fairness_debt: np.ndarray | None = None,
) -> PolicyNormalization:
    channel_energy_np = np.asarray(channel_energy, dtype=np.float32)
    if channel_energy_np.ndim != 3:
        raise ValueError(f"Expected channel_energy dataset with rank 3, got {channel_energy_np.shape}.")

    ebno_np = np.asarray(ebno_db, dtype=np.float32).reshape(-1)
    if ebno_np.shape[0] != channel_energy_np.shape[0]:
        raise ValueError(
            f"Expected ebno_db length {channel_energy_np.shape[0]}, got {ebno_np.shape[0]}."
        )

    if fairness_debt is None:
        fairness_mean = 1.0
        fairness_std = 1.0
    else:
        fairness_np = np.asarray(fairness_debt, dtype=np.float32)
        fairness_mean = float(np.mean(fairness_np))
        fairness_std = max(float(np.std(fairness_np)), _EPS)

    return PolicyNormalization(
        channel_mean=np.mean(channel_energy_np, axis=(0, 1)).astype(np.float32),
        channel_std=np.maximum(np.std(channel_energy_np, axis=(0, 1)), _EPS).astype(np.float32),
        ebno_mean=float(np.mean(ebno_np)),
        ebno_std=max(float(np.std(ebno_np)), _EPS),
        fairness_mean=fairness_mean,
        fairness_std=fairness_std,
    )


def normalize_policy_state(state: np.ndarray, normalization: PolicyNormalization) -> np.ndarray:
    state_np = np.asarray(state, dtype=np.float32)
    squeeze = state_np.ndim == 2
    if squeeze:
        state_np = state_np[None, ...]
    if state_np.ndim != 3:
        raise ValueError(f"Expected policy state rank 2 or 3, got shape {state_np.shape}.")

    channel_part = (state_np[..., :-2] - normalization.channel_mean) / normalization.channel_std
    fairness_part = (state_np[..., -2:-1] - normalization.fairness_mean) / normalization.fairness_std
    ebno_part = (state_np[..., -1:] - normalization.ebno_mean) / normalization.ebno_std
    normalized = np.concatenate([channel_part, fairness_part, ebno_part], axis=-1).astype(np.float32)
    return normalized[0] if squeeze else normalized


def build_policy_training_inputs(
    channel_energy: np.ndarray,
    ebno_db: np.ndarray,
    fairness_debt: np.ndarray | None = None,
) -> tuple[np.ndarray, PolicyNormalization]:
    normalization = compute_policy_normalization(channel_energy, ebno_db, fairness_debt=fairness_debt)
    state = build_policy_state(channel_energy, ebno_db, fairness_debt=fairness_debt)
    normalized_state = normalize_policy_state(state, normalization)
    return normalized_state.astype(np.float32), normalization


def create_policy_model(
    input_shape: tuple[int, int],
    output_dim: int,
    *,
    hidden_dim: int = 128,
    dropout_rate: float = 0.2,
):
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=input_shape, name="policy_state")
    x = layers.LayerNormalization(name="input_norm")(inputs)
    x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv_1")(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu", name="conv_2")(x)
    user_features = layers.Dense(hidden_dim, activation="relu", name="user_dense")(x)
    pooled = layers.GlobalAveragePooling1D(name="global_pool")(user_features)
    context = layers.Dense(hidden_dim, activation="relu", name="context_dense")(pooled)
    context = layers.Dropout(dropout_rate, name="context_dropout")(context)
    repeated_context = layers.RepeatVector(output_dim, name="context_repeat")(context)
    joint = layers.Concatenate(axis=-1, name="joint_features")([user_features, repeated_context])

    schedule_scores = layers.TimeDistributed(
        layers.Dense(1, activation="sigmoid"),
        name="schedule_td",
    )(joint)
    power_scores = layers.TimeDistributed(
        layers.Dense(1, activation="sigmoid"),
        name="power_td",
    )(joint)
    schedule_output = layers.Reshape((output_dim,), name="schedule_output")(schedule_scores)
    power_output = layers.Reshape((output_dim,), name="power_output")(power_scores)
    value_output = layers.Dense(1, name="value_output")(context)

    return models.Model(
        inputs=inputs,
        outputs=[schedule_output, power_output, value_output],
        name="factory6g_drl_policy",
    )


def compile_policy_model(
    model,
    *,
    learning_rate: float = 1e-3,
    value_loss_weight: float = 0.1,
) -> None:
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "schedule_output": "binary_crossentropy",
            "power_output": "mse",
            "value_output": "mse",
        },
        loss_weights={
            "schedule_output": 1.0,
            "power_output": 0.5,
            "value_output": value_loss_weight,
        },
        metrics={
            "schedule_output": "accuracy",
            "power_output": "mse",
            "value_output": "mse",
        },
    )


def _load_model_compat(model_path: str):
    import tensorflow as tf

    class _CompatibleDense(tf.keras.layers.Dense):
        def __init__(self, *args, quantization_config=None, **kwargs):
            super().__init__(*args, **kwargs)

    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        with tf.keras.utils.custom_object_scope({"Dense": _CompatibleDense}):
            return tf.keras.models.load_model(model_path, compile=False)


def save_policy_checkpoint(
    output_dir: str | Path,
    model,
    normalization: PolicyNormalization | None,
    metadata: dict[str, Any],
    *,
    history: dict[str, Any] | None = None,
) -> Path:
    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / POLICY_MODEL_FILENAME
    metadata_path = checkpoint_dir / POLICY_METADATA_FILENAME
    normalization_path = checkpoint_dir / POLICY_NORMALIZATION_FILENAME
    history_path = checkpoint_dir / POLICY_HISTORY_FILENAME

    model.save(model_path)
    if normalization is not None:
        np.savez(normalization_path, **normalization.to_npz_payload())

    payload = {
        "format": POLICY_CHECKPOINT_FORMAT,
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_filename": POLICY_MODEL_FILENAME,
        "normalization_filename": POLICY_NORMALIZATION_FILENAME if normalization is not None else None,
        "history_filename": POLICY_HISTORY_FILENAME if history is not None else None,
    }
    payload.update(metadata)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    if history is not None:
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    return checkpoint_dir


def load_policy_checkpoint(checkpoint_path: str | Path) -> PolicyCheckpoint:
    path = Path(checkpoint_path)
    if path.is_dir():
        checkpoint_dir = path
        model_path = checkpoint_dir / POLICY_MODEL_FILENAME
        metadata_path = checkpoint_dir / POLICY_METADATA_FILENAME
        normalization_path = checkpoint_dir / POLICY_NORMALIZATION_FILENAME
    elif path.is_file():
        checkpoint_dir = path.parent
        model_path = path
        metadata_path = checkpoint_dir / POLICY_METADATA_FILENAME
        normalization_path = checkpoint_dir / POLICY_NORMALIZATION_FILENAME
    else:
        raise FileNotFoundError(f"Policy checkpoint path not found: {path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Policy model file not found: {model_path}")

    metadata: dict[str, Any]
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    else:
        metadata = {
            "format": "legacy_model_file",
            "version": 0,
            "model_filename": model_path.name,
        }

    normalization = PolicyNormalization.from_npz(normalization_path) if normalization_path.exists() else None
    model = _load_model_compat(str(model_path))
    return PolicyCheckpoint(
        model=model,
        metadata=metadata,
        normalization=normalization,
        checkpoint_dir=checkpoint_dir,
    )


def _parse_policy_outputs(predictions: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if isinstance(predictions, (list, tuple)):
        schedule_output = predictions[0]
        power_output = predictions[1] if len(predictions) > 1 else predictions[0]
        value_output = predictions[2] if len(predictions) > 2 else None
    elif isinstance(predictions, dict):
        schedule_output = predictions.get("schedule_output", predictions.get("policy", predictions.get("output_1")))
        power_output = predictions.get("power_output", predictions.get("output_2", schedule_output))
        value_output = predictions.get("value_output")
    else:
        schedule_output = predictions
        power_output = predictions
        value_output = None

    schedule_np = np.asarray(_to_numpy(schedule_output), dtype=np.float32)
    power_np = np.asarray(_to_numpy(power_output), dtype=np.float32)
    value_np = None if value_output is None else np.asarray(_to_numpy(value_output), dtype=np.float32)
    return schedule_np, power_np, value_np


def predict_policy_outputs(checkpoint: PolicyCheckpoint, state: np.ndarray) -> dict[str, np.ndarray | float | None]:
    state_np = np.asarray(state, dtype=np.float32)
    squeeze = state_np.ndim == 2
    if squeeze:
        state_np = state_np[None, ...]
    if state_np.ndim != 3:
        raise ValueError(f"Expected policy state rank 2 or 3, got {state_np.shape}.")

    if checkpoint.normalization is not None:
        model_input = normalize_policy_state(state_np, checkpoint.normalization)
    else:
        model_input = state_np

    predictions = checkpoint.model(model_input, training=False)
    schedule_np, power_np, value_np = _parse_policy_outputs(predictions)

    if squeeze:
        schedule_np = schedule_np[0]
        power_np = power_np[0]
        if value_np is not None:
            value_np = float(np.asarray(value_np).reshape(-1)[0])
    return {
        "schedule_output": schedule_np,
        "power_output": power_np,
        "value_output": value_np,
    }


def project_policy_to_directives(
    schedule_scores: np.ndarray,
    power_scores: np.ndarray,
    *,
    num_active: int,
    max_power: float = 1.0,
    min_active_power: float = 0.2,
) -> tuple[list[int], list[float]]:
    schedule_np = np.asarray(schedule_scores, dtype=np.float32).reshape(-1)
    power_np = np.clip(np.asarray(power_scores, dtype=np.float32).reshape(-1), 0.0, None)
    if schedule_np.shape[0] != power_np.shape[0]:
        raise ValueError(
            f"schedule_scores length {schedule_np.shape[0]} does not match power_scores length {power_np.shape[0]}."
        )

    num_ut = schedule_np.shape[0]
    active_count = max(1, min(int(num_active), num_ut))
    selected = np.argsort(schedule_np)[::-1][:active_count]
    mask = np.zeros(num_ut, dtype=np.int32)
    mask[selected] = 1

    directives_power = np.zeros(num_ut, dtype=np.float32)
    active_power = power_np[selected]
    peak = max(float(np.max(active_power, initial=0.0)), _EPS)
    normalized = np.clip((active_power / peak) * max_power, min_active_power, max_power)
    directives_power[selected] = normalized.astype(np.float32)
    return mask.tolist(), directives_power.tolist()
