from __future__ import annotations

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import numpy as np
import tensorflow as tf

from src.models.drl_policy import (
    build_policy_training_inputs,
    compile_policy_model,
    create_policy_model,
    load_policy_checkpoint,
    load_rm_policy_dataset,
    save_policy_checkpoint,
)
from src.sim.config import load_config


def _validate_shapes(
    channel_energy: np.ndarray,
    y_mask: np.ndarray,
    y_power: np.ndarray,
    ebno_db: np.ndarray,
    *,
    config_path: str,
) -> None:
    app_config = load_config(config_path)
    expected_num_ut = app_config.system.num_ut
    expected_fft_size = app_config.system.fft_size

    if channel_energy.ndim != 3:
        raise ValueError(f"Expected channel_energy rank 3, got shape {channel_energy.shape}.")
    if tuple(channel_energy.shape[1:]) != (expected_num_ut, expected_fft_size):
        raise ValueError(
            f"Dataset feature shape {channel_energy.shape[1:]} does not match config "
            f"({expected_num_ut}, {expected_fft_size})."
        )
    if y_mask.shape != y_power.shape or y_mask.shape[1] != expected_num_ut:
        raise ValueError(
            f"Target shapes {y_mask.shape} / {y_power.shape} do not match config num_ut={expected_num_ut}."
        )
    if ebno_db.shape[0] != channel_energy.shape[0]:
        raise ValueError(
            f"Expected ebno_db length {channel_energy.shape[0]}, got {ebno_db.shape[0]}."
        )


def _ber_reliability_target(avg_ber: np.ndarray, ber_clip: float) -> np.ndarray:
    clip = max(float(ber_clip), 1e-12)
    ber = np.clip(np.asarray(avg_ber, dtype=np.float32), 0.0, clip)
    return (1.0 - (ber / clip)).astype(np.float32).reshape(-1, 1)


def _ber_log_reliability_target(avg_ber: np.ndarray, ber_clip: float, ber_floor: float) -> np.ndarray:
    floor = max(float(ber_floor), 1e-12)
    clip = max(float(ber_clip), floor * 10.0)
    ber = np.clip(np.asarray(avg_ber, dtype=np.float32), floor, clip)
    log_floor = np.log10(floor)
    log_clip = np.log10(clip)
    reliability = 1.0 - ((np.log10(ber) - log_floor) / max(log_clip - log_floor, 1e-6))
    return np.clip(reliability, 0.0, 1.0).astype(np.float32).reshape(-1, 1)


def _sample_weights(
    dataset: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    mode: str,
    strength: float,
) -> dict[str, np.ndarray] | None:
    if mode == "none":
        return None

    target_flat = np.asarray(target, dtype=np.float32).reshape(-1)
    if mode == "reliability":
        weights = 1.0 + float(strength) * target_flat
    elif mode == "ber_confidence":
        ber_upper = np.asarray(dataset["oracle_ber_upper_confidence"], dtype=np.float32)
        scale = np.maximum(np.percentile(ber_upper, 75), 1e-9)
        confidence = 1.0 - np.clip(ber_upper / scale, 0.0, 1.0)
        weights = 1.0 + float(strength) * confidence
    else:
        raise ValueError(f"Unknown sample weight mode: {mode}")

    weights = np.asarray(weights, dtype=np.float32)
    return {
        "schedule_output": weights,
        "power_output": weights,
        "value_output": weights,
    }


def train_drl_resource_manager(args: argparse.Namespace) -> None:
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    dataset = load_rm_policy_dataset(args.data)
    channel_energy = dataset["channel_energy"]
    y_mask = dataset["active_ut_mask"]
    y_power = dataset["per_ut_power"]
    ebno_db = dataset["ebno_db"]
    if args.value_target == "ber_reliability":
        y_utility = _ber_reliability_target(dataset["oracle_avg_ber"], args.ber_clip)
    elif args.value_target == "ber_log_reliability":
        y_utility = _ber_log_reliability_target(dataset["oracle_avg_ber"], args.ber_clip, args.ber_floor)
    else:
        y_utility = dataset["oracle_utility"].reshape(-1, 1)

    _validate_shapes(
        channel_energy,
        y_mask,
        y_power,
        ebno_db,
        config_path=args.config,
    )

    x_state, normalization = build_policy_training_inputs(channel_energy, ebno_db)
    if args.initial_checkpoint:
        checkpoint = load_policy_checkpoint(args.initial_checkpoint)
        model = checkpoint.model
        expected_input = tuple(x_state.shape[1:])
        actual_input = tuple(model.input_shape[1:])
        if actual_input != expected_input:
            raise ValueError(
                f"Initial checkpoint input shape {actual_input} does not match dataset state shape {expected_input}."
            )
    else:
        model = create_policy_model(
            input_shape=tuple(x_state.shape[1:]),
            output_dim=int(y_mask.shape[1]),
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout,
        )
    compile_policy_model(
        model,
        learning_rate=args.lr,
        value_loss_weight=args.value_loss_weight,
        schedule_loss_weight=args.schedule_loss_weight,
        power_loss_weight=args.power_loss_weight,
    )
    model.summary()

    callbacks: list[tf.keras.callbacks.Callback] = []
    if args.validation_split > 0.0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.early_stop_patience,
                restore_best_weights=True,
            )
        )

    sample_weight = _sample_weights(
        dataset,
        y_utility,
        mode=args.sample_weight_mode,
        strength=args.sample_weight_strength,
    )

    history = model.fit(
        x_state,
        {
            "schedule_output": y_mask,
            "power_output": y_power,
            "value_output": y_utility,
        },
        sample_weight=sample_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    metadata = {
        "config_path": args.config,
        "data_path": args.data,
        "num_samples": int(x_state.shape[0]),
        "num_ut": int(y_mask.shape[1]),
        "fft_size": int(channel_energy.shape[2]),
        "state_dim": int(x_state.shape[2]),
        "checkpoint_type": "offline_actor_pretraining",
        "policy_outputs": ["schedule_output", "power_output", "value_output"],
        "training_args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "validation_split": args.validation_split,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "value_loss_weight": args.value_loss_weight,
            "schedule_loss_weight": args.schedule_loss_weight,
            "power_loss_weight": args.power_loss_weight,
            "value_target": args.value_target,
            "ber_clip": args.ber_clip,
            "ber_floor": args.ber_floor,
            "sample_weight_mode": args.sample_weight_mode,
            "sample_weight_strength": args.sample_weight_strength,
            "initial_checkpoint": args.initial_checkpoint,
            "seed": args.seed,
        },
    }
    checkpoint_dir = save_policy_checkpoint(
        args.output_dir,
        model,
        normalization,
        metadata,
        history=history.history,
    )
    print(f"Training complete. Policy checkpoint saved to {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Factory6G DRL-style resource-manager policy checkpoint."
    )
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config JSON file")
    parser.add_argument(
        "--data",
        type=str,
        default="data/rm_training_data_sionna_1k.parquet",
        help="Path to the training parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/drl_resource_manager_policy",
        help="Output checkpoint directory",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--value-loss-weight", type=float, default=0.1)
    parser.add_argument("--schedule-loss-weight", type=float, default=1.0)
    parser.add_argument("--power-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--value-target",
        choices=["utility", "ber_reliability", "ber_log_reliability"],
        default="utility",
        help="Target used for the policy value head.",
    )
    parser.add_argument(
        "--ber-clip",
        type=float,
        default=0.1,
        help="BER value mapped to zero reliability when --value-target=ber_reliability.",
    )
    parser.add_argument(
        "--ber-floor",
        type=float,
        default=1e-7,
        help="BER value mapped to full reliability when --value-target=ber_log_reliability.",
    )
    parser.add_argument(
        "--sample-weight-mode",
        choices=["none", "reliability", "ber_confidence"],
        default="none",
        help="Optional sample weighting for fine-tuning BER-first policy heads.",
    )
    parser.add_argument(
        "--sample-weight-strength",
        type=float,
        default=1.0,
        help="Multiplier for --sample-weight-mode.",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=str,
        default=None,
        help="Existing policy checkpoint to fine-tune instead of training a new model from scratch.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    train_drl_resource_manager(parser.parse_args())
