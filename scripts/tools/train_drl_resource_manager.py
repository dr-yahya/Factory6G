from __future__ import annotations

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import numpy as np
import tensorflow as tf

from src.models.drl_policy import (
    build_policy_training_inputs,
    compile_policy_model,
    create_policy_model,
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


def train_drl_resource_manager(args: argparse.Namespace) -> None:
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    dataset = load_rm_policy_dataset(args.data)
    channel_energy = dataset["channel_energy"]
    y_mask = dataset["active_ut_mask"]
    y_power = dataset["per_ut_power"]
    ebno_db = dataset["ebno_db"]
    y_utility = dataset["oracle_utility"].reshape(-1, 1)

    _validate_shapes(
        channel_energy,
        y_mask,
        y_power,
        ebno_db,
        config_path=args.config,
    )

    x_state, normalization = build_policy_training_inputs(channel_energy, ebno_db)
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

    history = model.fit(
        x_state,
        {
            "schedule_output": y_mask,
            "power_output": y_power,
            "value_output": y_utility,
        },
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
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    train_drl_resource_manager(parser.parse_args())
