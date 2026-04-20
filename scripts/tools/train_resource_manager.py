from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.sim.config import load_config


def load_data(data_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to load the RM training parquet dataset.") from exc

    print(f"Loading dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    features = [np.stack(item).astype(np.float32) for item in df["channel_energy"].tolist()]
    x_data = np.stack(features)
    y_mask = np.array(df["active_ut_mask"].tolist(), dtype=np.float32)
    y_power = np.array(df["per_ut_power"].tolist(), dtype=np.float32)
    print(f"Data loaded: X={x_data.shape}, y_mask={y_mask.shape}, y_power={y_power.shape}")
    return x_data, y_mask, y_power


def create_model(input_shape: tuple[int, int], output_dim: int) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((1, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((1, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    mask_out = layers.Dense(output_dim, activation="sigmoid", name="mask_output")(x)
    power_out = layers.Dense(output_dim, activation="sigmoid", name="power_output")(x)
    return models.Model(inputs=inputs, outputs=[mask_out, power_out])


def _validate_shapes(
    x_data: np.ndarray,
    y_mask: np.ndarray,
    y_power: np.ndarray,
    *,
    config_path: str,
) -> None:
    app_config = load_config(config_path)
    expected_num_ut = app_config.system.num_ut
    expected_fft_size = app_config.system.fft_size
    if x_data.ndim != 3:
        raise ValueError(f"Expected feature tensor rank 3, got shape {x_data.shape}.")
    if tuple(x_data.shape[1:]) != (expected_num_ut, expected_fft_size):
        raise ValueError(
            f"Dataset feature shape {x_data.shape[1:]} does not match config "
            f"({expected_num_ut}, {expected_fft_size})."
        )
    if y_mask.shape[1] != expected_num_ut or y_power.shape[1] != expected_num_ut:
        raise ValueError(
            f"Target shapes {y_mask.shape} / {y_power.shape} do not match config num_ut={expected_num_ut}."
        )


def _write_metadata(output_path: str, data_path: str, num_ut: int, fft_size: int) -> None:
    metadata_path = os.path.splitext(output_path)[0] + ".metadata.json"
    payload = {
        "data_path": data_path,
        "num_ut": num_ut,
        "fft_size": fft_size,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Metadata saved to {metadata_path}")


def train_resource_manager(args: argparse.Namespace) -> None:
    x_data, y_mask, y_power = load_data(args.data)
    _validate_shapes(x_data, y_mask, y_power, config_path=args.config)

    input_shape = tuple(x_data.shape[1:])
    output_dim = int(y_mask.shape[1])
    model = create_model(input_shape, output_dim)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        loss={"mask_output": "binary_crossentropy", "power_output": "mse"},
        loss_weights={"mask_output": 1.0, "power_output": 0.5},
        metrics={"mask_output": "accuracy"},
    )
    model.summary()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.output,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
    )
    model.fit(
        x_data,
        [y_mask, y_power],
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=[checkpoint],
    )
    print(f"Training complete. Model saved to {args.output}")
    _write_metadata(args.output, args.data, input_shape[0], input_shape[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Factory6G resource-manager CNN.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config JSON file")
    parser.add_argument("--data", type=str, default="data/rm_training_data.parquet", help="Path to the training parquet file")
    parser.add_argument("--output", type=str, default="models/cnn_resource_manager.h5", help="Output model path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    train_resource_manager(parser.parse_args())
