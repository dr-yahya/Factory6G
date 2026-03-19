#!/usr/bin/env python3
"""Entry point for training the NeuralChannelEstimator inside Docker.

Usage:
    # Generate training data and train in one step (defaults):
    python train.py

    # Separate steps:
    python train.py --step generate --data data/channel_train.npz --batches 300
    python train.py --step train    --data data/channel_train.npz --output models/neural_estimator.keras

    # Full pipeline with custom paths:
    python train.py --data data/channel_train.npz --output models/neural_estimator.keras \\
        --batches 200 --epochs 20

Run via Docker Compose:
    docker compose run train-estimator
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NeuralChannelEstimator.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to simulation config JSON (default: config.json).",
    )
    parser.add_argument(
        "--step",
        choices=["generate", "train", "all"],
        default="all",
        help="Which step to run: 'generate' data, 'train' network, or 'all' (default).",
    )
    parser.add_argument(
        "--data",
        default="data/channel_train.npz",
        help="Path to the training dataset .npz file (default: data/channel_train.npz).",
    )
    parser.add_argument(
        "--output",
        default="models/neural_estimator.keras",
        help="Path to save the trained Keras model (default: models/neural_estimator.keras).",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=200,
        help="Number of simulation batches for dataset generation (default: 200).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for training (default: 256).",
    )
    parser.add_argument(
        "--ebno-min",
        type=float,
        default=0.0,
        help="Minimum Eb/No (dB) for dataset generation (default: 0.0).",
    )
    parser.add_argument(
        "--ebno-max",
        type=float,
        default=20.0,
        help="Maximum Eb/No (dB) for dataset generation (default: 20.0).",
    )
    return parser.parse_args()


def _load_flat_config(config_path: str) -> dict:
    """Load config.json and return a flat dict (merging all sections)."""
    with open(config_path) as f:
        raw = json.load(f)
    flat: dict = {}
    for section in raw.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    # Suppress TF noise before importing
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    try:
        flat_config = _load_flat_config(args.config)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logging.error("Failed to load config '%s': %s", args.config, exc)
        return 1

    # Ensure output directories exist
    Path(args.data).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    from src.training.train_estimator import generate_dataset, train

    if args.step in {"generate", "all"}:
        logging.info("=== Step 1: Generating dataset → %s ===", args.data)
        generate_dataset(
            config=flat_config,
            output_path=args.data,
            num_batches=args.batches,
            ebno_db_range=(args.ebno_min, args.ebno_max),
        )

    if args.step in {"train", "all"}:
        logging.info("=== Step 2: Training network → %s ===", args.output)
        train(
            dataset_path=args.data,
            model_output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
