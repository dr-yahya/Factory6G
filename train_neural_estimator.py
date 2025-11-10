#!/usr/bin/env python3
"""Train the neural channel estimator used in the physical-layer model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from tqdm import trange

from src.components.config import SystemConfig
from src.components.antenna import AntennaConfig
from src.components.transmitter import Transmitter
from src.components.channel import ChannelModel
from src.components.receiver import Receiver
from src.components.estimators import NeuralChannelEstimator, stack_complex
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.utils import ebnodb2no

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def build_system(config: SystemConfig) -> Tuple[
    Transmitter,
    ChannelModel,
    Receiver,
    ResourceGrid,
]:
    rg = ResourceGrid(
        num_ofdm_symbols=config.num_ofdm_symbols,
        fft_size=config.fft_size,
        subcarrier_spacing=config.subcarrier_spacing,
        num_tx=config.num_tx,
        num_streams_per_tx=config.num_streams_per_tx,
        cyclic_prefix_length=config.cyclic_prefix_length,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=config.pilot_ofdm_symbol_indices,
    )
    sm = StreamManagement(config.get_rx_tx_association(), config.num_streams_per_tx)
    ant = AntennaConfig(config)
    tx = Transmitter(config, rg)
    ch = ChannelModel(config, ant, rg)
    rx = Receiver(config, rg, sm, tx._encoder, perfect_csi=False)
    return tx, ch, rx, rg


def generate_batch(
    transmitter: Transmitter,
    channel: ChannelModel,
    receiver: Receiver,
    resource_grid: ResourceGrid,
    batch_size: int,
    ebno_db: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    channel.set_topology(batch_size)
    noise_var = ebnodb2no(
        tf.constant(ebno_db, dtype=tf.float32),
        transmitter.config.num_bits_per_symbol,
        transmitter.config.coderate,
        resource_grid,
    )
    x_rg, _ = transmitter.call(batch_size)
    y, h_true = channel(x_rg, noise_var)

    # Baseline LS estimate
    h_ls, _ = receiver._channel_estimator(y, noise_var)
    h_true_proc = receiver._remove_nulled_subcarriers(h_true)

    features = stack_complex(tf.math.real(h_ls), tf.math.imag(h_ls))
    targets = stack_complex(tf.math.real(h_true_proc), tf.math.imag(h_true_proc))
    return features, targets


def prepare_dataset(
    transmitter: Transmitter,
    channel: ChannelModel,
    receiver: Receiver,
    resource_grid: ResourceGrid,
    num_batches: int,
    batch_size: int,
    ebno_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    features_list = []
    targets_list = []
    ebno_min, ebno_max = ebno_range

    for _ in trange(num_batches, desc="Generating training data"):
        ebno_db = np.random.uniform(ebno_min, ebno_max)
        feats, targs = generate_batch(transmitter, channel, receiver, resource_grid, batch_size, ebno_db)
        # Flatten last dimensions to (N, 2)
        feats = tf.reshape(feats, [-1, 2])
        targs = tf.reshape(targs, [-1, 2])
        features_list.append(feats.numpy())
        targets_list.append(targs.numpy())

    features = np.concatenate(features_list, axis=0).astype(np.float32)
    targets = np.concatenate(targets_list, axis=0).astype(np.float32)
    return features, targets


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_neural_estimator(args: argparse.Namespace) -> Path:
    config = SystemConfig(scenario=args.scenario)
    transmitter, channel, receiver, resource_grid = build_system(config)

    features, targets = prepare_dataset(
        transmitter,
        channel,
        receiver,
        resource_grid,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        ebno_range=(args.ebno_min, args.ebno_max),
    )

    estimator = NeuralChannelEstimator(
        config,
        resource_grid,
        hidden_units=args.hidden_units,
    )

    estimator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )

    estimator.fit(
        features,
        targets,
        batch_size=args.train_batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        verbose=1,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    estimator.save_weights(output_path)
    print(f"✓ Saved neural estimator weights to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', type=str, default='umi', choices=['umi', 'uma', 'rma'], help='Channel scenario (default: umi).')
    parser.add_argument('--batch-size', type=int, default=16, help='Number of channel realizations per batch (default: 16).')
    parser.add_argument('--num-batches', type=int, default=200, help='Number of batches to generate for training (default: 200).')
    parser.add_argument('--ebno-min', type=float, default=-3.0, help='Minimum Eb/No in dB for training data (default: -3).')
    parser.add_argument('--ebno-max', type=float, default=15.0, help='Maximum Eb/No in dB for training data (default: 15).')
    parser.add_argument('--hidden-units', type=int, nargs='+', default=[32, 32], help='Hidden layer sizes for the neural estimator (default: 32 32).')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate for Adam optimizer (default: 1e-3).')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs (default: 5).')
    parser.add_argument('--train-batch-size', type=int, default=4096, help='Batch size used for neural network training (default: 4096).')
    parser.add_argument('--validation-split', type=float, default=0.1, help='Validation split ratio (default: 0.1).')
    parser.add_argument('--output', type=str, default='artifacts/neural_channel_estimator.weights.h5', help='Path to save trained weights (must end with .weights.h5).')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.output.endswith('.weights.h5'):
        args.output = f"{args.output}.weights.h5"
    Path('artifacts').mkdir(exist_ok=True)
    train_neural_estimator(args)


if __name__ == "__main__":
    main()
