#!/usr/bin/env python3
"""Train the neural channel estimator used in the physical-layer model.

This script trains a neural network-based channel estimator that refines
least squares (LS) channel estimates. The neural estimator learns to reduce
estimation error by exploiting channel statistics and noise characteristics.

Theory:
    Neural Channel Estimation Training:
    
    1. Dataset Generation:
       - Generate channel realizations using 3GPP TR 38.901 channel model
       - Transmit signals through channel with various Eb/No values
       - Compute LS estimates: Ĥ_LS = LS_Estimator(Y, X_pilot)
       - Extract true channels: H_true (from channel model)
       - Create training pairs: (Ĥ_LS, H_true)
       
    2. Training Process:
       - Input: LS estimates Ĥ_LS (real and imaginary parts)
       - Target: True channels H_true (real and imaginary parts)
       - Loss: MSE between predicted and true channels
       - Optimizer: Adam with learning rate scheduling
       - Regularization: Optional dropout, weight decay
       
    3. Training Objective:
       min_θ E[|H_true - f_θ(Ĥ_LS)|²]
       
       where f_θ is the neural network parameterized by θ.
       
    4. Data Augmentation:
       - Vary Eb/No values during training (ebno_min to ebno_max)
       - Generate diverse channel realizations (different topologies)
       - Expose model to various noise levels and channel conditions
       
    5. Validation:
       - Split data into training and validation sets
       - Monitor validation loss to prevent overfitting
       - Early stopping if validation loss stops improving
       
    6. Model Evaluation:
       - Test on held-out test set
       - Compare with LS estimator baseline
       - Measure improvement in estimation error

References:
    - Wen et al., "Deep Learning for Massive MIMO Channel State Acquisition"
    - Soltani et al., "Deep Learning-Based Channel Estimation"
"""

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
    """
    Generate a batch of training data for neural channel estimator.
    
    Creates training examples by:
    1. Generating random channel realizations
    2. Transmitting signals through the channel
    3. Computing LS channel estimates
    4. Extracting true channel responses
    
    Theory:
        Training data generation:
        
        1. Channel Realization:
           - Generate random UT positions and orientations
           - Compute channel response H_true using 3GPP TR 38.901 model
           - Channel varies with topology (distance, angles, etc.)
           
        2. Signal Transmission:
           - Generate random information bits
           - Encode, modulate, and map to resource grid
           - Transmit through channel: Y = H·X + N
           - Add AWGN noise with variance σ² = N₀
           
        3. LS Estimation:
           - Estimate channel from received pilots: Ĥ_LS = Y_pilot / X_pilot
           - LS estimate is noisy: Ĥ_LS = H_true + ε, where ε ~ CN(0, σ²_LS)
           
        4. Training Pair:
           - Input (features): Ĥ_LS (LS estimate)
           - Target (labels): H_true (true channel)
           - Goal: Learn mapping f: Ĥ_LS → H_true to reduce estimation error
        
    Args:
        transmitter: Transmitter component for generating signals
        channel: Channel model for generating channel realizations
        receiver: Receiver component (contains LS channel estimator)
        resource_grid: OFDM resource grid defining time-frequency structure
        batch_size: Number of channel realizations per batch
        ebno_db: Eb/No value in dB for this batch
            Varying Eb/No during training improves generalization
            
    Returns:
        Tuple of:
        - features: LS channel estimates [batch_size, ..., 2]
            Last dimension is [Re(Ĥ_LS), Im(Ĥ_LS)]
        - targets: True channel responses [batch_size, ..., 2]
            Last dimension is [Re(H_true), Im(H_true)]
    """
    # Generate new channel topology for this batch
    channel.set_topology(batch_size)
    
    # Compute noise variance from Eb/No
    noise_var = ebnodb2no(
        tf.constant(ebno_db, dtype=tf.float32),
        transmitter.config.num_bits_per_symbol,
        transmitter.config.coderate,
        resource_grid,
    )
    
    # Transmit signals through channel
    x_rg, _ = transmitter.call(batch_size)
    y, h_true = channel(x_rg, noise_var)

    # Compute LS channel estimate
    h_ls, _ = receiver._channel_estimator(y, noise_var)
    
    # Process true channel (remove nulled subcarriers to match LS estimate)
    h_true_proc = receiver._remove_nulled_subcarriers(h_true)

    # Prepare training data: stack real and imaginary parts
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
    """
    Train the neural channel estimator.
    
    Performs end-to-end training of the neural channel estimator by:
    1. Generating training dataset (LS estimates and true channels)
    2. Building neural network model
    3. Training with MSE loss and Adam optimizer
    4. Saving trained weights to file
    
    Theory:
        Training Process:
        
        1. Dataset Preparation:
           - Generate diverse channel realizations
           - Vary Eb/No values (ebno_min to ebno_max)
           - Create (Ĥ_LS, H_true) training pairs
           
        2. Model Training:
           - Loss function: L(θ) = E[|H_true - f_θ(Ĥ_LS)|²]
           - Optimizer: Adam with learning rate α
           - Update rule: θ ← θ - α·∇_θ L(θ)
           - Batch training: Process multiple examples simultaneously
           
        3. Validation:
           - Split data: (1 - validation_split) for training, validation_split for validation
           - Monitor validation loss to detect overfitting
           - Early stopping: Stop if validation loss stops improving
           
        4. Model Evaluation:
           - Test on held-out test set
           - Measure estimation error reduction vs LS baseline
           - Compare performance across different Eb/No values
        
    Args:
        args: Command-line arguments containing training configuration:
            - scenario: Channel scenario ("umi", "uma", "rma")
            - num_batches: Number of batches to generate
            - batch_size: Channel realizations per batch
            - ebno_min, ebno_max: Eb/No range for training
            - hidden_units: Neural network architecture
            - learning_rate: Adam optimizer learning rate
            - epochs: Number of training epochs
            - train_batch_size: Batch size for neural network training
            - validation_split: Fraction of data for validation
            - output: Path to save trained weights
            
    Returns:
        Path to saved weights file
    """
    # Initialize system configuration
    config = SystemConfig(scenario=args.scenario)
    
    # Build system components
    transmitter, channel, receiver, resource_grid = build_system(config)

    # Generate training dataset
    features, targets = prepare_dataset(
        transmitter,
        channel,
        receiver,
        resource_grid,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        ebno_range=(args.ebno_min, args.ebno_max),
    )

    # Create neural channel estimator
    estimator = NeuralChannelEstimator(
        config,
        resource_grid,
        hidden_units=args.hidden_units,
    )

    # Compile model with optimizer and loss function
    estimator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),  # MSE loss for regression
        metrics=[tf.keras.metrics.MeanSquaredError()],  # Track MSE during training
    )

    # Train the model
    estimator.fit(
        features,
        targets,
        batch_size=args.train_batch_size,  # Batch size for neural network training
        epochs=args.epochs,  # Number of training epochs
        validation_split=args.validation_split,  # Fraction of data for validation
        verbose=1,  # Print training progress
    )

    # Save trained weights
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
