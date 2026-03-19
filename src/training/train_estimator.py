"""Training utilities for the NeuralChannelEstimator.

Usage (inside Docker via docker-compose):

    # Generate training data then train (default output paths):
    docker compose run train-estimator

    # Override paths:
    docker compose run train-estimator \
        --data data/channel_train.npz \
        --output models/neural_estimator.keras \
        --batches 500 --epochs 30

How it works:
    1. ``generate_dataset`` runs the simulation (LS estimator only) to collect
       pairs of (LS estimate, true channel) for many random SNR/batch samples.
    2. ``train`` loads the dataset, builds the Conv2D network, and trains with
       MSE loss. The trained model is saved in Keras format.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    config: dict,
    output_path: str,
    num_batches: int = 200,
    ebno_db_range: tuple[float, float] = (0.0, 20.0),
) -> None:
    """Run simulation batches and save (h_ls, h_true) pairs to a .npz file.

    Args:
        config: Flat system config dict (same format used by Model).
        output_path: Path to write the .npz dataset.
        num_batches: Number of simulation batches to collect.
        ebno_db_range: (min, max) Eb/No range to sample uniformly.
    """
    import tensorflow as tf
    from src.models.model import Model

    model = Model(config, estimator_type="ls")
    batch_size = int(config.get("batch_size", 64))

    h_ls_list: list[np.ndarray] = []
    h_true_list: list[np.ndarray] = []

    logger.info("Generating %d batches of channel data...", num_batches)
    for i in range(num_batches):
        ebno_db = float(np.random.uniform(*ebno_db_range))
        ctx = model.prepare_batch_context(batch_size, ebno_db, include_feedback=False)
        result = model.run_batch(ctx, include_details=True)

        # h_true: true channel, h_ls approximated by channel_hat (LS estimate)
        # shape: [batch, num_rx, num_tx, num_streams, num_ofdm_symbols, fft_size]
        h_true = result["channel"]        # numpy, complex64
        h_ls = result["channel_hat"]      # numpy, complex64 (LS estimate)

        # Flatten batch/rx/tx/stream dims → [N, num_ofdm_symbols, fft_size]
        def _flatten(arr: np.ndarray) -> np.ndarray:
            # arr shape: [batch, num_rx, num_tx, num_streams, ofdm_syms, fft_size]
            s = arr.shape
            return arr.reshape(-1, s[-2], s[-1])

        h_ls_list.append(_flatten(h_ls))
        h_true_list.append(_flatten(h_true))

        if (i + 1) % 50 == 0:
            logger.info("  %d / %d batches done", i + 1, num_batches)

    h_ls_all = np.concatenate(h_ls_list, axis=0)    # [N, ofdm_syms, fft_size], complex64
    h_true_all = np.concatenate(h_true_list, axis=0)

    # Split complex → real/imag for Keras: [N, ofdm_syms, fft_size, 2]
    def _to_ri(arr: np.ndarray) -> np.ndarray:
        return np.stack([arr.real, arr.imag], axis=-1).astype(np.float32)

    np.savez_compressed(
        output_path,
        h_ls=_to_ri(h_ls_all),
        h_true=_to_ri(h_true_all),
    )
    logger.info("Dataset saved to '%s' (%d samples).", output_path, len(h_ls_all))


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

def build_model(num_ofdm_symbols: int, fft_size: int):
    """Build the SRCNN-style Conv2D estimator network."""
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(num_ofdm_symbols, fft_size, 2), name="h_ls_ri")
    x = tf.keras.layers.Conv2D(64, (9, 9), padding="same", activation="relu", name="conv1")(inputs)
    x = tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation="relu", name="conv2")(x)
    outputs = tf.keras.layers.Conv2D(2, (5, 5), padding="same", activation="linear", name="conv3")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="neural_channel_estimator")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    dataset_path: str,
    model_output_path: str,
    epochs: int = 20,
    batch_size: int = 256,
) -> None:
    """Load dataset, train the network, and save the Keras model.

    Args:
        dataset_path: Path to .npz file produced by ``generate_dataset``.
        model_output_path: Where to save the trained Keras model.
        epochs: Number of training epochs.
        batch_size: Mini-batch size for gradient descent.
    """
    import tensorflow as tf

    logger.info("Loading dataset from '%s'...", dataset_path)
    data = np.load(dataset_path)
    h_ls = data["h_ls"]    # [N, ofdm_syms, fft_size, 2], float32
    h_true = data["h_true"]

    _, num_ofdm_symbols, fft_size, _ = h_ls.shape
    logger.info(
        "Dataset loaded: %d samples, grid %d×%d.", len(h_ls), num_ofdm_symbols, fft_size
    )

    net = build_model(num_ofdm_symbols, fft_size)
    net.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    net.summary(print_fn=logger.info)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        ),
    ]

    net.fit(
        h_ls,
        h_true,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=True,
        callbacks=callbacks,
    )

    net.save(model_output_path)
    logger.info("Model saved to '%s'.", model_output_path)
