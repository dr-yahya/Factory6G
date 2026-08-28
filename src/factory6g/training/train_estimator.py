"""Training utilities for the NeuralChannelEstimator.

Usage (inside Docker via docker-compose):

    # Generate training data then train (default output paths):
    docker compose run train-estimator

    # Override paths:
    docker compose run train-estimator \
        --data data/channel_train.npz \
        --output models/neural_estimator.keras \
        --batches 500 --epochs 30 \
        --channel-types rayleigh rician tr38901

How it works:
    1. ``generate_dataset`` runs the simulation (LS estimator only) across
       multiple channel types using a curriculum schedule.  It collects
       triplets of (h_ls, h_residual, ebno_db) where h_residual = h_true - h_ls.
       Learning to predict the *correction* is much easier than predicting the
       full channel, because the corrections are small at high SNR and the
       network only needs to learn where LS is wrong.
    2. ``train`` builds a 3-input-channel Conv2D network (re, im, snr_map),
       trains it to predict h_residual with MSE loss, and saves the model.
    3. At inference time the network output (delta) is added back to h_ls.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# SNR normalisation constants — must match neural_estimator.py
EBNO_MIN_DB: float = 0.0
EBNO_MAX_DB: float = 20.0


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    config: dict,
    output_path: str,
    num_batches: int = 200,
    ebno_db_range: tuple[float, float] = (0.0, 20.0),
    channel_types: Sequence[str] = ("tr38901",),
) -> None:
    """Run simulation batches and save (h_ls, h_residual, ebno_db) to .npz.

    Curriculum schedule (when multiple channel_types are given):
      - First third of batches:  easiest channel only (channel_types[0])
      - Middle third:            first two channels equally
      - Final third:             all channels equally

    This lets the network learn basic denoising on Rayleigh before tackling
    the harder multipath of TR 38.901.

    Args:
        config: Flat system config dict (same format used by Model).
        output_path: Path to write the .npz dataset.
        num_batches: Total number of simulation batches to collect.
        ebno_db_range: (min, max) Eb/No in dB, sampled uniformly per batch.
        channel_types: Ordered list of channel model names for curriculum.
                       Typical: ["rayleigh", "rician", "tr38901"].
    """
    import tensorflow as tf  # noqa: F401 — ensure TF is initialised
    from factory6g.models.model import Model

    channel_types = list(channel_types)
    batch_size = int(config.get("batch_size", 64))

    h_ls_list: list[np.ndarray] = []
    h_residual_list: list[np.ndarray] = []
    ebno_list: list[float] = []

    # One Model instance per channel type (re-created only when type changes)
    model_cache: dict[str, Model] = {}

    def _get_model(ch_type: str) -> Model:
        if ch_type not in model_cache:
            cfg = copy.deepcopy(config)
            cfg["channel_model_type"] = ch_type
            model_cache[ch_type] = Model(cfg, estimator_type="ls")
            logger.info("Created LS model for channel_model_type='%s'.", ch_type)
        return model_cache[ch_type]

    def _curriculum_type(batch_idx: int) -> str:
        """Return channel type according to curriculum stage."""
        n = len(channel_types)
        if n == 1:
            return channel_types[0]
        stage = int(batch_idx / num_batches * n)
        stage = min(stage, n - 1)
        pool = channel_types[: stage + 1]
        return pool[np.random.randint(len(pool))]

    def _flatten(arr: np.ndarray) -> np.ndarray:
        """[batch, rx, tx, streams, ofdm_syms, fft_size] → [N, ofdm_syms, fft_size]."""
        s = arr.shape
        return arr.reshape(-1, s[-2], s[-1])

    # Flush accumulated arrays to disk every FLUSH_EVERY batches to avoid OOM.
    FLUSH_EVERY = 50
    flush_files: list[str] = []
    flush_idx = 0

    def _flush(h_ls_buf, h_res_buf, ebno_buf):
        nonlocal flush_idx
        chunk_path = f"{output_path}.chunk{flush_idx}.npz"
        np.savez_compressed(
            chunk_path,
            h_ls=np.concatenate(h_ls_buf, axis=0),
            h_residual=np.concatenate(h_res_buf, axis=0),
            ebno_db=np.array(ebno_buf, dtype=np.float32),
        )
        flush_files.append(chunk_path)
        flush_idx += 1

    logger.info(
        "Generating %d batches | channel types: %s | Eb/No: %.1f–%.1f dB",
        num_batches, channel_types, *ebno_db_range,
    )

    for i in range(num_batches):
        ch_type = _curriculum_type(i)
        ebno_db = float(np.random.uniform(*ebno_db_range))

        model = _get_model(ch_type)
        ctx = model.prepare_batch_context(batch_size, ebno_db, include_feedback=False)
        result = model.run_batch(ctx, include_details=True)

        h_true = result["channel"]      # complex64
        h_ls = result["channel_hat"]    # complex64 (LS estimate)

        h_ls_flat = _flatten(h_ls)
        h_residual_flat = _flatten(h_true) - h_ls_flat  # correction to learn

        h_ls_list.append(h_ls_flat)
        h_residual_list.append(h_residual_flat)
        ebno_list.extend([ebno_db] * h_ls_flat.shape[0])

        if (i + 1) % FLUSH_EVERY == 0:
            logger.info("  %d / %d batches done — flushing chunk to disk", i + 1, num_batches)
            _flush(h_ls_list, h_residual_list, ebno_list)
            h_ls_list.clear()
            h_residual_list.clear()
            ebno_list.clear()

    # Flush any remaining batches
    if h_ls_list:
        _flush(h_ls_list, h_residual_list, ebno_list)

    # Skip the in-memory merge — write a manifest so train() can stream chunks.
    manifest_path = output_path + ".manifest.json"
    # Count total samples from chunk headers (no full load needed)
    total_samples = 0
    for p in flush_files:
        with np.load(p) as c:
            total_samples += c["h_ls"].shape[0]
    manifest = {"chunks": flush_files, "total_samples": total_samples}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "Dataset manifest saved to '%s' (%d samples across %d chunks).",
        manifest_path, total_samples, len(flush_files),
    )


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

def build_model(num_ofdm_symbols: int, fft_size: int) -> "tf.keras.Model":
    """Build SNR-conditioned residual Conv2D estimator network.

    Input shape:  [N, num_ofdm_symbols, fft_size, 3]
                  channels: re(h_ls), im(h_ls), snr_map (scalar broadcast)
    Output shape: [N, num_ofdm_symbols, fft_size, 2]
                  channels: re(delta), im(delta)   where delta = h_true - h_ls

    At inference: h_hat = h_ls + complex(re(delta), im(delta))
    """
    import tensorflow as tf

    inputs = tf.keras.layers.Input(
        shape=(num_ofdm_symbols, fft_size, 3), name="h_ls_snr"
    )
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
    """Stream chunk files, train the residual network, and save the Keras model.

    Accepts either:
      - A manifest JSON written by ``generate_dataset`` (dataset_path + ".manifest.json"), or
      - A direct .npz file (legacy single-file format).

    Chunks are loaded one at a time to stay within memory limits.

    Args:
        dataset_path: Base path passed to ``generate_dataset`` (e.g. data/channel_train.npz).
        model_output_path: Where to save the trained Keras model.
        epochs: Number of training epochs.
        batch_size: Mini-batch size for gradient descent.
    """
    import json
    import tensorflow as tf

    manifest_path = dataset_path + ".manifest.json"

    # Discover chunk files
    if os.path.exists(manifest_path):
        logger.info("Loading manifest from '%s'...", manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)
        chunk_paths = manifest["chunks"]
        total_samples = manifest["total_samples"]
    elif os.path.exists(dataset_path):
        logger.info("No manifest found — loading single file '%s'...", dataset_path)
        chunk_paths = [dataset_path]
        with np.load(dataset_path) as d:
            total_samples = d["h_ls"].shape[0]
    else:
        raise FileNotFoundError(
            f"Neither '{manifest_path}' nor '{dataset_path}' found. "
            "Run train.py --step generate first."
        )

    # Peek at first chunk to get grid dimensions
    # Chunks may be complex64 [N, ofdm, fft] or float32 [N, ofdm, fft, 2]
    with np.load(chunk_paths[0]) as first:
        h_ls_peek = first["h_ls"]
        if h_ls_peek.ndim == 3:
            # complex64 format written by generate_dataset
            _, num_ofdm_symbols, fft_size = h_ls_peek.shape
            chunks_are_complex = True
        else:
            _, num_ofdm_symbols, fft_size, _ = h_ls_peek.shape
            chunks_are_complex = False
    logger.info(
        "Training on %d samples, grid %d×%d, %d chunk(s), complex=%s.",
        total_samples, num_ofdm_symbols, fft_size, len(chunk_paths), chunks_are_complex,
    )

    def _to_ri(arr: np.ndarray) -> np.ndarray:
        """complex64 [N, ofdm, fft] → float32 [N, ofdm, fft, 2]."""
        return np.stack([arr.real, arr.imag], axis=-1).astype(np.float32)

    def _chunk_generator():
        """Yield (x, y) sample pairs by streaming one chunk at a time."""
        for path in chunk_paths:
            data = np.load(path)
            h_ls_raw = data["h_ls"]
            h_res_raw = data["h_residual"]
            ebno_db = data["ebno_db"]  # [N] float32

            # Normalise to float32 re/im if chunks are complex
            h_ls_ri = _to_ri(h_ls_raw) if chunks_are_complex else h_ls_raw.astype(np.float32)
            h_res_ri = _to_ri(h_res_raw) if chunks_are_complex else h_res_raw.astype(np.float32)

            ebno_norm = np.clip(
                (ebno_db - EBNO_MIN_DB) / (EBNO_MAX_DB - EBNO_MIN_DB), 0.0, 1.0
            )
            snr_map = np.tile(
                ebno_norm[:, None, None, None],
                (1, num_ofdm_symbols, fft_size, 1),
            ).astype(np.float32)

            x = np.concatenate([h_ls_ri, snr_map], axis=-1)  # [N, ofdm, fft, 3]

            for i in range(len(x)):
                yield x[i], h_res_ri[i]

    sig_x = tf.TensorSpec(shape=(num_ofdm_symbols, fft_size, 3), dtype=tf.float32)
    sig_y = tf.TensorSpec(shape=(num_ofdm_symbols, fft_size, 2), dtype=tf.float32)

    # Hold out last 10 % of samples (last chunk portion) for validation
    val_samples = max(1, int(total_samples * 0.1))
    train_samples = total_samples - val_samples
    steps_per_epoch = max(1, train_samples // batch_size)
    val_steps = max(1, val_samples // batch_size)

    ds_full = tf.data.Dataset.from_generator(
        _chunk_generator, output_signature=(sig_x, sig_y)
    )
    ds_train = ds_full.take(train_samples).shuffle(2048).batch(batch_size).prefetch(2)
    ds_val = ds_full.skip(train_samples).batch(batch_size).prefetch(2)

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
        ds_train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_val,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    net.save(model_output_path)
    logger.info("Model saved to '%s'.", model_output_path)
