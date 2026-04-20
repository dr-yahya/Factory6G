from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class ResourceManagerFeedback:
    h_hat: tf.Tensor
    err_var: tf.Tensor


@dataclass(frozen=True)
class BatchContext:
    batch_size: int
    ebno_db: float
    noise_variance: tf.Tensor
    h_freq: tf.Tensor
    probe_noise: tf.Tensor
    data_noise: tf.Tensor
    source_bits: tf.Tensor
    feedback: ResourceManagerFeedback | None
