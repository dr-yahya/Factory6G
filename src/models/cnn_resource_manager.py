
from __future__ import annotations

import tensorflow as tf
import numpy as np
from typing import Optional, Dict, Any, List

from ..components.config import SystemConfig
from .resource_manager import ResourceManager, ResourceDirectives

class CNNResourceManager(ResourceManager):
    """
    Resource manager that uses a CNN to predict resource allocation directives.
    
    This implementation loads a pre-trained Keras model to predict:
    - Active UT mask (User Scheduling)
    - Per-UT power allocation (Power Control)
    
    The input to the CNN is the estimated channel (H_hat) processed into
    channel energy profiles per user and subcarrier.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the CNN resource manager.
        
        Args:
            model_path: Path to the saved Keras model (.h5 or SavedModel).
                        If None, operates in passthrough mode (all active, full power).
            confidence_threshold: Threshold for binary decisions (e.g., active/inactive).
        """
        self.model = None
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        if model_path:
            try:
                # Load model roughly
                self.model = tf.keras.models.load_model(model_path, compile=False)
                print(f"Loaded CNN Resource Manager model from {model_path}")
            except Exception as e:
                print(f"Failed to load model from {model_path}: {e}")
                print("CNNResourceManager will use default fallback (all active).")
    
    def preprocess_channel(self, h_hat: tf.Tensor) -> tf.Tensor:
        """
        Preprocess channel estimates into CNN input features.
        
        Matches the feature extraction used in generate_dataset.py.
        
        Args:
            h_hat: Channel estimate [batch, num_rx, num_tx, num_streams, num_ofdm, fft_size]
            
        Returns:
            Features tensor [batch, num_tx, fft_size]
        """
        # Ensure complex64
        h_hat = tf.cast(h_hat, tf.complex64)
        
        # Power profile: sum(|h|^2) over Rx antennas (dim 1) 
        # H_hat: [batch, num_rx, num_rx_ant, num_tx, num_streams, num_ofdm, fft_size]
        # (1, 1, 32, 8, 2, 14, 512)
        
        power = tf.abs(h_hat)**2
        # Sum over Rx receivers (axis 1) -> [batch, 32, 8, 2, 14, 512]
        power = tf.reduce_sum(power, axis=1)
        # Sum over Rx antennas (axis 1) -> [batch, 8, 2, 14, 512]
        power = tf.reduce_sum(power, axis=1)
        # Sum over Streams (axis 2) -> [batch, 8, 14, 512]
        power = tf.reduce_sum(power, axis=2)
        # Avg over Time (axis 2) -> [batch, 8, 512]
        channel_energy = tf.reduce_mean(power, axis=2)
        
        return channel_energy

    def get_runtime_directives(
        self,
        config: SystemConfig,
        ebno_db: float,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> ResourceDirectives:
        """
        Return resource directives predicted by the CNN.
        """
        # Get h_hat from feedback (passed by Model.call/run_batch)
        h_hat = None
        if feedback and "h_hat" in feedback:
            h_hat = feedback["h_hat"]
        
        if self.model is None or h_hat is None:
            # Fallback to default: all active, full power
            return ResourceDirectives(
                active_ut_mask=[1] * config.num_ut,
                per_ut_power=[1.0] * config.num_ut,
                pilot_reuse_factor=1
            )
            
        # Preprocess: h_hat [batch, num_rx, num_tx, num_streams, num_ofdm, fft_size]
        features = self.preprocess_channel(h_hat) # [batch, num_tx, fft_size]
        
        # Inference
        # Depending on how it was trained, it might return a list [sched, power] or just sched
        predictions = self.model(features, training=False)
        
        if isinstance(predictions, list):
            sched_probs = predictions[0] # [batch, num_ut]
            power_alloc = predictions[1] # [batch, num_ut]
        elif isinstance(predictions, dict):
            sched_probs = predictions.get('scheduling', predictions.get('output_1'))
            power_alloc = predictions.get('power', predictions.get('output_2', tf.ones_like(sched_probs)))
        else:
            sched_probs = predictions
            power_alloc = tf.ones_like(predictions)
            
        # Convert to numpy for easier handling
        sched_probs_np = sched_probs.numpy()
        power_alloc_np = power_alloc.numpy()
        
        # Take the first sample from the batch to determine the mask for this simulation batch
        # (Since SystemConfig.active_ut_mask is not per-sample in Sionna's typical Model flow)
        idx = 0
        mask_pred = sched_probs_np[idx]
        power_pred = power_alloc_np[idx]
        
        mask = (mask_pred > self.confidence_threshold).astype(int).tolist()
        power = power_pred.tolist()
        
        # Ensure it's the right length
        if len(mask) != config.num_ut:
            mask = mask[:config.num_ut]
        if len(power) != config.num_ut:
            power = power[:config.num_ut]
            
        return ResourceDirectives(
            active_ut_mask=mask,
            per_ut_power=power,
            pilot_reuse_factor=1
        )
