"""Neural network-based channel estimator for the OFDM MIMO receiver.

This module implements a machine learning-based channel estimator that refines
least squares (LS) estimates using a neural network. The estimator combines
traditional signal processing (LS estimation) with learned refinements to
improve channel estimation accuracy.

Theory:
    Machine Learning for Channel Estimation:
    
    1. Problem Formulation:
       - Traditional LS estimator: Ĥ_LS = Y_pilot / X_pilot
       - LS estimate is noisy: Ĥ_LS = H + ε_LS, where ε_LS ~ CN(0, σ²_LS)
       - Goal: Learn a function f that reduces estimation error: Ĥ_ML = f(Ĥ_LS)
       
    2. Neural Network Approach:
       - Point-wise processing: Each resource element processed independently
       - Input: Real and imaginary parts of LS estimate: [Re(Ĥ_LS), Im(Ĥ_LS)]
       - Output: Refined real and imaginary parts: [Re(Ĥ_ML), Im(Ĥ_ML)]
       - Architecture: Fully connected layers with nonlinear activations
       - Training: Supervised learning with true channel as target
       
    3. Training Objective:
       - Loss function: MSE between predicted and true channel
       - L(θ) = E[|H - f_θ(Ĥ_LS)|²]
       - Minimizes estimation error over training distribution
       - Gradient descent: θ ← θ - α·∇_θ L(θ)
       
    4. Advantages:
       - Can learn complex nonlinear relationships
       - Adapts to channel statistics and noise characteristics
       - Lightweight: Point-wise processing keeps model small
       - Differentiable: Enables end-to-end training
       
    5. Limitations:
       - Requires training data (channel realizations)
       - Generalization to unseen scenarios may be limited
       - Computational overhead (though minimal for point-wise processing)
       
    6. Hybrid Approach:
       - Combines LS (data-efficient) with neural refinement (learned)
       - LS provides good initial estimate
       - Neural network refines estimate to reduce error
       - Best of both worlds: Interpretability + performance

References:
    - Wen et al., "Deep Learning for Massive MIMO Channel State Acquisition"
    - Soltani et al., "Deep Learning-Based Channel Estimation"
    - Ye et al., "Channel Estimation for OFDM Systems with Neural Networks"
"""

from __future__ import annotations

import tensorflow as tf
from pathlib import Path
from typing import Iterable, Optional, Sequence

from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid

from ..config import SystemConfig


class NeuralChannelEstimator(Block):
    """
    Channel estimator that refines LS estimates using a small neural network.

    This estimator uses a two-stage approach:
    1. Initial LS estimation: Provides baseline channel estimate
    2. Neural refinement: Learns to correct LS estimation errors
    
    The neural network operates point-wise on each resource element, processing
    the real and imaginary parts of the LS estimate to produce a refined estimate.
    This approach keeps the model lightweight while enabling learned improvements
    over traditional estimation methods.
    
    Theory:
        The estimator can be described as:
        
        Ĥ_LS = LS_Estimator(Y, X_pilot)  # Initial LS estimate
        Ĥ_ML = f_θ(Ĥ_LS)                 # Neural refinement
        
        where f_θ is a neural network parameterized by θ.
        
        The network architecture:
        - Input: [Re(Ĥ_LS), Im(Ĥ_LS)] (2 features per resource element)
        - Hidden layers: Fully connected with ReLU activation
        - Output: [Re(Ĥ_ML), Im(Ĥ_ML)] (2 features per resource element)
        
        Training:
        - Dataset: (Ĥ_LS, H_true) pairs from channel realizations
        - Loss: L(θ) = E[|H_true - f_θ(Ĥ_LS)|²]
        - Optimizer: Adam with learning rate scheduling
        - Regularization: Dropout, weight decay (optional)
        
        Advantages:
        - Learns channel-dependent corrections
        - Reduces estimation error compared to LS alone
        - Lightweight: Point-wise processing
        - Differentiable: Can be used in end-to-end systems
    """

    def __init__(
        self,
        config: SystemConfig,
        resource_grid: ResourceGrid,
        hidden_units: Sequence[int] | None = None,
        activation: str = "relu",
        weights_path: Optional[str] = None,
    ) -> None:
        """
        Initialize neural channel estimator.
        
        Creates a neural channel estimator that refines LS estimates using
        a point-wise neural network. The network processes each resource element
        independently, taking the real and imaginary parts of the LS estimate
        as input and producing refined real and imaginary parts as output.
        
        Theory:
            The estimator architecture:
            
            Input: [Re(Ĥ_LS), Im(Ĥ_LS)] (2D feature vector per RE)
            ↓
            Hidden Layer 1: Dense(hidden_units[0]) + Activation
            ↓
            Hidden Layer 2: Dense(hidden_units[1]) + Activation
            ↓
            ...
            ↓
            Output: Dense(2) → [Re(Ĥ_ML), Im(Ĥ_ML)]
            
            The network learns a mapping:
            f_θ: R² → R², where f_θ([Re(Ĥ_LS), Im(Ĥ_LS)]) = [Re(Ĥ_ML), Im(Ĥ_ML)]
            
            Training objective:
            min_θ E[|H_true - f_θ(Ĥ_LS)|²]
            
            The network is trained to minimize the mean squared error between
            the refined estimate and the true channel over a training dataset.
        
        Args:
            config: System configuration parameters.
            resource_grid: OFDM resource grid defining the time-frequency structure.
            hidden_units: List of hidden layer sizes. Default [32, 32] means two
                hidden layers with 32 units each. Larger networks can model more
                complex relationships but require more training data and computation.
            activation: Activation function for hidden layers. Default "relu" (rectified
                linear unit). Other options: "tanh", "sigmoid", "elu", etc.
            weights_path: Path to pre-trained weights file. If provided and the file
                exists, weights are loaded automatically. Weights should be saved in
                Keras HDF5 format (.weights.h5).
        """
        super().__init__()
        self.config = config
        self.resource_grid = resource_grid
        self.hidden_units = list(hidden_units or [32, 32])
        self.activation = activation
        self.weights_path = Path(weights_path) if weights_path else None

        # Base estimator (LS) used for initial estimation
        # LS provides a good starting point that the neural network can refine
        self._base_estimator = LSChannelEstimator(
            resource_grid,
            interpolation_type="nn",  # Nearest neighbor interpolation
        )

        # Build point-wise neural network
        # Processes each resource element independently
        self.model = self._build_network()

        self._weights_loaded = False
        if self.weights_path and self.weights_path.exists():
            self.load_weights(self.weights_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def call(self, y: tf.Tensor, noise_variance: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Estimate the channel response using neural refinement.

        Performs channel estimation in two stages:
        1. Initial LS estimation from pilot symbols
        2. Neural network refinement of LS estimate
        
        The neural network processes each resource element independently,
        taking the real and imaginary parts of the LS estimate as input
        and producing refined real and imaginary parts as output.
        
        Theory:
            The estimation process:
            
            1. LS Estimation:
               Ĥ_LS = LS_Estimator(Y, X_pilot)
               
            2. Feature Extraction:
               features = [Re(Ĥ_LS), Im(Ĥ_LS)]
               
            3. Neural Refinement:
               [Re(Ĥ_ML), Im(Ĥ_ML)] = f_θ(features)
               
            4. Complex Reconstruction:
               Ĥ_ML = Re(Ĥ_ML) + j·Im(Ĥ_ML)
            
            The neural network learns to correct systematic errors in the
            LS estimate, such as:
            - Noise amplification in low-SNR regions
            - Interpolation artifacts
            - Channel-dependent biases
            
            Error variance:
            The error variance is currently set to the LS estimator's error
            variance. In practice, the neural network may reduce the actual
            error, but computing the true error variance would require
            knowledge of the true channel (not available at inference time).
        
        Args:
            y: Received resource grid (complex tensor)
                Shape: [batch_size, num_rx, num_streams, num_ofdm_symbols, fft_size]
                Contains received signal including pilots and data
            noise_variance: Noise variance for each resource element
                Used by LS estimator to compute error variance
                Shape: scalar or same as resource grid
                
        Returns:
            Tuple of:
            - h_pred: Refined channel estimate
                Shape: [batch_size, num_rx, num_tx, num_streams, num_ofdm_symbols, fft_size]
                Neural network refined estimate of channel frequency response
            - err_var: Channel estimation error variance
                Currently set to LS estimator's error variance
                Shape: same as h_pred, or scalar if uniform
        """
        # Stage 1: Initial LS estimation
        h_ls, err_var = self._base_estimator(y, noise_variance)

        # Stage 2: Prepare neural network input
        # Stack real and imaginary parts along last dimension
        # Shape: [..., 2] where last dimension is [Re, Im]
        features = tf.stack(
            [tf.math.real(h_ls), tf.math.imag(h_ls)],
            axis=-1,
        )

        # Stage 3: Neural network refinement
        # Process each resource element independently
        refined = self.model(features)
        
        # Stage 4: Extract refined real and imaginary parts
        real_part = refined[..., 0]
        imag_part = refined[..., 1]
        
        # Stage 5: Reconstruct complex channel estimate
        h_pred = tf.complex(real_part, imag_part)

        return h_pred, err_var

    def save_weights(self, path: str | Path) -> None:
        """Save neural network weights to *path*."""
        self.model.save_weights(str(path))
        self._weights_loaded = True

    def load_weights(self, path: str | Path) -> None:
        """Load neural network weights from *path*."""
        self.model.build(input_shape=self._dummy_input_shape())
        self.model.load_weights(str(path))
        self._weights_loaded = True

    def compile(self, *args, **kwargs) -> None:
        """Proxy ``compile`` to the underlying keras model."""
        self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):  # pragma: no cover - passthrough
        """Proxy ``fit`` to the underlying keras model."""
        return self.model.fit(*args, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_network(self) -> tf.keras.Model:
        """
        Build the point-wise neural network for channel estimation refinement.
        
        Constructs a fully connected neural network that processes each resource
        element independently. The network takes 2 inputs (real and imaginary parts
        of LS estimate) and produces 2 outputs (refined real and imaginary parts).
        
        Theory:
            Network architecture:
            
            Input Layer: 2 neurons (Re, Im)
            ↓
            Hidden Layer 1: hidden_units[0] neurons + activation
            ↓
            Hidden Layer 2: hidden_units[1] neurons + activation
            ↓
            ...
            ↓
            Output Layer: 2 neurons (Re, Im) - no activation (linear)
            
            The network learns a nonlinear mapping:
            f: R² → R²
            
            Activation functions:
            - ReLU: f(x) = max(0, x) - most common, addresses vanishing gradients
            - Tanh: f(x) = tanh(x) - bounded output, centered at zero
            - Sigmoid: f(x) = 1/(1 + e^(-x)) - bounded to (0, 1)
            
            The output layer uses no activation (linear) to allow unbounded
            output values, as channel estimates can have arbitrary magnitude.
        
        Returns:
            Keras Model instance representing the neural network.
            The model processes inputs of shape [..., 2] and produces outputs
            of shape [..., 2], where the last dimension represents [Re, Im].
        """
        model = tf.keras.Sequential(name="neural_channel_estimator")
        
        # Hidden layers with specified sizes and activation
        for idx, units in enumerate(self.hidden_units):
            model.add(
                tf.keras.layers.Dense(
                    units,
                    activation=self.activation,
                    name=f"dense_{idx+1}",
                )
            )
        
        # Output layer: 2 neurons (Re, Im) with no activation (linear)
        # Linear activation allows unbounded output values
        model.add(tf.keras.layers.Dense(2, activation=None, name="dense_out"))
        return model

    def _dummy_input_shape(self) -> tuple[int, ...]:
        """
        Return a dummy input shape for weight loading.
        
        Provides a representative input shape for building the model before
        loading weights. The shape corresponds to the expected channel estimate
        dimensions with an additional dimension for real/imaginary parts.
        
        The network operates point-wise on the last dimension (size 2 for Re/Im),
        so the preceding dimensions can be arbitrary. This shape is used only
        for model building, not for actual inference.
        
        Returns:
            Tuple representing input shape:
            (batch, num_rx, num_tx, num_ut, num_streams, num_ofdm_symbols, fft_size, 2)
            The last dimension (2) represents [Re, Im] parts.
        """
        # The network operates point-wise on the last dimension of size 2.
        # We provide a shape with arbitrary batch size for building.
        return (None, 1, self.config.num_bs_ant, self.config.num_ut, 1,
                self.config.num_ofdm_symbols, self.config.fft_size, 2)


# --------------------------------------------------------------------------
# Utility functions for dataset generation (used by the training script)
# --------------------------------------------------------------------------
def stack_complex(real: tf.Tensor, imag: tf.Tensor) -> tf.Tensor:
    """Utility to stack real/imaginary parts along the last dimension."""
    return tf.stack([real, imag], axis=-1)
