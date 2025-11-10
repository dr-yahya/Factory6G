"""
Receiver components for 6G smart factory physical layer
"""

import tensorflow as tf
from sionna.phy import Block
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    ResourceGrid,
    LSChannelEstimator,
    LMMSEEqualizer,
    RemoveNulledSubcarriers
)
from sionna.phy.mapping import Demapper
from sionna.phy.fec.ldpc import LDPC5GDecoder
from .config import SystemConfig


class Receiver(Block):
    """
    Receiver chain: Channel Estimation -> Equalization -> Demapping -> LDPC Decoding
    """
    
    def __init__(self, config: SystemConfig, resource_grid: ResourceGrid,
                 stream_management: StreamManagement, encoder,
                 perfect_csi: bool = False,
                 channel_estimator: Block | None = None):
        """
        Initialize receiver components.
        
        Args:
            config: System configuration parameters
            resource_grid: OFDM resource grid
            stream_management: Stream management for MIMO
            encoder: LDPC encoder (needed for decoder)
            perfect_csi: Whether to use perfect channel state information
        """
        super().__init__()
        self.config = config
        self.resource_grid = resource_grid
        self.stream_management = stream_management
        self.perfect_csi = perfect_csi
        
        # Initialize components
        self._remove_nulled_subcarriers = RemoveNulledSubcarriers(resource_grid)
        
        self._default_channel_estimator = None
        if perfect_csi:
            self._channel_estimator = None
        else:
            if channel_estimator is not None:
                self._channel_estimator = channel_estimator
            else:
                self._default_channel_estimator = LSChannelEstimator(
                    resource_grid,
                    interpolation_type="nn"
                )
                self._channel_estimator = self._default_channel_estimator
        
        self._equalizer = LMMSEEqualizer(resource_grid, stream_management)
        self._demapper = Demapper("app", "qam", config.num_bits_per_symbol)
        self._decoder = LDPC5GDecoder(encoder)
    
    def estimate_channel(self, y: tf.Tensor, noise_var: tf.Tensor) -> tuple:
        """
        Estimate channel from received signal.
        
        Args:
            y: Received signal
            noise_var: Noise variance
            
        Returns:
            Tuple of (channel estimate, error variance)
        """
        if self.perfect_csi:
            # Perfect CSI case - should be handled separately with true channel
            raise ValueError("Perfect CSI should use true channel, not estimation")
        else:
            if self._channel_estimator is None:
                raise ValueError("Channel estimator is not initialized")
            return self._channel_estimator(y, noise_var)
    
    def equalize(self, y: tf.Tensor, h_hat: tf.Tensor, err_var: tf.Tensor,
                 noise_var: tf.Tensor) -> tuple:
        """
        Equalize received signal using channel estimate.
        
        Args:
            y: Received signal
            h_hat: Channel estimate
            err_var: Channel estimation error variance
            noise_var: Noise variance
            
        Returns:
            Tuple of (equalized symbols, effective noise variance)
        """
        return self._equalizer(y, h_hat, err_var, noise_var)
    
    def demap(self, x_hat: tf.Tensor, no_eff: tf.Tensor) -> tf.Tensor:
        """
        Demap equalized symbols to LLRs.
        
        Args:
            x_hat: Equalized symbols
            no_eff: Effective noise variance
            
        Returns:
            Log-likelihood ratios (LLRs)
        """
        return self._demapper(x_hat, no_eff)
    
    def decode(self, llr: tf.Tensor) -> tf.Tensor:
        """
        Decode LLRs to information bits.
        
        Args:
            llr: Log-likelihood ratios
            
        Returns:
            Decoded information bits
        """
        return self._decoder(llr)
    
    def call(self, y: tf.Tensor, h_hat: tf.Tensor, err_var: tf.Tensor,
             noise_var: tf.Tensor) -> tf.Tensor:
        """
        Complete receiver chain processing.
        
        Args:
            y: Received signal
            h_hat: Channel estimate (or true channel for perfect CSI)
            err_var: Channel estimation error variance (0.0 for perfect CSI)
            noise_var: Noise variance
            
        Returns:
            Decoded information bits
        """
        # Equalize
        x_hat, no_eff = self.equalize(y, h_hat, err_var, noise_var)
        
        # Demap to LLRs
        llr = self.demap(x_hat, no_eff)
        
        # Decode
        b_hat = self.decode(llr)
        
        return b_hat
    
    def __call__(self, y: tf.Tensor, h_hat: tf.Tensor, err_var: tf.Tensor,
                 noise_var: tf.Tensor) -> tf.Tensor:
        """Alias for call method for convenience"""
        return self.call(y, h_hat, err_var, noise_var)
    
    def process_with_perfect_csi(self, y: tf.Tensor, h: tf.Tensor,
                                  noise_var: tf.Tensor) -> tf.Tensor:
        """
        Process received signal with perfect channel knowledge.
        
        Args:
            y: Received signal
            h: True channel response
            noise_var: Noise variance
            
        Returns:
            Decoded information bits
        """
        # Remove nulled subcarriers from channel
        h_hat = self._remove_nulled_subcarriers(h)
        err_var = 0.0
        
        # Process through receiver chain
        return self.__call__(y, h_hat, err_var, noise_var)

