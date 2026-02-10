
import tensorflow as tf
from sionna.phy import Block
from sionna.phy.ofdm import LSChannelEstimator, ResourceGrid

class DFTChannelEstimator(Block):
    """DFT-based Channel Estimator.
    
    Performs channel estimation by filtering LS estimates in the delay domain.
    1. Obtain LS estimates in frequency domain.
    2. Convert to time domain (IDFT).
    3. Apply time-domain window (filter out noise beyond max delay/CP).
    4. Convert back to frequency domain (DFT).
    """
    def __init__(self, resource_grid, config=None):
        super().__init__()
        self._rg = resource_grid
        self._ls_estimator = LSChannelEstimator(resource_grid, interpolation_type="nn")
        self.fft_size = resource_grid.fft_size
        self.cp_length = resource_grid.cyclic_prefix_length
        
    def call(self, y, no):
        """
        Args:
            y: Received signal
            no: Noise variance
        
        Returns:
            h_hat: Channel estimate
            err_var: Error variance
        """
        # 1. Initial LS Estimate
        h_ls, err_var_ls = self._ls_estimator(y, no)
        # h_ls: [batch, rx, tx, streams, sym, fft_size]
        
        # 2. Signal Processing (DFT Filtering)
        # Move last dim (freq) to apply FFT
        
        # IDFT to delay domain
        h_delay = tf.signal.ifft(h_ls)
        
        # 3. Windowing
        # Keep only taps within CP length (conservative assumption for meaningful channel taps)
        # Or a bit more to be safe. CP length is a good threshold for OFDM systems.
        # We create a mask: 1s for [0, cp_length] and [fft_size - small_margin, fft_size] (if non-causal/wrapping effects)
        # Typically just [0, cp_length] is significant for causal channel.
        
        # Create mask
        # Shape: [fft_size]
        mask_indices = tf.range(self.fft_size)
        # Keep taps 0 to cp_length
        mask = tf.cast(mask_indices < self.cp_length, dtype=h_delay.dtype)
        
        # Apply mask
        h_filtered_delay = h_delay * mask
        
        # 4. DFT back to frequency domain
        h_dft = tf.signal.fft(h_filtered_delay)
        
        # Update error variance
        # Theoretical noise reduction is proportional to (CP_length / FFT_size)
        # because we zeroed out substantial noise-only taps.
        noise_reduction_factor = float(self.cp_length) / float(self.fft_size)
        err_var_dft = err_var_ls * noise_reduction_factor
        
        return h_dft, err_var_dft
