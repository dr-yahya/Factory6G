"""
6G LDPC Encoder/Decoder with Code Block Segmentation.

This module implements a 6G-compliant LDPC encoder and decoder that supports
larger block sizes by automatically segmenting large code blocks into smaller
ones that fit within the 5G LDPC encoder limits.

Theory:
    Code Block Segmentation:
    - 5G LDPC encoders have maximum block sizes:
      * Base Graph 1: k_max = 8448 bits
      * Base Graph 2: k_max = 3840 bits
    - For 6G systems with larger FFT sizes, information bits (k) can exceed
      these limits
    - Solution: Split large blocks into multiple smaller code blocks
    - Each code block is encoded independently
    - Encoded blocks are concatenated to form the final codeword
    
    Segmentation Strategy:
    1. Calculate number of code blocks: num_blocks = ceil(k / k_max)
    2. Split information bits into num_blocks segments
    3. Encode each segment independently
    4. Concatenate encoded blocks
    
    For decoding:
    1. Split LLRs into num_blocks segments
    2. Decode each segment independently
    3. Concatenate decoded information bits

References:
    - 3GPP TS 38.212: Multiplexing and channel coding (LDPC)
    - Richardson & Urbanke, "Modern Coding Theory" (LDPC codes)
"""

import tensorflow as tf
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from typing import Optional, Tuple, Union


class LDPC6GEncoder:
    """
    6G LDPC Encoder with automatic code block segmentation.
    
    This encoder wraps the 5G LDPC encoder and automatically handles code block
    segmentation when the information block size exceeds the encoder's maximum
    supported size. This enables support for larger FFT sizes in 6G systems.
    
    Theory:
        The encoder performs the following steps:
        
        1. Check if k exceeds maximum supported size
        2. If yes, calculate number of code blocks needed
        3. Split information bits into segments
        4. Encode each segment with 5G LDPC encoder
        5. Concatenate encoded blocks
        
        Maximum block sizes (from 3GPP TS 38.212):
        - Base Graph 1: k_max = 8448 bits
        - Base Graph 2: k_max = 3840 bits
        
        The encoder automatically selects the appropriate base graph and
        segments accordingly.
    """
    
    # Maximum information bits per code block for 5G LDPC
    # Base Graph 1: up to 8448, Base Graph 2: up to 3840
    # We use a conservative limit to ensure compatibility
    MAX_K_BG1 = 8448
    MAX_K_BG2 = 3840
    
    def __init__(self, k: int, n: int, bg: Optional[str] = None):
        """
        Initialize 6G LDPC encoder with code block segmentation.
        
        Args:
            k: Number of information bits per codeword.
                Can exceed 5G limits; will be automatically segmented.
            n: Desired codeword length (coded bits).
            bg: Base graph selection ("bg1", "bg2", or None for auto).
                If None, automatically selects based on k and code rate.
        """
        self._k_total = k
        self._n_total = n
        self._code_rate = k / n if n > 0 else 0.0
        
        # Determine maximum k per block based on base graph
        # Base graph selection: BG1 for k > 3840 or high rates, BG2 otherwise
        if bg == "bg1" or (bg is None and (k > 3840 or self._code_rate > 0.67)):
            self._max_k_per_block = self.MAX_K_BG1
            self._bg = "bg1"
        else:
            self._max_k_per_block = self.MAX_K_BG2
            self._bg = "bg2" if bg is None else bg
        
        # Calculate number of code blocks needed
        if k <= self._max_k_per_block:
            # Single code block - no segmentation needed
            self._num_blocks = 1
            self._k_per_block = k
            self._n_per_block = n
            self._encoders = [LDPC5GEncoder(k, n, bg=self._bg)]
        else:
            # Multiple code blocks - segmentation required
            self._num_blocks = (k + self._max_k_per_block - 1) // self._max_k_per_block
            self._k_per_block = k // self._num_blocks
            # Ensure last block gets any remainder
            self._k_last_block = k - (self._num_blocks - 1) * self._k_per_block
            
            # Calculate n per block (maintain same code rate)
            self._n_per_block = int(self._k_per_block / self._code_rate)
            self._n_last_block = int(self._k_last_block / self._code_rate)
            
            # Create encoders for each block
            self._encoders = []
            for i in range(self._num_blocks):
                if i == self._num_blocks - 1:
                    # Last block may have different size
                    k_block = self._k_last_block
                    n_block = self._n_last_block
                else:
                    k_block = self._k_per_block
                    n_block = self._n_per_block
                
                # Ensure valid block sizes
                if k_block > 0 and n_block > 0:
                    self._encoders.append(LDPC5GEncoder(k_block, n_block, bg=self._bg))
    
    def __call__(self, bits: tf.Tensor) -> tf.Tensor:
        """
        Encode information bits with automatic code block segmentation.
        
        Args:
            bits: Information bits to encode
                Shape: [..., k_total] where k_total is total information bits
                
        Returns:
            Encoded codeword bits
            Shape: [..., n_total] where n_total is total coded bits
        """
        if self._num_blocks == 1:
            # Single block - direct encoding
            return self._encoders[0](bits)
        
        # Multiple blocks - segment and encode
        # Get input shape
        shape = tf.shape(bits)
        batch_dims = shape[:-1]
        k_total = shape[-1]
        
        # Split bits into segments
        encoded_blocks = []
        start_idx = 0
        
        for i, encoder in enumerate(self._encoders):
            if i == self._num_blocks - 1:
                # Last block
                k_block = self._k_last_block
                n_block = self._n_last_block
            else:
                k_block = self._k_per_block
                n_block = self._n_per_block
            
            # Extract segment
            bits_segment = bits[..., start_idx:start_idx + k_block]
            start_idx += k_block
            
            # Encode segment
            encoded_block = encoder(bits_segment)
            encoded_blocks.append(encoded_block)
        
        # Concatenate encoded blocks
        return tf.concat(encoded_blocks, axis=-1)
    
    @property
    def num_blocks(self) -> int:
        """Number of code blocks used for segmentation."""
        return self._num_blocks
    
    @property
    def k_total(self) -> int:
        """Total number of information bits."""
        return self._k_total
    
    @property
    def n_total(self) -> int:
        """Total number of coded bits."""
        return self._n_total


class LDPC6GDecoder:
    """
    6G LDPC Decoder with automatic code block concatenation.
    
    This decoder wraps the 5G LDPC decoder and automatically handles decoding
    of segmented code blocks, concatenating the results.
    """
    
    def __init__(self, encoder: LDPC6GEncoder, num_iter: int = 50, hard_out: bool = True, return_num_iter: bool = True):
        """
        Initialize 6G LDPC decoder.
        
        Args:
            encoder: Corresponding LDPC6GEncoder instance (defines code structure).
            num_iter: Maximum number of decoding iterations.
            hard_out: If True, return hard-decoded bits. If False, return soft LLRs.
            return_num_iter: If True, return tuple (decoded_bits, num_iterations).
        """
        self._encoder = encoder
        self._num_iter = num_iter
        self._hard_out = hard_out
        self._return_num_iter = return_num_iter
        
        # Create decoders for each code block
        self._decoders = []
        for enc in encoder._encoders:
            self._decoders.append(LDPC5GDecoder(enc, num_iter=num_iter, hard_out=hard_out, return_num_iter=return_num_iter))
    
    def __call__(self, llr: tf.Tensor) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Decode LLRs with automatic code block handling.
        
        Args:
            llr: Log-likelihood ratios from demapper
                Shape: [..., n_total] where n_total is total coded bits
                
        Returns:
            If return_num_iter is True:
                Tuple of (decoded_bits, num_iterations)
            Otherwise:
                Decoded information bits
            Shape: [..., k_total] where k_total is total information bits
        """
        if self._encoder._num_blocks == 1:
            # Single block - direct decoding
            return self._decoders[0](llr)
        
        # Multiple blocks - split and decode
        decoded_blocks = []
        num_iter_list = []
        start_idx = 0
        
        for i, decoder in enumerate(self._decoders):
            if i == self._encoder._num_blocks - 1:
                # Last block
                n_block = self._encoder._n_last_block
            else:
                n_block = self._encoder._n_per_block
            
            # Extract LLR segment
            llr_segment = llr[..., start_idx:start_idx + n_block]
            start_idx += n_block
            
            # Decode segment
            decoder_out = decoder(llr_segment)
            if isinstance(decoder_out, tuple):
                decoded_block, num_iter = decoder_out
                num_iter_list.append(num_iter)
            else:
                decoded_block = decoder_out
                num_iter_list.append(tf.constant(0, dtype=tf.int32))
            
            decoded_blocks.append(decoded_block)
        
        # Concatenate decoded blocks
        decoded = tf.concat(decoded_blocks, axis=-1)
        
        # Return format matches 5G decoder
        if self._return_num_iter:
            # Return maximum number of iterations across all blocks
            max_iter = tf.reduce_max(tf.stack(num_iter_list))
            return decoded, max_iter
        else:
            return decoded

