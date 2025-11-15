# 6G LDPC Encoder/Decoder Implementation

## Overview

This document describes the implementation of a 6G-compliant LDPC encoder and decoder that supports larger block sizes through automatic code block segmentation.

## Problem Statement

The 5G LDPC encoder (LDPC5GEncoder from Sionna) has maximum block size limitations:
- **Base Graph 1**: Maximum k = 8448 information bits
- **Base Graph 2**: Maximum k = 3840 information bits

For 6G systems with larger FFT sizes (1024, 2048, 4096, etc.), the number of information bits (k) can exceed these limits, causing encoding failures with errors like:
```
ValueError: Unsupported code length (k too large).
```

## Solution: Code Block Segmentation

The 6G LDPC encoder automatically segments large information blocks into smaller code blocks that fit within the 5G encoder limits. Each segment is encoded independently, and the encoded blocks are concatenated to form the final codeword.

### Architecture

```
Large Information Block (k > 8448)
    ↓
[Segmentation]
    ↓
Block 1 (k₁ ≤ 8448) → [5G Encoder] → Coded Block 1 (n₁)
Block 2 (k₂ ≤ 8448) → [5G Encoder] → Coded Block 2 (n₂)
...
Block N (kₙ ≤ 8448) → [5G Encoder] → Coded Block N (nₙ)
    ↓
[Concatenation]
    ↓
Final Codeword (n = n₁ + n₂ + ... + nₙ)
```

## Implementation Details

### LDPC6GEncoder

**Location**: `src/components/ldpc_6g.py`

**Key Features**:
1. **Automatic Segmentation**: Detects when k exceeds maximum and segments accordingly
2. **Base Graph Selection**: Automatically selects BG1 or BG2 based on k and code rate
3. **Code Rate Preservation**: Maintains the same code rate across all segments
4. **Transparent Interface**: Compatible with existing 5G encoder interface

**Usage**:
```python
from src.components.ldpc_6g import LDPC6GEncoder

# Works for both small and large k
encoder = LDPC6GEncoder(k=10000, n=20000)  # k exceeds 5G limits
bits = tf.random.uniform([batch, k], maxval=2, dtype=tf.float32)
encoded = encoder(bits)  # Shape: [batch, n]
```

**Segmentation Strategy**:
- If k ≤ max_k_per_block: Single code block (no segmentation)
- If k > max_k_per_block: Split into `ceil(k / max_k_per_block)` blocks
- Each block maintains the same code rate: R = k_block / n_block
- Last block may have different size to handle remainder

### LDPC6GDecoder

**Location**: `src/components/ldpc_6g.py`

**Key Features**:
1. **Automatic Desegmentation**: Splits LLRs into segments matching encoder blocks
2. **Independent Decoding**: Each segment decoded separately
3. **Result Concatenation**: Concatenates decoded bits to recover full information block
4. **Compatible Return Format**: Matches 5G decoder return format (bits, num_iter)

**Usage**:
```python
from src.components.ldpc_6g import LDPC6GDecoder

decoder = LDPC6GDecoder(encoder, num_iter=50)
llr = ...  # Log-likelihood ratios from demapper
decoded, num_iter = decoder(llr)  # Shape: [batch, k]
```

## Integration

### Transmitter Integration

The `Transmitter` class automatically uses `LDPC6GEncoder` when available:

```python
# src/components/transmitter.py
from .ldpc_6g import LDPC6GEncoder

try:
    self._encoder = LDPC6GEncoder(self._k, self._n)  # 6G encoder
except Exception as e:
    # Fallback to 5G encoder for compatibility
    self._encoder = LDPC5GEncoder(self._k, self._n)
```

### Receiver Integration

The `Receiver` class automatically uses `LDPC6GDecoder` when the encoder is 6G:

```python
# src/components/receiver.py
from .ldpc_6g import LDPC6GDecoder, LDPC6GEncoder

if isinstance(encoder, LDPC6GEncoder):
    self._decoder = LDPC6GDecoder(encoder)
else:
    self._decoder = LDPC5GDecoder(encoder, ...)
```

## Benefits

1. **6G Compliance**: Supports larger FFT sizes (1024, 2048, 4096, etc.) required for 6G
2. **Backward Compatible**: Falls back to 5G encoder for smaller blocks
3. **Transparent**: No changes needed to existing code that uses the encoder
4. **Efficient**: Segmentation only occurs when necessary
5. **Standards Compliant**: Uses 3GPP TS 38.212 base graphs

## Testing

The encoder has been tested with:
- Small blocks (k ≤ 3840): Single code block, works like 5G encoder
- Medium blocks (3840 < k ≤ 8448): Single code block with BG1
- Large blocks (k > 8448): Multiple code blocks with segmentation

**Test Example**:
```python
# Test with k=10000 (exceeds BG1 max of 8448)
encoder = LDPC6GEncoder(k=10000, n=20000)
print(f'Number of code blocks: {encoder.num_blocks}')  # Output: 2
```

## Performance Considerations

1. **Segmentation Overhead**: Minimal - only splits when necessary
2. **Memory**: Slightly higher memory usage for multiple encoder instances
3. **Computation**: Parallel encoding/decoding of segments (TensorFlow handles this)
4. **Latency**: Negligible increase due to segmentation/concatenation

## Future Enhancements

1. **Adaptive Segmentation**: Optimize block sizes for better performance
2. **Rate Matching**: Enhanced rate matching across segments
3. **Error Correction**: Cross-segment error correction for better reliability

## References

- 3GPP TS 38.212: Multiplexing and channel coding (LDPC)
- Richardson & Urbanke, "Modern Coding Theory" (LDPC codes)
- Sionna Documentation: https://nvlabs.github.io/sionna/

