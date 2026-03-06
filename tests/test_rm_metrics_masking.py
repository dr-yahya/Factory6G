from __future__ import annotations

import numpy as np

from src.models.resource_manager import ResourceDirectives
from src.sim.stages.common import extract_error_stats, transmitted_ut_mask


def test_extract_error_stats_filters_muted_users():
    bits = np.zeros((1, 4, 1, 8), dtype=np.int32)
    bits_hat = bits.copy()

    # Muted users are intentionally wrong; they must not affect BER/throughput stats.
    bits_hat[:, 2, :, :] = 1
    bits_hat[:, 3, :, :] = 1

    mask = [1, 1, 0, 0]
    filtered = extract_error_stats(bits, bits_hat, ut_mask=mask)
    unfiltered = extract_error_stats(bits, bits_hat)

    assert filtered["bit_errors"] == 0
    assert filtered["total_bits"] == 16
    assert unfiltered["bit_errors"] == 16
    assert unfiltered["total_bits"] == 32


def test_transmitted_ut_mask_uses_active_and_power():
    directives = ResourceDirectives(
        active_ut_mask=[1, 1, 1, 0],
        per_ut_power=[1.0, 0.0, 0.5, 1.0],
        pilot_reuse_factor=1,
    )
    mask = transmitted_ut_mask(directives, num_ut=4)
    assert mask == [1, 0, 1, 0]
