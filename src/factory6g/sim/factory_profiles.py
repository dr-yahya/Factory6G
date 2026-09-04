"""Factory size presets.

Single source of truth: this table used to be duplicated verbatim in
`cli/run.py` and `sim/flow.py`, which is a drift hazard.

Each preset carries both the user count and the hall geometry. The geometry now
reaches the channel model through `Factory6GConfig.system_runtime_config`, so
with the `inf` channel type the hall size and machine count genuinely change
propagation instead of only relabelling a user-count sweep.
"""

from __future__ import annotations

from typing import Any

FACTORY_SIZE_PRESETS: dict[str, dict[str, Any]] = {
    "s": {
        "room_dimensions": [15.0, 15.0, 5.0],
        "num_machines": 5,
        "machine_size_range": [[0.5, 2.0], [0.5, 2.0], [0.5, 1.5]],
        "num_ut": 4,
    },
    "m": {
        "room_dimensions": [25.0, 25.0, 6.0],
        "num_machines": 10,
        "machine_size_range": [[1.0, 3.0], [1.0, 3.0], [1.0, 2.5]],
        "num_ut": 8,
    },
    "l": {
        "room_dimensions": [40.0, 40.0, 8.0],
        "num_machines": 20,
        "machine_size_range": [[1.5, 4.0], [1.5, 4.0], [1.0, 3.0]],
        "num_ut": 16,
    },
    "apple": {
        # Consumer electronics precision assembly hall (iPhone-style)
        # 60x35m floor, 8m ceiling - rectangular assembly line layout
        # Dense compact workstations: SMT placers, robotic arms, test stations
        # num_ut=8: fft_size(128) must be divisible by num_ut for Kronecker pilots
        "room_dimensions": [60.0, 35.0, 8.0],
        "num_machines": 22,
        "machine_size_range": [[0.8, 2.5], [0.8, 2.0], [1.0, 2.5]],
        "num_ut": 8,
    },
}

FACTORY_SIZE_DISPLAY: dict[str, str] = {
    "s": "Small",
    "m": "Medium",
    "l": "Large",
    "apple": "Apple Factory",
}
