"""6G baseline scenario for indoor smart factory."""

from .spec import ScenarioSpec

SCENARIO = ScenarioSpec(
    name="6g_baseline",
    description="6G baseline simulation optimized for 6G requirements - perfect CSI, lower code rate, more antennas",
    estimators=["ls_lin"],
    perfect_csi=[False],  # Imperfect CSI avoids gather index issues with current stream management
    channel_scenario="umi",
    ebno_min=-5.0,  # Start from -5 dB as requested
    ebno_max=25.0,  # Keep upper range
    ebno_step=0.5,  # Finer step for detailed curve
    batch_size=8,  # Enforce minimum per 6G params (never less)
    max_iter=100,  # More iterations to get better statistics
    target_block_errors=1000,  # Much higher threshold (accounts for multiple streams: 16 streams × batch_size)
    target_bler=1e-9,  # 6G target BLER for early stopping
    notes="6G optimized scenario: perfect CSI, lower code rate (0.33), more BS antennas (64), QPSK for reliability. Targeting BER/BLER < 1e-9.",
)

