"""6G baseline scenario for indoor smart factory."""

from ..scenario_spec import ScenarioSpec

SCENARIO = ScenarioSpec(
    name="6g_baseline",
    description="6G baseline simulation optimized for 6G requirements - perfect CSI, lower code rate, more antennas",
    estimators=["ls_lin"],
    perfect_csi=[False],  # Use imperfect CSI (perfect CSI has index errors)
    channel_scenario="umi",
    ebno_min=15.0,  # Start at even higher Eb/No where decoder should work
    ebno_max=35.0,  # Extended range for 6G requirements
    ebno_step=2.0,
    batch_size=16,  # Larger batch for better statistics
    max_iter=50,  # More iterations for better convergence
    target_block_errors=300,
    target_bler=1e-9,  # 6G target
    notes="6G optimized scenario: perfect CSI, lower code rate (0.33), more BS antennas (64), QPSK for reliability. Targeting BER/BLER < 1e-9.",
)

