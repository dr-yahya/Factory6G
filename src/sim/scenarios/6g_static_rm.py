"""6G static resource manager scenario for indoor smart factory."""

from ..scenario_spec import ScenarioSpec

SCENARIO = ScenarioSpec(
    name="6g_static_rm",
    description="6G indoor smart factory static resource manager with LS linear estimator and per-UT power shaping",
    estimators=["ls_lin"],
    perfect_csi=[False],
    channel_scenario="umi",
    resource_manager={
        "active_ut_mask": [1, 1, 1, 1, 1, 1, 1, 1],  # All 8 UTs scheduled (6G: 8-256 UTs)
        "per_ut_power": [1.10, 0.95, 1.05, 0.90, 1.00, 1.08, 0.92, 1.03],  # Power control for 8 UTs
    },
    ebno_min=-5.0,
    ebno_max=15.0,
    ebno_step=2.0,
    batch_size=8,
    max_iter=30,
    target_block_errors=300,
    target_bler=5e-4,
    notes="6G indoor smart factory scenario with static resource management. UMi channel model for dense indoor factory environments. All 8 user terminals (sensors/devices) scheduled with power control to balance edge and center devices. Essential for managing multiple IoT devices in indoor factory settings. Uses 6G-compliant parameters.",
)

