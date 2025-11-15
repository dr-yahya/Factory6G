"""6G channel estimator comparison with resource management scenario for indoor smart factory."""

from ..scenario_spec import ScenarioSpec

SCENARIO = ScenarioSpec(
    name="6g_estimators_with_rm",
    description="6G indoor smart factory channel estimator comparison with static resource management",
    estimators=["ls_lin", "ls_nn"],
    perfect_csi=[False],
    channel_scenario="umi",
    resource_manager={
        "active_ut_mask": [1, 1, 1, 1, 1, 1, 1, 1],  # All 8 UTs scheduled
        "per_ut_power": [1.10, 0.95, 1.05, 0.90, 1.00, 1.08, 0.92, 1.03],
    },
    ebno_min=-5.0,
    ebno_max=15.0,
    ebno_step=2.0,
    batch_size=8,
    max_iter=30,
    target_block_errors=300,
    target_bler=5e-4,
    notes="6G indoor smart factory scenario comparing channel estimators (LS linear vs LS nearest-neighbor) with static resource management. UMi channel model captures indoor factory multipath. Evaluates estimator performance under resource management constraints for managing multiple IoT devices in dense indoor environments.",
)

