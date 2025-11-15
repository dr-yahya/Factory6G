"""6G comprehensive evaluation scenario for indoor smart factory."""

from ..scenario_spec import ScenarioSpec

SCENARIO = ScenarioSpec(
    name="6g_full_evaluation",
    description="6G comprehensive evaluation for indoor smart factory: multiple estimators, perfect/imperfect CSI, with resource management",
    estimators=["ls_lin", "ls_nn"],
    perfect_csi=[True, False],
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
    notes="6G comprehensive evaluation scenario for indoor smart factory environments. Combines multiple channel estimators (LS linear, LS nearest-neighbor), both perfect and imperfect CSI conditions, and static resource management. UMi channel model captures dense indoor factory environments with machinery, equipment, and metallic structures causing rich multipath propagation.",
)

