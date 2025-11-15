"""6G comprehensive channel estimator comparison scenario for indoor smart factory."""

from ..scenario_spec import ScenarioSpec

SCENARIO = ScenarioSpec(
    name="6g_estimators_full",
    description="6G comprehensive channel estimator comparison for indoor smart factory: LS linear, LS nearest-neighbor, and neural",
    estimators=["ls_lin", "ls_nn", "neural"],
    perfect_csi=[False],
    channel_scenario="umi",
    ebno_min=-5.0,
    ebno_max=15.0,
    ebno_step=2.0,
    batch_size=8,
    max_iter=30,
    target_block_errors=300,
    target_bler=5e-4,
    estimator_kwargs={
        "neural": {"hidden_units": [32, 32]}
    },
    notes="6G comprehensive estimator comparison for indoor smart factory environments. Includes classical (LS linear, LS nearest-neighbor) and neural channel estimators. UMi channel model captures indoor factory multipath with machinery and metallic structures. Requires neural estimator weights for neural estimator.",
)

