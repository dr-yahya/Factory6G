"""6G channel estimator comparison scenario for indoor smart factory."""

from ..scenario_spec import ScenarioSpec

SCENARIO = ScenarioSpec(
    name="6g_estimators_comparison",
    description="6G indoor smart factory channel estimator comparison: LS linear vs LS nearest-neighbor",
    estimators=["ls_lin", "ls_nn"],
    perfect_csi=[False],
    channel_scenario="umi",
    ebno_min=-5.0,
    ebno_max=15.0,
    ebno_step=2.0,
    batch_size=8,
    max_iter=30,
    target_block_errors=300,
    target_bler=5e-4,
    notes="6G indoor smart factory scenario comparing LS linear interpolation vs LS nearest-neighbor interpolation channel estimators. UMi channel model ideal for indoor factory environments with rich multipath. Evaluates interpolation method impact on 6G system performance in dense indoor settings.",
)

