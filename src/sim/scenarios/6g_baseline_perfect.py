"""6G baseline perfect CSI comparison scenario for indoor smart factory."""

from ..scenario_spec import ScenarioSpec

SCENARIO = ScenarioSpec(
    name="6g_baseline_perfect",
    description="6G indoor smart factory baseline comparing perfect vs imperfect CSI with LS linear estimator",
    estimators=["ls_lin"],
    perfect_csi=[True, False],
    channel_scenario="umi",
    ebno_min=-5.0,
    ebno_max=15.0,
    ebno_step=2.0,
    batch_size=8,
    max_iter=30,
    target_block_errors=300,
    target_bler=5e-4,
    notes="6G indoor smart factory simulation comparing perfect vs imperfect CSI performance with LS linear estimator. UMi channel model captures indoor factory multipath propagation. Demonstrates channel estimation impact on system performance in dense indoor environments.",
)

