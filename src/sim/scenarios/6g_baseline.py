"""6G baseline scenario for indoor smart factory."""

from ..scenario_spec import ScenarioSpec

SCENARIO = ScenarioSpec(
    name="6g_baseline",
    description="6G baseline simulation for indoor smart factory with LS linear estimator, no resource management",
    estimators=["ls_lin"],
    perfect_csi=[False],
    channel_scenario="umi",
    ebno_min=-5.0,
    ebno_max=15.0,
    ebno_step=2.0,
    batch_size=8,
    max_iter=25,
    target_block_errors=300,
    target_bler=5e-4,
    notes="6G indoor smart factory baseline scenario using LS linear interpolation channel estimator. UMi channel model ideal for indoor factory environments with dense equipment, machinery, and metallic structures. Uses 6G-compliant parameters: fft_size=512, num_bs_ant=32, num_ut=8, num_ut_ant=2.",
)

