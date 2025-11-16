"""6G scenario running with AI-based channel estimator for indoor smart factory."""

from .spec import ScenarioSpec

SCENARIO = ScenarioSpec(
    name="6g_ai_estimator",
    description="6G scenario using neural (AI) channel estimator only",
    estimators=["neural"],
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
    notes="6G indoor smart factory run with neural (AI) channel estimator. UMi channel model captures indoor factory multipath with machinery and metallic structures. Requires neural estimator weights for the neural estimator.",
)


