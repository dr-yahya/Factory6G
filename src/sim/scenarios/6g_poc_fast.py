"""Fast 6G PoC scenario using Rayleigh fading."""

from .spec import ScenarioSpec

SCENARIO = ScenarioSpec(
    name="6g_poc_fast",
    description="Fast 6G PoC simulation using Rayleigh fading channel for speed.",
    estimators=["ls_lin"],
    perfect_csi=[True],  # Focus on perfect CSI for PoC speed
    channel_scenario="umi", # Placeholder, will be overridden by channel_model_type
    ebno_min=0.0,
    ebno_max=20.0,
    ebno_step=2.0,
    batch_size=128,  # High batch size for GPU efficiency
    max_iter=10,     # Low iterations for fast turnaround
    target_block_errors=100,
    target_bler=1e-2,
    resource_manager={
        "channel_model_type": "rayleigh", # Custom config to trigger Rayleigh
    },
    notes="Optimized for speed: Rayleigh channel, high batch size, low iterations.",
)
