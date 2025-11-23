"""6G Smart Factory scenario (Pre-enhancement) using Rayleigh channel for fast execution."""

from .spec import ScenarioSpec

SCENARIO = ScenarioSpec(
    name="6g_smart_factory_sionna_baseline",
    description="6G Sionna Baseline (Pre-enhancement) - Smart Factory simulation using Rayleigh channel",
    estimators=["ls_lin"],
    perfect_csi=[False],
    channel_scenario="umi", # Kept for config compatibility, but channel_model_type overrides
    channel_model_type="rayleigh",
    ebno_min=-4.0,
    ebno_max=6.0,
    ebno_step=1.0,
    batch_size=16, # Increased from 4 to match PSO enhanced for fair comparison
    max_iter=500,  # Increased from 50 to match PSO enhanced for statistical stability
    target_block_errors=1000,  # Increased from 200 to match PSO enhanced
    target_bler=1e-5,
    num_bs_ant=4, # Reduced from 32 to reduce array gain and show errors
    num_ut=4,
    num_ut_ant=1,
    notes="Official 6G Smart Factory scenario (Pre-enhancement). Uses Rayleigh fading and reduced antenna count for efficient simulation. Parameters matched to PSO enhanced for fair comparison.",
)
