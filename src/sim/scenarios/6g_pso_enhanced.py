from .spec import ScenarioSpec

# PSO-enhanced scenario for 6G Smart Factory
# Uses PSO channel estimator to improve performance over LS
# Based on 6g_smart_factory_sionna_baseline configuration

SCENARIO = ScenarioSpec(
    name="6g_pso_enhanced",
    description="6G Smart Factory with PSO Channel Estimation (Stabilized)",
    channel_scenario="umi",
    
    # Simulation parameters - MAXIMIZED for stability
    ebno_min=-4.0,
    ebno_max=6.0,
    ebno_step=1.0,
    batch_size=16,  # Increased from 8 for maximum statistical averaging
    max_iter=500,  # Increased from 200 for excellent statistical stability
    target_block_errors=1000,  # Increased from 500 for smooth convergence
    target_bler=1e-5,
    
    # Channel configuration
    channel_model_type="rayleigh",  # Use Rayleigh for speed/stability
    min_ut_velocity=0.0,
    max_ut_velocity=0.0,
    
    # Antenna configuration (same as smart factory)
    num_bs_ant=4,
    num_ut=4,
    num_ut_ant=1,
    
    # Estimator configuration - MAXIMIZED for stability and convergence
    estimators=["pso"],
    estimator_kwargs={
        "pso": {
            "degree": 3,
            "swarm_size": 30,  # Increased from 20 for thorough exploration
            "iters": 40,  # Increased from 30 for guaranteed convergence
            "inertia_start": 0.7,
            "inertia_end": 0.4,
            "c1": 1.5,
            "c2": 1.5,
            "seed": 42  # Fixed seed for reproducibility
        }
    },
    perfect_csi=[False]
)
