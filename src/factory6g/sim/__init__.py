"""Simulation modules for the Factory6G runtime."""

__all__ = [
    "BatchContext",
    "ConfigError",
    "Factory6GConfig",
    "ResourceManagerFeedback",
    "configure_env",
    "load_config",
    "run_simulation_flow",
    "setup_gpu",
]


def __getattr__(name: str):
    if name in {"ConfigError", "Factory6GConfig", "load_config"}:
        from .config import ConfigError, Factory6GConfig, load_config

        return {
            "ConfigError": ConfigError,
            "Factory6GConfig": Factory6GConfig,
            "load_config": load_config,
        }[name]
    if name in {"configure_env", "setup_gpu"}:
        from .env import configure_env, setup_gpu

        return {"configure_env": configure_env, "setup_gpu": setup_gpu}[name]
    if name in {"BatchContext", "ResourceManagerFeedback"}:
        from .types import BatchContext, ResourceManagerFeedback

        return {
            "BatchContext": BatchContext,
            "ResourceManagerFeedback": ResourceManagerFeedback,
        }[name]
    if name == "run_simulation_flow":
        from .flow import run_simulation_flow

        return run_simulation_flow
    raise AttributeError(name)
