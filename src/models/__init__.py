"""
Models for 6G wireless communication systems
"""

__all__ = ["Model"]


def __getattr__(name: str):
    if name == "Model":
        from .model import Model

        return Model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

