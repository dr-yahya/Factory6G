"""
Models for 6G wireless communication systems
"""

from .model import Model

# Optional import for E2E model (may have different dependencies)
try:
    from .e2e_channel_estimation import E2EChannelEstimationModel
    __all__ = ['Model', 'E2EChannelEstimationModel']
except ImportError:
    __all__ = ['Model']

