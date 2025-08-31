"""
TTS models for synthesis pipeline.
"""

from .base import BaseSynthesizer
from .xtts import XTTSSynthesizer
from ..config import ModelType


# Model registry for factory pattern
MODEL_REGISTRY = {
    ModelType.XTTS_V2: XTTSSynthesizer,
}


def create_synthesizer(model_type: ModelType, config) -> BaseSynthesizer:
    """Factory function to create synthesizer instances.

    Args:
        model_type: Type of model to create
        config: Model configuration

    Returns:
        Synthesizer instance

    Raises:
        ValueError: If model type is not supported
    """
    if model_type not in MODEL_REGISTRY:
        available_types = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unsupported model type: {model_type}. Available: {available_types}")

    synthesizer_class = MODEL_REGISTRY[model_type]
    return synthesizer_class(config)


def get_available_models():
    """Get list of available model types."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    "BaseSynthesizer",
    "XTTSSynthesizer",
    "MODEL_REGISTRY",
    "create_synthesizer",
    "get_available_models"
]