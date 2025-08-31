"""
Generation strategies for synthesis pipeline.
"""

from .base import BaseGenerationStrategy
from .method1 import Method1Strategy
from .method2 import Method2Strategy
from config import GenerationMethod


# Strategy registry for factory pattern
STRATEGY_REGISTRY = {
    GenerationMethod.METHOD1: Method1Strategy,
    GenerationMethod.METHOD2: Method2Strategy,
}


def create_strategy(
    method: GenerationMethod,
    config,
    dataset,
    synthesizer
) -> BaseGenerationStrategy:
    """Factory function to create generation strategy instances.

    Args:
        method: Generation method type
        config: Configuration object
        dataset: Dataset instance
        synthesizer: Synthesizer instance

    Returns:
        Strategy instance

    Raises:
        ValueError: If generation method is not supported
    """
    if method not in STRATEGY_REGISTRY:
        available_methods = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unsupported generation method: {method}. Available: {available_methods}")

    strategy_class = STRATEGY_REGISTRY[method]
    return strategy_class(config, dataset, synthesizer)


def get_available_strategies():
    """Get list of available generation strategies."""
    return list(STRATEGY_REGISTRY.keys())


__all__ = [
    "BaseGenerationStrategy",
    "Method1Strategy",
    "Method2Strategy",
    "STRATEGY_REGISTRY",
    "create_strategy",
    "get_available_strategies"
]