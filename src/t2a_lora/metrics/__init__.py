"""
Audio quality evaluation metrics module.
"""

from enum import Enum
from .base import (
    BaseMetricCalculator,
    ModelConfig,
    MetricCalculationError,
    ModelLoadError
)
from .utmos import UTMOSCalculator
from .wer import WERCalculator
from .sim import SIMCalculator
from .ffe import FFECalculator
from .mcd import MCDCalculator


class MetricType(Enum):
    """Supported metric types."""
    UTMOS = "utmos"
    WER = "wer"
    SIM = "sim"
    FFE = "ffe"
    MCD = "mcd"


# Metric registry for easy access
METRIC_CALCULATORS = {
    MetricType.UTMOS: UTMOSCalculator,
    MetricType.WER: WERCalculator,
    MetricType.SIM: SIMCalculator,
    MetricType.FFE: FFECalculator,
    MetricType.MCD: MCDCalculator
}


def create_calculator(metric_type: MetricType, config: ModelConfig) -> BaseMetricCalculator:
    """
    Factory function to create metric calculators.

    Args:
        metric_type: Type of metric calculator
        config: Model configuration

    Returns:
        Metric calculator instance

    Raises:
        ValueError: If metric type is not supported
    """
    if metric_type not in METRIC_CALCULATORS:
        available_metrics = list(METRIC_CALCULATORS.keys())
        raise ValueError(f"Unsupported metric: {metric_type}. Available metrics: {available_metrics}")

    calculator_class = METRIC_CALCULATORS[metric_type]
    return calculator_class(config)


def get_available_metrics() -> list[MetricType]:
    """Get list of available metric types."""
    return list(METRIC_CALCULATORS.keys())


__all__ = [
    # Base classes and exceptions
    'BaseMetricCalculator',
    'ModelConfig',
    'MetricCalculationError',
    'ModelLoadError',

    # Metric calculators
    'UTMOSCalculator',
    'WERCalculator',
    'SIMCalculator',
    'FFECalculator',
    'MCDCalculator',

    # Enums and registry
    'MetricType',
    'METRIC_CALCULATORS',
    'create_calculator',
    'get_available_metrics'
]
