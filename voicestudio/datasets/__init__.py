"""
Dataset loaders for synthesis pipeline.
"""

from .base import BaseSynthesisDataset
from .vctk import VctkSynthesisDataset
from .ljspeech import LJSpeechSynthesisDataset
from .libritts import LibriTTSSynthesisDataset
from ..metrics.presets import DatasetType


# Dataset registry for factory pattern
DATASET_REGISTRY = {
    DatasetType.VCTK: VctkSynthesisDataset,
    DatasetType.LJSPEECH: LJSpeechSynthesisDataset,
    DatasetType.LIBRITTS: LibriTTSSynthesisDataset,
}


def create_dataset(dataset_type: DatasetType, config, **kwargs) -> BaseSynthesisDataset:
    """Factory function to create dataset instances.

    Args:
        dataset_type: Type of dataset to create
        config: Dataset configuration
        **kwargs: Additional arguments for dataset initialization

    Returns:
        Dataset instance

    Raises:
        ValueError: If dataset type is not supported
    """
    if dataset_type not in DATASET_REGISTRY:
        available_types = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Available: {available_types}")

    dataset_class = DATASET_REGISTRY[dataset_type]
    return dataset_class(config, **kwargs)


def get_available_datasets():
    """Get list of available dataset types."""
    return list(DATASET_REGISTRY.keys())


__all__ = [
    "BaseSynthesisDataset",
    "VctkSynthesisDataset",
    "DATASET_REGISTRY",
    "create_dataset",
    "get_available_datasets"
]