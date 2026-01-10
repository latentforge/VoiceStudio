"""
Configuration module for synthesis pipeline.
"""

from enum import Enum

from .synthesis_config import (
    DatasetConfig,
    GenerationConfig,
    ModelConfig,
    SynthesisConfig,
)


class DatasetType(Enum):
    """Supported dataset types."""

    VCTK = "vctk"
    LJSPEECH = "ljspeech"
    LIBRITTS = "libritts"


class ModelType(Enum):
    """Supported model types."""

    XTTS_V2 = "xtts_v2"
    PARLER_TTS_MINI_V1 = "parler_tts_mini_v1"
    HIGGS_V2 = "higgs_v2"
    DIA_TTS = "dia_tts"
    CHATTERBOX_TTS = "chatterbox_tts"


class GenerationMethod(Enum):
    """Generation method types."""

    METHOD1 = "method1"  # 100 1:1 pairs
    METHOD2 = "method2"  # 10 refs × 10 syn each
    METHOD3 = "method3"


__all__ = [
    "DatasetConfig",
    "ModelConfig",
    "GenerationConfig",
    "SynthesisConfig",
    "DatasetType",
    "ModelType",
    "GenerationMethod",
]
