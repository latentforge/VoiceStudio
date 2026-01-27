try:
    from ..._qwen_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSSpeakerEncoderConfig,
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig,
        Qwen3TTSConfig,
    )
except ImportError:
    from voicestudio._qwen_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSSpeakerEncoderConfig,
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig,
        Qwen3TTSConfig,
    )