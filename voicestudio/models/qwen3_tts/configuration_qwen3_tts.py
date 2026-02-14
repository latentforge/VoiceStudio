try:
    from ..._qwen3_tts.core import Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV2Config
    from ..._qwen3_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSSpeakerEncoderConfig, Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig, Qwen3TTSConfig
    )
except ImportError:
    from voicestudio._qwen3_tts.core import Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV2Config
    from voicestudio._qwen3_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSSpeakerEncoderConfig, Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig, Qwen3TTSConfig
    )
