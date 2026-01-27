try:
    from ..._qwen_tts.core.models.processing_qwen3_tts import (
        Qwen3TTSProcessor,
        Qwen3TTSProcessorKwargs,
    )
except ImportError:
    from voicestudio._qwen_tts.core.models.processing_qwen3_tts import (
        Qwen3TTSProcessor,
        Qwen3TTSProcessorKwargs,
    )
