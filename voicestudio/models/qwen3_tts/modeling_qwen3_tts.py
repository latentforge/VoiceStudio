try:
    from ..._qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSPreTrainedModel,
        Qwen3TTSTalkerTextPreTrainedModel,
        Qwen3TTSTalkerCodePredictorModel,
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
        Qwen3TTSTalkerModel,
        Qwen3TTSTalkerForConditionalGeneration,
        Qwen3TTSForConditionalGeneration,
    )
except ImportError:
    from voicestudio._qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSPreTrainedModel,
        Qwen3TTSTalkerTextPreTrainedModel,
        Qwen3TTSTalkerCodePredictorModel,
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
        Qwen3TTSTalkerModel,
        Qwen3TTSTalkerForConditionalGeneration,
        Qwen3TTSForConditionalGeneration,
    )
