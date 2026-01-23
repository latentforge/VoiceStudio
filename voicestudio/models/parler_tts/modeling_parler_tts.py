try:
    from ...._parler_tts.modeling_parler_tts import ParlerTTSForCausalLM, ParlerTTSForConditionalGeneration
except ImportError:
    from voicestudio._parler_tts.modeling_parler_tts import ParlerTTSForCausalLM, ParlerTTSForConditionalGeneration
