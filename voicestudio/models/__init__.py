import importlib


__all__ = [
    "chroma",
    "cosyvoice_v1",
    "cosyvoice_v2",
    "cosyvoice_v3",
    "dia",
    "f5_tts",
    "higgs_audio_v2",
    "higgs_audio_v3",
    "omnivoice",
    "parler_tts",
    "prompt_tts_pp",
    "qwen3_tts",
    "spark_tts",
]


def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
