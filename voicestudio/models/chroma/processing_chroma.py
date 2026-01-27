try:
    from ..._chroma.processing_chroma import (
        ChromaAudioKwargs,
        ChromaProcessorKwargs,
        ChromaProcessor,
    )
except ImportError:
    from voicestudio._chroma.processing_chroma import (
        ChromaAudioKwargs,
        ChromaProcessorKwargs,
        ChromaProcessor,
    )
