# Higgs TTS 3

Higgs TTS 3 pairs a plain [`Qwen3Model`] text backbone with a fused multi-codebook audio embedding and output head, replacing v2's custom Llama-derived dual-FFN decoder.
This implementation reuses `Qwen3Model` for the backbone and its `HiggsTTS2Embeddings`/`HiggsTTS2PreTrainedModel` for the fused audio embedding and shared `PreTrainedModel` plumbing, since those pieces are architecturally unchanged from v2.
The backbone itself (`HiggsTTS3Model`, `HiggsTTS3ForConditionalGeneration`) is written directly in this folder rather than inherited from `HiggsTTS3Model`, because v2's decoder layer bakes the dual-FFN audio/text routing into every layer, which v3 does not have; `HiggsTTS2Model` could not be reused as-is for the v3 backbone.

Original model and code: [boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio)


## Usage

```python
from transformers import AutoModelForTextToWaveform, AutoProcessor

model_id = "bosonai/higgs-tts-3-4b"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id)
model.to("cuda")
processor.audio_tokenizer.to(model.device)
```

```python
import soundfile as sf

inputs = processor(text="The sun rises in the east.").to(model.device)

audio_codes = model.generate(**inputs, max_new_tokens=1024, temperature=0.8, top_k=50)
waveform = processor.decode(audio_codes)
sf.write("output.wav", waveform.numpy(), model.config.sample_rate)
```

Zero-shot voice cloning from a reference clip, with its transcript:

```python
reference_audio, _ = sf.read("reference.wav")
inputs = processor(
    text="The sun rises in the east.",
    reference_audio=reference_audio,
    reference_text="Transcript of reference.wav.",
).to(model.device)

audio_codes = model.generate(**inputs, max_new_tokens=1024, temperature=0.8, top_k=50)
waveform = processor.decode(audio_codes)
```

The prompt the processor builds is framed by the checkpoint's own specials, not by its chat
template: `<|tts|>` opens the prompt, `<|ref_text|>`/`<|ref_audio|>` carry an optional voice-cloning
reference, `<|text|>` introduces the target text, and a trailing `<|audio|>` opens the audio stream.
Prompting the model through `tokenizer.apply_chat_template` instead produces degenerate babble that
never terminates.

`bosonai/higgs-tts-3-4b` ships no `preprocessor_config.json` and no audio tokenizer weights of its
own, so `HiggsTTS3Processor.from_pretrained` loads both from the codec repository named by
`config.audio_tokenizer_id`, which is `bosonai/higgs-audio-v2-tokenizer`.
