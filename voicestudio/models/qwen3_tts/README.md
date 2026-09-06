# Qwen3-TTS

Qwen3-TTS pairs a Qwen3-based talker language model, which autoregressively predicts the first-layer speech token per step, with a separate code-predictor submodule that fills in the remaining residual codebooks, conditioned through a speaker encoder for preset/custom voices. Each checkpoint is trained for a single task (`base`, `custom_voice`, or `voice_design`); this folder's `Qwen3TTSProcessor` subclass adds `encode`/`encode_voice_design`/`encode_custom_voice` task-dispatch methods on top of the relayed `transformers` classes, raising `RuntimeError` on a task/checkpoint mismatch.

Original model and code: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)


## Usage

```python
import torch
from voicestudio.models.qwen3_tts import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

processor = Qwen3TTSProcessor.from_pretrained(model_id)
model = Qwen3TTSForConditionalGeneration.from_pretrained(model_id, dtype=torch.float16).to("cuda")
processor.audio_tokenizer.to(model.device)
```

The published weight layout loads as it stands. A conversion mapping registered against `Qwen3TTSForConditionalGeneration` drops the checkpoint's `talker.` prefix, renames its codec embedding and text projection, and concatenates its per-codebook output heads into the single fused head the code predictor declares. `Qwen3TTSConfig` reads the `rope_scaling` and `rope_theta` keys the checkpoint records, at every depth of the configuration, as the single `rope_parameters` mapping the classes it inherits from expect. Both the talker and its nested code predictor carry a rope base of 1000000 that way, which is the base the upstream `qwen_tts` package serves them at. The `speech_tokenizer` subfolder the processor reads is in the original Qwen3-TTS-Tokenizer-12Hz format, which loads as it stands too: a second conversion mapping, registered against `Qwen3TTSTokenizerMultiCodebookConfig`, renames the decoder quantizer's `rvq_first`/`rvq_rest` keys as the tokenizer loads, and the model is built from the defaults that layout implies since its own configuration is of a schema `Qwen3TTSTokenizerMultiCodebookConfig` does not read.

`generate` returns one list of audio codes and one list of talker hidden states, one entry each per sample:

```python
import soundfile as sf

inputs = processor.encode_voice_design(
    text="The sun rises in the east and sets in the west.",
    instruct="A calm, warm female voice.",
).to(model.device)

audio_codes, _ = model.generate(**inputs)
waveform = processor.decode(audio_codes[0])
sf.write("output.wav", waveform.float().cpu().numpy(), processor.audio_tokenizer.config.output_sample_rate)
```

`encode_voice_design` and `encode_custom_voice` carry `non_streaming_mode=True`, which puts the whole text in the talker's prefill. Passing `non_streaming_mode=False` to `generate` instead leaves only the first text token there and feeds one further text token per generated codec frame, so the model begins speaking before it has seen the rest of the sentence and can open on a few seconds of unrelated speech.

