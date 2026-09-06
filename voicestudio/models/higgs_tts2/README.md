# Higgs TTS 2

This folder only re-exports `HiggsTTS2Config`, `HiggsTTS2TokenizerConfig`, `HiggsTTS2Model`, `HiggsTTS2ForConditionalGeneration`, `HiggsTTS2PreTrainedModel`, `HiggsTTS2Processor`, and `HiggsTTS2TokenizerModel`; it does not vendor or reimplement the model.

Original model and code: [boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio)


## Usage

```python
from voicestudio.models.higgs_tts2 import HiggsTTS2ForConditionalGeneration, HiggsTTS2Processor

model_id = "bosonai/higgs-tts-2-3b-base"

processor = HiggsTTS2Processor.from_pretrained(model_id)
model = HiggsTTS2ForConditionalGeneration.from_pretrained(model_id)
model.to("cuda")
processor.audio_tokenizer.to(model.device)
```

```python
import soundfile as sf

conversation = [
    {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
    {"role": "user", "content": [{"type": "text", "text": "The sun rises in the east."}]},
]
inputs = processor.apply_chat_template(
    conversation,
    return_dict=True,
    tokenize=True,
    add_generation_prompt=True,
    sampling_rate=24000,
    return_tensors="pt",
).to(model.device)

audio_codes = model.generate(**inputs, max_new_tokens=1024)
waveform = processor.decode(audio_codes)
sf.write("output.wav", waveform.numpy(), processor.audio_tokenizer.config.sample_rate)
```

Three arguments above are load-bearing:

- `add_generation_prompt=True` appends the `<|audio_out_bos|>` token that opens the audio stream.
  Without it `generate` emits an end-of-stream frame almost immediately and the decoded waveform is
  a fraction of a second of noise, with no error raised.
- `max_new_tokens` bounds the audio stream. The checkpoint's `generation_config` sets no length, so
  without it `generate` falls back to the model-agnostic `max_length=53` default, which for a 33
  token prompt is 20 audio frames, about half a second of speech cut off mid-word.
- `processor.audio_tokenizer.to(model.device)` puts the codec on the same device as the generated
  codes. `HiggsAudioV2Processor.decode` does not move them itself, so a CUDA-resident model with a
  CPU-resident audio tokenizer raises a device mismatch in the codec's first linear layer.

The `Auto` classes reach the same two objects. The model ships in `transformers` itself, which the
names above alias, so this route needs no registration from this repository and works without
importing `voicestudio.models` first:

```python
from transformers import AutoModelForTextToWaveform, AutoProcessor

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id)
```
