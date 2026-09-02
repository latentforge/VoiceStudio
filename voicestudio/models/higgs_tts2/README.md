# Higgs TTS 2

This folder only re-exports `HiggsTTS2Config`, `HiggsTTS2TokenizerConfig`, `HiggsTTS2Model`, `HiggsTTS2ForConditionalGeneration`, `HiggsTTS2PreTrainedModel`, `HiggsTTS2Processor`, and `HiggsTTS2TokenizerModel`; it does not vendor or reimplement the model.

Original model and code: [boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio)


## Usage

```python
from transformers import AutoModelForTextToWaveform, AutoProcessor

model_id = "bosonai/higgs-tts-2-3b-base"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id)
model.to("cuda")
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

audio_codes = model.generate(**inputs)
waveform = processor.decode(audio_codes)
sf.write("output.wav", waveform.numpy(), processor.audio_tokenizer.config.sample_rate)
```

`add_generation_prompt=True` is required: it appends the `<|audio_out_bos|>` token that opens the
audio stream. Without it `generate` emits an end-of-stream frame almost immediately and the decoded
waveform is a fraction of a second of noise, with no error raised.
