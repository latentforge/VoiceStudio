# Parler-TTS

Parler-TTS conditions speech generation on a free-form natural language description of the voice: a T5 text encoder encodes the description, and a decoder-only audio language model cross-attends to it while autoregressively predicting DAC residual codec tokens, with each of the 9 codebooks offset by one step under a delay pattern. The transcript to speak is tokenized by the same tokenizer and prepended to the decoder input as prompt embeddings, or routed through cross-attention when `prompt_cross_attention` is set. The generated codes are decoded back to a 44.1 kHz waveform by the `DacModel` audio codec.

Original model and code: [huggingface/parler-tts](https://github.com/huggingface/parler-tts)


## Usage

The published checkpoints ship the audio codec in the original `descript-audio-codec` config and weight layout, so they need a one-time conversion before they load:

```python
from voicestudio.models.parler_tts.weight_conversion import convert

model_id = convert("parler-tts/parler-tts-mini-v1", "parler-tts-mini-v1-converted")
```

```python
import torch
from transformers import AutoTokenizer
from voicestudio.models.parler_tts import ParlerTTSForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ParlerTTSForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32).to("cuda")
```

```python
import soundfile as sf

description = tokenizer(
    "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.",
    return_tensors="pt",
).to(model.device)
prompt = tokenizer("Hey, how are you doing today?", return_tensors="pt").to(model.device)

generation = model.generate(
    input_ids=description.input_ids,
    attention_mask=description.attention_mask,
    prompt_input_ids=prompt.input_ids,
    prompt_attention_mask=prompt.attention_mask,
)
sf.write("output.wav", generation.cpu().numpy().squeeze(), model.config.audio_encoder.sampling_rate)
```

`num_return_sequences` must stay at its default of 1; values above 1 raise, because the delay-pattern logits processor and the post-processing reshape are both sized from the unexpanded batch.
