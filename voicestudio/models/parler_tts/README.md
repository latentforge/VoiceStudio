# Parler-TTS

Parler-TTS conditions speech generation on a free-form natural language description of the voice: a T5 text encoder encodes the description, and a decoder-only audio language model cross-attends to it while autoregressively predicting DAC residual codec tokens, with each of the 9 codebooks offset by one step under a delay pattern. The transcript to speak is tokenized by the same tokenizer and prepended to the decoder input as prompt embeddings, or routed through cross-attention when `prompt_cross_attention` is set. The generated codes are decoded back to a 44.1 kHz waveform by the `DacModel` audio codec the model owns internally.

Original model and code: [huggingface/parler-tts](https://github.com/huggingface/parler-tts)


## Usage

The published checkpoints ship the audio codec in the original `descript-audio-codec` config and weight layout, so they need a one-time conversion before they load:

```python
from voicestudio.models.parler_tts.weight_conversion import convert

model_id = convert("parler-tts/parler-tts-mini-v1", "parler-tts-mini-v1-converted")
```

```python
import torch
from transformers import AutoModelForTextToWaveform, AutoProcessor

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id, dtype=torch.float16).to("cuda")
```

```python
import soundfile as sf

inputs = processor(
    description="Jon's voice is monotone yet slightly fast in delivery, with a very close recording that "
    "almost has no background noise.",
    transcript="Hey, how are you doing today?",
).to(model.device)

generation = model.generate(**inputs)
sf.write("output.wav", generation.float().cpu().numpy().squeeze(), model.config.audio_encoder.sampling_rate)
```

One thing above is load-bearing: `num_return_sequences` must stay at its default of 1. Values above 1 raise,
because the delay-pattern logits processor and the post-processing reshape are both sized from the
unexpanded batch.

`generate` already returns a finished waveform, not codes: `ParlerTTSForConditionalGeneration` owns the
`DacModel` codec as `self.audio_encoder` and decodes internally before returning, the same shape MusicGen
uses, so no separate `.decode()` call is needed in the flow above. `ParlerTTSProcessor` still carries its
own `audio_tokenizer`, loaded standalone from the checkpoint's `audio_encoder` subfolder (written by
`weight_conversion.convert`), for decoding DAC codes obtained some other way than `generate`.

`ParlerTTSForConditionalGeneration` also accepts raw target audio for training, through `input_values`
rather than through the processor: the model encodes it into codes with its own `audio_encoder` to derive
`decoder_input_ids` when neither `decoder_input_ids` nor `labels` are given (see `forward`). That path,
together with `from_sub_models_config`/`from_sub_models_pretrained`, is why `audio_encoder` stays a
submodule of the model itself, sharing one checkpoint with `text_encoder` and `decoder`, rather than moving
out to the processor entirely.
