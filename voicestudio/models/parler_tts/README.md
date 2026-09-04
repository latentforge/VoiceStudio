# Parler-TTS

Parler-TTS conditions speech generation on a free-form natural language description of the voice: a T5 text encoder encodes the description, and a decoder-only audio language model cross-attends to it while autoregressively predicting DAC residual codec tokens, with each of the 9 codebooks offset by one step under a delay pattern. The transcript to speak is tokenized by the same tokenizer and prepended to the decoder input as prompt embeddings, or routed through cross-attention when `prompt_cross_attention` is set. The generated codes are decoded back to a 44.1 kHz waveform by the `DacModel` audio codec the model owns internally.

Original model and code: [huggingface/parler-tts](https://github.com/huggingface/parler-tts)


## Usage

`from_pretrained` takes any published repository id as it stands:

```python
import torch
from transformers import AutoModelForTextToWaveform, AutoProcessor

model_id = "parler-tts/parler-tts-mini-v1"

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

A published checkpoint holds its audio codec as the `descript-audio-codec` module Parler-TTS trained with, in
that module's own weight layout, described by a vendored `DACConfig` serialized under `model_type: "dac"` that
declares the codec's bitrate and latent width and none of the architecture hyperparameters. Both calls above read
that layout as it stands. `ParlerTTSConfig` builds the `DacConfig` the entry describes, and
`ParlerTTSForConditionalGeneration.from_pretrained` folds the codec's weight norm reparameterization back into
plain weights and maps its 301 tensors onto `DacModel`'s 223 on the way in.

Three things about that path are load-bearing:

- `num_return_sequences` must stay at its default of 1. Values above 1 raise, because the delay-pattern logits
  processor and the post-processing reshape are both sized from the unexpanded batch.
- `generate` already returns a finished waveform, not codes: `ParlerTTSForConditionalGeneration` owns the
  `DacModel` codec as `self.audio_encoder` and decodes internally before returning, the same shape MusicGen uses,
  so no separate `.decode()` call is needed in the flow above.
- `ParlerTTSProcessor` still carries its own `audio_tokenizer`, for decoding DAC codes obtained some other way
  than `generate`. It is read out of the published checkpoint's own weights, or out of the `audio_encoder`
  subfolder of a converted directory, the same one `ParlerTTSForConditionalGeneration.from_pretrained` writes
  and caches, so loading the processor after the model touches no published weight file again.

`ParlerTTSForConditionalGeneration` also accepts raw target audio for training, through `input_values` rather
than through the processor: the model encodes it into codes with its own `audio_encoder` to derive
`decoder_input_ids` when neither `decoder_input_ids` nor `labels` are given (see `forward`). That path, together
with `from_sub_models_config`/`from_sub_models_pretrained`, is why `audio_encoder` stays a submodule of the model
itself, sharing one checkpoint with `text_encoder` and `decoder`, rather than moving out to the processor
entirely.

A first load converts the published layout into a directory under `HF_HOME`, keyed on the repository and the commit it
resolved to, and later loads read that directory through the ordinary loading path instead of converting again. The
key comes from `config.json` alone, so a cache hit resolves nothing beyond that file; once the conversion is
written, the checkpoint's safetensors shards are dropped from the `huggingface_hub` cache.

`weight_conversion.convert` writes that same conversion to a directory of the caller's choosing, for a checkpoint
that is shipped elsewhere or kept outside the cache. It also saves the codec standalone under an `audio_encoder`
subfolder, and both classes load the result without converting anything again:

```python
from voicestudio.models.parler_tts.weight_conversion import convert

convert("parler-tts/parler-tts-mini-v1", "parler-tts-mini-v1-converted")
```
