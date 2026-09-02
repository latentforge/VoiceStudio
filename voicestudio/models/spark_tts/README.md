# Spark-TTS

Spark-TTS is a single-stage LLM text-to-speech model: a Qwen2 decoder whose vocabulary is extended with the tokens of BiCodec, its own audio tokenizer, so that synthesis, zero-shot voice cloning and attribute control are all next-token prediction over one flat sequence. There is no separate flow-matching or diffusion stage.

BiCodec splits speech into two streams. The time-varying semantic stream is a single 8192-entry factorized codebook over wav2vec2-large-xlsr-53 hidden states, at 50 tokens per second. The time-invariant global stream is 32 finite-scalar-quantized tokens describing the speaker, produced by an ECAPA-TDNN over a 6 second reference mel spectrogram followed by a perceiver resampler. Reconstruction runs the semantic codes through a ConvNeXt prenet conditioned on the global embedding, then through a DAC-style wave generator at 16 kHz.

This implementation reuses `Qwen2Model` for the language model and `transformers`' own `Snake1d`/`DacResidualUnit` for the wave generator's residual units, which BiCodec adapted from DAC unchanged. The rest of BiCodec is written directly in this folder: its ConvNeXt backbone, its factorized vector quantizer, its wespeaker-style ECAPA-TDNN and its perceiver resampler each differ in their internals from the nearest classes already in `transformers` (`Xcodec2`'s finite scalar quantizer, `VibeVoiceAcousticTokenizer`'s ConvNeXt layer, `Qwen3TTSTokenizerSingleCodebook`'s speechbrain-style ECAPA-TDNN, `Idefics2`'s perceiver resampler).

Original model and code: [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS)


## Usage

The published checkpoint is three independently saved models in three subfolders plus two YAML files, so it needs a one-time conversion before it loads. The converted directory holds the language model, its tokenizer and the feature extractor, with BiCodec and its wav2vec2 semantic model under `audio_tokenizer/`:

```python
from voicestudio.models.spark_tts.weight_conversion import convert

model_id = convert("SparkAudio/Spark-TTS-0.5B", "Spark-TTS-0.5B-converted")
```

```python
import torch
from voicestudio.models.spark_tts import SparkTTSForConditionalGeneration, SparkTTSProcessor

processor = SparkTTSProcessor.from_pretrained(model_id)
model = SparkTTSForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32).to("cuda")
processor.audio_tokenizer.to(model.device)
```

Building a voice from attribute labels, which makes the model emit its own global tokens:

```python
import soundfile as sf

inputs = processor(
    text="The sun rises in the east.",
    gender="female",
    pitch="moderate",
    speed="moderate",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=3000, do_sample=True, temperature=0.8, top_k=50, top_p=0.95)
waveform = processor.decode(outputs, input_length=inputs["input_ids"].shape[-1])
sf.write("output.wav", waveform.numpy(), model.config.sampling_rate)
```

Zero-shot voice cloning from a reference clip, whose global tokens are prefixed to the prompt. Passing `prompt_text`, the transcript of the clip, additionally prefixes the clip's own semantic tokens so that generation continues it:

```python
reference_audio, sampling_rate = sf.read("reference.wav")
inputs = processor(
    text="The sun rises in the east.",
    reference_audio=reference_audio,
    prompt_text="Transcript of reference.wav.",
    sampling_rate=sampling_rate,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=3000, do_sample=True, temperature=0.8, top_k=50, top_p=0.95)
waveform = processor.decode(outputs, input_length=inputs["input_ids"].shape[-1])
```

`processor.decode` reads global tokens from the whole sequence, since a cloning prompt carries them and an attribute prompt makes the model emit them, and reads semantic tokens only from beyond `input_length`, so a reference clip's own semantic tokens do not leak into the output.

`convert` records the absolute path of the `audio_tokenizer/` subdirectory in `processor_config.json`. Publishing the converted checkpoint means either uploading that subdirectory as its own repository and pointing `audio_tokenizer_name_or_path` at its repo id, or passing an explicit `audio_tokenizer=` to `SparkTTSProcessor`.

BiCodec is usable on its own through `SparkTTSBiCodecModel`, whose `forward` returns the reconstruction together with the codes, the postnet feature prediction and the quantizer's commitment/codebook loss.
