# Spark-TTS BiCodec

BiCodec is the audio tokenizer of Spark-TTS. It splits speech into two streams. A time-varying semantic stream comes
from a single factorized vector quantizer over the averaged hidden states of a frozen `wav2vec2-large-xlsr-53`, at 50
tokens per second over an 8192-entry codebook. A time-invariant global stream comes from an ECAPA-TDNN speaker
encoder whose frame features a perceiver resampler compresses into 32 latents, each quantized by a finite scalar
quantizer over `4^6` levels. Decoding conditions a ConvNeXt prenet on the flattened global latents through adaptive
layer normalization, adds them again to its output, and runs a DAC-style wave generator at 16 kHz.

The codec ships inside the Spark-TTS checkpoint rather than as a repository of its own, and
[`SparkTTSProcessor`](../spark_tts) holds it as its `audio_tokenizer`. It is a model in its own right and can be used
on its own.

Original model and code: [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS)


## Usage

The published `SparkAudio/Spark-TTS-0.5B` repo needs the one-time conversion in
[`voicestudio/models/spark_tts/weight_conversion.py`](../spark_tts/weight_conversion.py), which writes BiCodec and
the self-supervised model it reads features from to the `audio_tokenizer` subfolder of its output:

```python
from voicestudio.models.spark_tts.weight_conversion import convert

convert("SparkAudio/Spark-TTS-0.5B", "spark-tts-converted")
```

```python
import soundfile as sf
import torch

from voicestudio.models.spark_tts import SparkTTSFeatureExtractor
from voicestudio.models.spark_tts_bicodec import SparkTTSBiCodecModel

model_id = "spark-tts-converted"

feature_extractor = SparkTTSFeatureExtractor.from_pretrained(model_id)
model = SparkTTSBiCodecModel.from_pretrained(model_id, subfolder="audio_tokenizer").to("cuda")

audio, sampling_rate = sf.read("speech.wav")
inputs = feature_extractor(audio, sampling_rate=sampling_rate).to(model.device)

codes = model.encode(**inputs)
audio_values = model.decode(codes.audio_codes, codes.global_codes).audio_values
sf.write("reconstruction.wav", audio_values.reshape(-1).cpu().numpy(), feature_extractor.sampling_rate)
```

`encode` returns `audio_codes` of shape `(batch_size, num_frames)`, the semantic stream, and `global_codes` of shape
`(batch_size, num_quantizers, num_speaker_tokens)`, the speaker stream. Those are exactly the two token families
Spark-TTS's vocabulary carries as `<|bicodec_semantic_*|>` and `<|bicodec_global_*|>`, so `decode` accepts codes a
language model sampled just as readily as codes `encode` produced.

The reference clip the global stream describes is the leading `ref_segment_duration` seconds of the input, tiled
first if the input is shorter. `SparkTTSFeatureExtractor` cuts it and computes its mel spectrogram, and returns it as
`reference_input_features` alongside the `input_values` the self-supervised model runs on.


## Training

`SparkTTSBiCodecModel.forward` returns a `loss` that is the weighted sum

```
1.0 * vq_loss + 1.0 * feature_loss + 15.0 * mel_loss + 1.0 * speaker_loss
```

with each term also reported on its own in `SparkTTSBiCodecOutput`. The objective, the weights, the schedule and the
list of what upstream freezes are documented term by term in
[`voicestudio/models/spark_tts/README.md`](../spark_tts/README.md#bicodec), together with the parts of the upstream
codec objective that are not carried over. All of it is derived from
[SparkAudio/SparkVox](https://github.com/SparkAudio/SparkVox), which is where the codec is actually trained; the
inference-only `SparkAudio/Spark-TTS` repo ships no trainer.

```python
inputs = feature_extractor(audio, sampling_rate=sampling_rate, return_labels=True).to(model.device)

model.train()
model.freeze_semantic_model()
loss = model(**inputs, step=global_step).loss
```

`step` drives the `d_vector_train_start` schedule and `freeze_semantic_model` keeps the self-supervised feature
source in eval mode, which a plain `train()` would otherwise undo.
