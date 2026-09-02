# CosyVoice v2

CosyVoice v2 keeps v1's three-stage design (speech-token language model, conditional flow matching mel decoder, HiFTNet vocoder) but replaces the Conformer speech-token language model with a pretrained Qwen2 backbone and makes the flow decoder causal.
`CosyVoiceV2LLM` uses `Qwen2Model` directly, with no separate text encoder and no speaker embedding in the language model input; the checkpoint stores that backbone under a `Qwen2ForCausalLM` namespace, so `weight_conversion.py` remaps it and drops the unused `lm_head`.
The flow front end is `CosyVoiceV2UpsampleConformerEncoder`: the pre-lookahead convolution is folded inside the encoder, a 6-layer token-rate Conformer stack is followed by a fixed 2x nearest-neighbor `CosyVoiceV2Upsample1D` and an independently re-embedded 4-layer mel-rate Conformer stack, with the final `LayerNorm` applied once at the very end rather than after each stack. There is no length regulator, since the token-rate to mel-rate ratio is fixed by construction.

`CosyVoiceV2CausalConditionalDecoder` and `CosyVoiceV2FlowMatchingModel` subclass the CosyVoice v1 flow-matching classes in `voicestudio/models/cosyvoice_v1/`, mirroring the original repository's own `CausalMaskedDiffWithXvec` / `CausalConditionalDecoder` inheritance, and the vocoder (`CosyVoiceV1HiFTGenerator`) is reused unchanged because the v1 and v2 checkpoints instantiate the same `HiFTGenerator` class with only different upsample rates in the config. The vendored upstream source for all three CosyVoice versions lives in `voicestudio/models/cosyvoice_v1/cosyvoice/`.

Checkpoint: `FunAudioLLM/CosyVoice2-0.5B`, 24 kHz output.

Original model and code: [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice)


## Usage

```python
from transformers import AutoModel, AutoProcessor

model_id = "FunAudioLLM/CosyVoice2-0.5B"

processor = AutoProcessor.from_pretrained(model_id, speech_tokenizer_filename="speech_tokenizer_v2.onnx")
model = AutoModel.from_pretrained(model_id)
model.to("cuda")
```

`speech_tokenizer_filename` has to be passed explicitly: `CosyVoiceV2Processor` inherits `CosyVoiceV1Processor.from_pretrained`, whose default names the v1 tokenizer.

Zero-shot voice cloning from a reference clip:

```python
import soundfile as sf
import torchaudio

reference, sample_rate = torchaudio.load("reference.wav")
prompt_speech_token, _ = processor.extract_speech_token(reference, sample_rate)
embedding = processor.extract_speaker_embedding(reference, sample_rate)

inputs = processor(text="The sun rises in the east.").to(model.device)
waveform = model.generate(
    inputs["text_token"],
    embedding.to(model.device),
    prompt_speech_token=prompt_speech_token.to(model.device),
)

audio, output_sample_rate = processor.decode(waveform)
sf.write("output.wav", audio[0].cpu().numpy(), output_sample_rate)
```

Training uses the speech-token language model's next-token cross-entropy objective:

```python
outputs = model(
    text_token=inputs["text_token"],
    speech_token=speech_token,
    labels=labels,
)
outputs.loss.backward()
```
