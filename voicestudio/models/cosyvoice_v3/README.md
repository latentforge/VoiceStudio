# CosyVoice v3

CosyVoice v3 (Fun-CosyVoice3) keeps v2's Qwen2-backbone speech-token language model and replaces the U-Net flow-matching estimator with a diffusion transformer and the vocoder with a causal HiFTNet.
`CosyVoiceV3LLM` subclasses `CosyVoiceV2LLM`, mirroring the original repository's own `CosyVoice3LM(Qwen2LM)` inheritance; it extends the speech-token vocabulary so the start, task, fill, and end-of-speech ids live in the speech-token embedding table itself rather than in a separate two-entry embedding.

`CosyVoiceV3DiT` is the flow decoder's estimator: each `CosyVoiceV3DiTBlock` has separate `to_q`/`to_k`/`to_v`/`to_out` projections with interleaved rotary position embeddings, AdaLN-Zero modulation recomputed every block (6-way before attention and the feed-forward, 2-way at the final layer), and a plain, non-gated tanh-GELU feed-forward. The surrounding `CosyVoiceV3FlowMatchingModel` has no Conformer text encoder and no length regulator: `CosyVoiceV3PreLookaheadLayer` applies a widen-then-narrow convolutional bottleneck to the token embedding, which is then upsampled straight to the mel rate by `repeat_interleave`.
The vocoder is `CosyVoiceV3HiFTGenerator`, a causal generator with three `[8, 5, 3]` upsample stages each built from a `Conv1d` plus `Upsample` pair (a different weight layout than `ConvTranspose1d`), causal-padded ResBlocks and `conv_pre`/`conv_post`, and a `CosyVoiceV3F0Predictor` whose first layer has kernel size 4. It is not the `CosyVoiceV1HiFTGenerator` reused by v2.

Everything else is inherited: `CosyVoiceV3LLM` from `voicestudio/models/cosyvoice_v2/`, and the conditional-flow-matching solver, sine generator, and Snake activation from `voicestudio/models/cosyvoice_v1/`. The vendored upstream source for all three CosyVoice versions lives in `voicestudio/models/cosyvoice_v1/cosyvoice/`.

Checkpoint: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`.

Original model and code: [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice)


## Usage

```python
from transformers import AutoModel, AutoProcessor

model_id = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.to("cuda")
```

Zero-shot voice cloning from a reference clip. CosyVoice v3 expects the tokenized text to contain an `<|endofprompt|>` marker separating the instruction span from the text to synthesize:

```python
import soundfile as sf
import torchaudio

reference, sample_rate = torchaudio.load("reference.wav")
prompt_speech_token, _ = processor.extract_speech_token(reference, sample_rate)
embedding = processor.extract_speaker_embedding(reference, sample_rate)

inputs = processor(text="Speak in a calm voice.<|endofprompt|>The sun rises in the east.").to(model.device)
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
