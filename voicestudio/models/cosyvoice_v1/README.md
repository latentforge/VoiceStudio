# CosyVoice v1

CosyVoice is a scalable multilingual zero-shot text-to-speech synthesizer built on supervised semantic tokens: a text-to-token language model predicts discrete speech tokens, a conditional flow matching module converts those tokens to mel-spectrograms, and a HiFi-GAN vocoder renders the waveform.
In v1 both the text encoder (`CosyVoiceV1TextEncoder`) and the speech-token language model (`CosyVoiceV1LLM`) are relative-position Conformer stacks rather than a pretrained LLM backbone, so they share `CosyVoiceV1RelPositionEncoder`, which also accepts a precomputed causal attention bias and therefore serves both the bidirectional text encoder and the causal speech-token decoder. `CosyVoiceV1EncoderLayer` borrows `Wav2Vec2ConformerSelfAttention` and `Wav2Vec2ConformerFeedForward` from `transformers` but is a pre-norm single-feed-forward block with no macaron feed-forward and no depthwise convolution module, matching the original CosyVoice/WeNet `TransformerEncoderLayer`, plus the input `Linear` and `LayerNorm` projection the Conformer lineage does not have.

The flow-matching estimator (`CosyVoiceV1ConditionalDecoder`) is a U-Net whose transformer blocks match the `diffusers`-style `BasicTransformerBlock` that the original instantiates through `Matcha-TTS`: separate unbiased `to_q`/`to_k`/`to_v` projections with a biased `to_out`, plain `LayerNorm` (the flow-matching timestep is injected only into the surrounding `ResnetBlock1D` via FiLM, never into the attention block), and a plain non-gated GELU feed-forward. The vocoder (`CosyVoiceV1HiFTGenerator`) is a HiFTNet neural-source-filter plus ISTFT generator with learnable Snake activations.

This folder also holds the vendored upstream CosyVoice source under `cosyvoice/`, which covers all three versions as a single inheritance chain and so is not split; `voicestudio/models/cosyvoice_v2/` and `voicestudio/models/cosyvoice_v3/` contain only their migrated files and subclass the classes here. See `INFO.md` for the upstream project's own README.

Checkpoint: `FunAudioLLM/CosyVoice-300M`, 22.05 kHz output.

Original model and code: [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice)


## Usage

```python
from transformers import AutoModel, AutoProcessor

model_id = "FunAudioLLM/CosyVoice-300M"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.to("cuda")
```

Zero-shot voice cloning from a reference clip. `CosyVoiceV1Processor` runs the checkpoint's `speech_tokenizer_v1.onnx` and `campplus.onnx` directly through `onnxruntime`, since neither has a `transformers` equivalent:

```python
import soundfile as sf
import torchaudio

reference, sample_rate = torchaudio.load("reference.wav")
prompt_speech_token, _ = processor.extract_speech_token(reference, sample_rate)
embedding = processor.extract_speaker_embedding(reference, sample_rate)

inputs = processor(text="The sun rises in the east.").to(model.device)
waveform = model.generate(
    inputs["text_token"],
    inputs["text_token_len"],
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
    text_token_len=inputs["text_token_len"],
    speech_token=speech_token,
    speech_token_len=speech_token_len,
    embedding=embedding,
    labels=labels,
)
outputs.loss.backward()
```
