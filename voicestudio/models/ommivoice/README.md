# OmniVoice

OmniVoice is a massively multilingual zero-shot text-to-speech model covering 600+ languages, with voice cloning
and voice design. It is a masked diffusion language model: a language-model backbone reads one interleaved stream
of text tokens and 8-codebook audio frames, and a fixed-length canvas of masked frames is filled in over a
handful of unmasking steps rather than one frame at a time. Attention is therefore bidirectional and no cache is
ever used.

Original model and code: [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)

## Lineage

The backbone is reused verbatim: `OmniVoiceModel` holds a `transformers` model built with
`AutoModel.from_config(config.llm_config)`, which for the released `k2-fsa/OmniVoice` checkpoint is a plain
`Qwen3Model`. The checkpoint's `llm.*` weights are Qwen3's exactly, down to the per-head `self_attn.q_norm` /
`self_attn.k_norm` that distinguish Qwen3 from Qwen2 and Llama, and it carries no `lm_head`, so it is
`Qwen3Model` rather than `Qwen3ForCausalLM`.

The fused multi-codebook audio embedding is `HiggsAudioV2Embeddings`, whose lookup (one table of
`num_codebooks * codebook_size` rows, indexed with a per-codebook offset, summed over the codebook axis) is the
same computation OmniVoice performs. `OmniVoiceAudioEmbeddings` only adapts the tensor layout, zeroes the text
positions that share the id tensor, and makes the offsets buffer persistent because the checkpoint ships it.
`OmniVoicePreTrainedModel` inherits `HiggsAudioV2PreTrainedModel` for the matching `_init_weights` branch, as
`voicestudio/models/higgs_tts3` does.

Audio is tokenized and reconstructed by `HiggsAudioV2TokenizerModel`, the same codec Higgs uses, bundled in the
checkpoint's `audio_tokenizer` subfolder.

Two things are expressed through `transformers` primitives rather than by hand. The bidirectional mask comes
from `create_bidirectional_mask` with `allow_is_bidirectional_skip=False`, since leaving the mask unset would
make the backbone build a causal one. Sequence packing goes through `packed_sequence_mask_function`, which makes
`document_ids` work under every attention implementation instead of only `flex_attention`.

## Usage

```python
import soundfile as sf

from voicestudio.models.ommivoice import OmniVoiceForConditionalGeneration, OmniVoiceProcessor

model_id = "k2-fsa/OmniVoice"

processor = OmniVoiceProcessor.from_pretrained(model_id)
model = OmniVoiceForConditionalGeneration.from_pretrained(model_id)
model.to("cuda")
processor.audio_tokenizer.to(model.device)
```

Voice design, describing the voice instead of supplying one:

```python
inputs = processor(
    text="Hello, this is a test of zero-shot voice design.",
    language="English",
    instruct="female, low pitch, british accent",
).to(model.device)

audio_codes = model.generate(**inputs)
sf.write("out.wav", processor.batch_decode(audio_codes)[0], processor.sampling_rate)
```

Voice cloning from a reference clip and its transcript:

```python
reference_audio, reference_sampling_rate = sf.read("ref.wav")

inputs = processor(
    text="Hello, this is a test of zero-shot voice cloning.",
    reference_audio=reference_audio,
    reference_text="Transcription of the reference audio.",
    sampling_rate=reference_sampling_rate,
).to(model.device)

audio_codes = model.generate(**inputs, num_step=32, guidance_scale=2.0)
waveform = processor.decode(
    audio_codes[0], reference_audio=reference_audio, sampling_rate=reference_sampling_rate
)
```

Passing neither `instruct` nor a reference lets the model pick a voice itself. Output length is estimated from
the text; `duration=10.0` fixes it in seconds and `speed=1.2` scales the estimate.

Text may carry non-verbal tags (`[laughter]`, `[sigh]`, ...), CMU pronunciations (`[B EY1 S]`) and pinyin tone
markers (`打ZHE2出售`); the processor keeps each of those as its own token run.

`generate` returns codes of shape `(batch_size, num_codebooks, max_target_length)`; items shorter than the
longest one keep `config.audio_mask_id` in their trailing frames, which `decode` trims.

### Training

`forward` accepts `labels` and returns the codebook-weighted cross-entropy loss of the original implementation:
the mean loss of each codebook over the positions it had to predict, weighted by `config.audio_codebook_weights`
normalized to sum to one.

```python
target_audio, target_sampling_rate = sf.read("target.wav")

inputs = processor(
    text="Hello, this is a training example.",
    language="English",
    audio_codes=processor.encode_audio(target_audio, target_sampling_rate),
    prompt_ratio=(0.0, 0.5),
    mask_ratio=(0.2, 1.0),
    output_labels=True,
)

loss = model(**inputs).loss
loss.backward()
```

`drop_conditioning=True` builds the unconditional example that classifier-free guidance is trained against.
Packing several examples into one sequence is supported by passing `document_ids` alongside `position_ids`.

## Not migrated

The following pieces of the upstream repository are deliberately absent and are still open:

- **Automatic long-text chunking.** Upstream `generate` splits any text estimated over 30 s into ~15 s chunks,
  synthesizes them one after another (feeding the first chunk back as the reference when no reference audio was
  given), and cross-fades the results. `OmniVoiceProcessor.chunk_text` and the cross-fading branch of
  `OmniVoiceProcessor.decode` are migrated, so the loop can be written by hand, but nothing drives it
  automatically.
- **Silence removal.** Upstream trims reference audio longer than 20 s at its largest silence gap, strips
  leading/trailing and long interior silences from both the reference and the output, all through `pydub`, which
  is not a dependency of this repository. `OmniVoiceProcessor` warns about long references and applies the
  loudness matching, fades and edge padding, but performs no silence detection.
- **Whisper auto-transcription.** Upstream loads `openai/whisper-large-v3-turbo` inside the TTS model to fill in
  a missing `reference_text`. `reference_text` must be supplied here.
- **Optional text normalization.** Upstream `generate(normalize_text=True)` rewrites numbers, dates and currency
  into their spoken form through WeTextProcessing (Chinese/English) or `num2words`, protecting the inline
  control syntax. Neither package is a dependency of this repository.
- **FlashInfer inference path.** Upstream ships `omnivoice_flashinfer.py`, which patches fused RMSNorm/RoPE/GEMM
  kernels and CUDA graphs over the backbone for a 2-2.9x speedup.
- **Training, evaluation and CLI trees.** The WebDataset readers, length-grouped batching, trainer, LoRA
  helpers, evaluation harness and `omnivoice-*` command line tools are infrastructure rather than model code.
