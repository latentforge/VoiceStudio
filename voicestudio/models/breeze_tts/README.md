# Breeze TTS 2

Breeze TTS 2 is a real-time instruction-following text-to-speech model built on a Sesame CSM-style backbone plus depth decoder over multi-codebook audio tokens, with a T5Gemma 2 text encoder conditioning the backbone for voice design and voice direction from natural-language instructions.
This implementation inherits `CsmConfig`/`CsmDepthDecoderConfig` and `CsmDepthDecoderModel`/`CsmDepthDecoderForCausalLM`/`CsmForConditionalGeneration`/`CsmGenerationMixin`, whose layer stack, depth decoder, codebook head and frame-per-step decoding loop the upstream source reproduces line for line.
The backbone is `Qwen3Model`, which is what the released checkpoint's `backbone_model_type` names, extended with the summed multi-codebook frame embedding and the DimFusion text-encoder conditioning the upstream backbone adapter applies between layers.
The text encoder is the native `T5Gemma2TextEncoder` and the codec is the native `MimiModel`, both of which the released checkpoint carries as sub-configs.

Original model and code: [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts)


## Usage

```python
from transformers import AutoModelForTextToWaveform, AutoProcessor

model_id = "BreezeBlue/breeze-tts-2"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id)
model.to("cuda")
processor.audio_tokenizer.to(model.device)
```

Voice design, from a natural-language description of the voice alone:

```python
import soundfile as sf

inputs = processor(
    text="(sigh) It is good to hear your voice again after all this time.",
    instruction="A warm, low-pitched voice speaking slowly and thoughtfully.",
).to(model.device)

audio_codes = model.generate(**inputs, max_new_tokens=750, temperature=0.9, top_k=50)
waveform = processor.decode(audio_codes)[0]
sf.write("output.wav", waveform.numpy(), processor.audio_tokenizer.get_output_sample_rate())
```

Voice clone from a reference clip, with its exact transcript:

```python
reference_audio, _ = sf.read("reference.wav")
inputs = processor(
    text="The sun rises in the east.",
    reference_audio=reference_audio,
    reference_text="Transcript of reference.wav.",
).to(model.device)

audio_codes = model.generate(**inputs, max_new_tokens=750, temperature=0.9, top_k=50)
```

Voice direction steers a cloned voice with an instruction. Passing `guidance_scale` also builds the
negative prompt the instruction is guided against, which is the same prompt with the instruction
dropped; passing `guidance_scale_ref` and `guidance_scale_ins` instead guides the reference audio and
the instruction with separate scales, against a text-only unconditional branch:

```python
inputs = processor(
    text="The sun rises in the east.",
    instruction="Speak faster, and sound delighted.",
    reference_audio=reference_audio,
    reference_text="Transcript of reference.wav.",
    guidance_scale_ref=1.5,
    guidance_scale_ins=2.0,
).to(model.device)

audio_codes = model.generate(**inputs, max_new_tokens=750, temperature=0.9, top_k=50)
```

Every text span of a prompt opens with a speaker token, `[S0]` by default, and an instruction is
wrapped in `<ins_bos>`/`<ins_eos>` right before the target text. Each span is run through the text
encoder on its own row of a padded batch, so the bidirectional encoder never attends across span or
sample boundaries; `text_ids_mask` and `text_ids_len` carry that segmentation to the model.

`BreezeTTSProcessor.decode` decodes generated frames with the Qwen3-TTS 12 Hz audio tokenizer
bundled in the checkpoint's `audio_tokenizer` subfolder, which is also what encodes reference audio.
`model.generate(..., output_audio=True)` decodes them with `config.codec_config`'s Mimi model
instead, which the checkpoint carries as `codec_model`.

The bundled audio tokenizer is a raw Qwen3-TTS-Tokenizer-12Hz checkpoint whose decoder quantizer is
stored under `decoder.quantizer.rvq_first`/`rvq_rest`, which is not where
`Qwen3TTSTokenizerMultiCodebookModel` looks for it. Run `weight_conversion.convert` over a local copy
of the checkpoint once to rename those keys, and load the processor from the converted directory.


## Training

Training uses the standard `forward`: pass `labels` of shape `(batch_size, sequence_length)` alongside the
`input_ids` and `input_values` the processor builds, or let the processor build them with
`output_labels=True`. Indices are `config.audio_token_id` for a frame both heads are scored on,
`config.audio_eos_token_id` for the end-of-audio frame, `-100` for a position neither head is scored on, and
`-101` for a frame only the backbone is scored on; `depth_decoder_labels_ratio` controls how many frames get
`-101`. The audio frames scored are the ones the prompt carries, so a prompt with no audio in it is rejected.

`forward` expands those over the codebook dimension and returns
`backbone_loss + config.depth_header_loss_weight * depth_decoder_loss` while training, and
`backbone_loss + depth_decoder_loss` otherwise, with both terms also reported on their own in
`BreezeTTSOutputWithPast`. `backbone_loss` is the cross-entropy of the backbone head, which is one class wider
than a codebook vocabulary, over the first codebook of every frame plus the extra end-of-audio class.
`depth_decoder_loss` is the cross-entropy of the position-specific codebook head over codebooks
`1 .. num_codebooks - 1`, teacher-forced on the frames whose expanded labels are not uniformly `-100` past
codebook 0, with the backbone hidden state of the preceding position spliced in at depth position 0. Neither
term takes `num_items_in_batch`, since the two are scored over different token counts, which is why
`accepts_loss_kwargs` is `False`.

`codec_model` is frozen and kept in eval mode unconditionally, and `text_encoder` is frozen and kept in eval
mode unless `text_encoder_config.requires_grad` is set, which `BreezeBlue/breeze-tts-2` sets to `false`. A
frozen text encoder must also be deterministic, so a non-zero dropout rate on its config is rejected.


## Not carried over from upstream

Recorded per CLAUDE.md section 2.6. None of these is resolved here.

- **The CUDA graph fast path.** `models/fast_streaming.py`, `models/cudagraph/backbone_graph.py`,
  `models/cudagraph/backbone_prefill_graph.py`, `models/cudagraph/depth_decoder_graph.py`,
  `models/text_encoder_graph.py`, `models/warmup_profile.py` and `configs/fast.json` captured the text
  encoder, the backbone prefill and decode steps, the depth decoder loop and the codec decode into CUDA
  graphs over `StaticCache`, warmed from a shape manifest. They replay the same forwards this code runs, and
  the guidance formula they fold into the graph is the one in `_sample`, so no model behaviour is missing.
  What is missing is the speed those graphs buy: the upstream README's 40 ms time to first audio and 0.32
  real time factor are measured with `--fast-all`. `generate` runs eagerly, through
  `generation_config.compile_config` at most.
- **Chunked streaming audio output.** `models/stream_runtime/` decoded the Qwen3-TTS 12 Hz codec decoder in
  chunks, with convolution left-caches, a static shift KV cache and one execution lane per request, which is
  what let `infer.py` and the HTTP server emit audio while the backbone was still decoding. It has no
  counterpart here: `BreezeTTSProcessor.decode` and `generate(output_audio=True)` decode a whole utterance at
  once. It is also built entirely on the third-party `qwen_tts` package, which CLAUDE.md H11 forbids adding.
- **The `"llama3"` and native `"breeze"` backbone branches.** `models/breeze_backbone_factory.py` could build
  the backbone from `LlamaDecoderLayer`, or from the native `BreezeDecoderLayer` stack in `models/breeze.py`.
  `BreezeTTSBackboneModel` inherits `Qwen3Model`, which is what `backbone_model_type` names in the only
  released checkpoint. A checkpoint selecting either other branch would load into a Qwen3 layer stack.
- **The `"breeze_text_encoder_adapter"` text encoder projection.** Upstream `models/breeze.py` imports
  `BreezeTextEncoderAdapter` from `models/t5_adapter.py` for that projection type. That module is not in the
  vendored tree, so there is nothing to migrate and `_build_text_encoder_proj` raises for it.
- **The `drop_last_frame` audio segment option.** `breeze_infer/templates.py` could drop the last frame of a
  reference audio span. No template in that file sets it, and the processor has no equivalent argument.
- **Deployment scaffolding.** `infer.py`, `breeze_infer/api.py`, `docker/`, `.dockerignore` and `tests/` are
  dropped along with `fastapi`, `uvicorn`, `python-multipart`, `soundfile` and `pytest`. The prompt building
  in `breeze_infer/templates.py` moved into `BreezeTTSProcessor`, the loading in `breeze_infer/runtime.py`
  into `from_pretrained`, and the generation defaults into the checkpoint's own `generation_config.json`.
  `tests/test_logits_process.py` and `tests/test_templates.py` covered code that is migrated, but there is no
  test layout for a model folder here to move them into.


## Repository integration

Two things outside this folder are still needed and were deliberately not touched:

- `voicestudio/models/__init__.py` needs a `from .breeze_tts import *` line.
- `PROJECT.md` needs a Breeze TTS 2 status entry carrying the gaps listed above, and a
  `BreezeBlue/breeze-tts-2` row in its checkpoint table.

No new dependency is required. The migration removes `qwen-tts`, `numpy`, `soundfile`, `fastapi`, `uvicorn`,
`python-multipart`, `pytest` and `flash-attn` from what this model needs, leaving `torch`, `transformers`,
`safetensors` and `huggingface_hub`.
