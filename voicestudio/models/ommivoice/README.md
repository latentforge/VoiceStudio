# OmniVoice

OmniVoice synthesizes speech in over 600 languages from a single non-autoregressive stack. A Qwen3 backbone reads one sequence that interleaves text tokens with audio frames, where a frame is the sum of its eight residual codebook embeddings drawn from one fused table, and a single linear head projects every backbone position to all eight codebook distributions at once. Generation starts from a canvas of frames whose codebook entries all carry an explicit mask id and fills it by iterative unmasking: each step scores every still masked entry under classifier-free guidance, commits the highest scoring ones on a shifted timestep schedule, and feeds the partially filled canvas back in. Attention is bidirectional throughout and no key-value cache is ever used. Voice cloning prepends the reference audio codes and its transcript; voice design instead passes speaker attributes between `<|instruct_start|>` and `<|instruct_end|>`.

Original model and code: [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)


## Usage

```python
import soundfile as sf

from voicestudio.models.ommivoice import OmniVoiceForConditionalGeneration, OmniVoiceProcessor

model_id = "k2-fsa/OmniVoice"

processor = OmniVoiceProcessor.from_pretrained(model_id)
model = OmniVoiceForConditionalGeneration.from_pretrained(model_id).to("cuda").eval()

inputs = processor(
    text="The quick brown fox jumps over the lazy dog.",
    language="English",
    instruct="female, british accent",
).to(model.device)

audio_codes = model.generate(**inputs)
waveform = processor.batch_decode(audio_codes)[0]
sf.write("output.wav", waveform, processor.sampling_rate)
```

`language` takes an id such as `"en"` or a full name such as `"English"`; `OmniVoiceProcessor.supported_language_ids` and `supported_language_names` list what is accepted, and leaving it unset runs the language agnostic mode. `instruct` is validated against the closed attribute vocabulary and raises on an unknown or self contradictory item.

For voice cloning, pass the reference waveform. The processor removes its silences, encodes it with the bundled `HiggsAudioV2TokenizerModel`, and transcribes it when no `reference_text` is given:

```python
reference, sampling_rate = sf.read("reference.wav")

inputs = processor(
    text="Words to speak in the cloned voice.",
    reference_audio=reference,
    reference_text="Transcript of the reference clip.",
    sampling_rate=sampling_rate,
).to(model.device)
```

Leaving `reference_text` unset loads `openai/whisper-large-v3-turbo` on first use and transcribes the clip, which is what upstream's `create_voice_clone_prompt` does. Pass `asr_model_id` to the processor constructor to change that model.

`generate` takes the fields of `OmniVoiceGenerationConfig` as keyword overrides: `num_step`, `guidance_scale`, `t_shift`, `layer_penalty_factor`, `position_temperature` and `class_temperature`. The defaults are upstream's.


## Training

Pass the target audio codes to the processor and the standard `forward` returns a loss:

```python
batch = processor(
    text=["Words that were spoken."],
    language="en",
    audio_codes=[audio_codes],      # (8, num_frames), from processor.encode_audio
    prompt_ratio=(0.0, 0.3),
    mask_ratio=(0.0, 1.0),
    output_labels=True,
)
outputs = model(**batch)
outputs.loss.backward()
```

**What the upstream loss computes.** `OmniVoice.forward` in `omnivoice/models/omnivoice.py` is a single term. It reshapes the head output to `(batch, 8, sequence, 1025)`, takes an unreduced cross-entropy against `labels` with `ignore_index=-100`, and then reduces in two stages: each codebook is averaged over its own valid positions, `sum(dim=(0, 2)) / valid.sum(dim=(0, 2)).clamp(min=1.0)`, and the eight per codebook means are combined by `audio_codebook_weights` normalized to sum to one, `[8, 8, 6, 6, 4, 4, 2, 2]` in every released config. Averaging per codebook before weighting is load bearing: it stops the weighting from being skewed by how many entries each codebook happened to have masked. `OmniTrainer.train` in `omnivoice/training/trainer.py` backpropagates that scalar and nothing else, and `OmniTrainer.evaluate` reports its mean. There is no auxiliary term, no reconstruction term and no discriminator.

**Which positions carry a target.** `OmniVoiceSampleProcessor` in `omnivoice/data/processor.py` writes `-100` on every style token, every text token, every audio entry left unmasked, and, unless conditioning was dropped, every entry in the leading prompt region. So the loss covers exactly the entries the model was asked to predict. Per sample it draws `prompt_ratio` uniformly from `(0.0, 0.3)`, `mask_ratio` uniformly from `(0.0, 1.0)`, and with probability `drop_cond_ratio` 0.1 drops the style and text prefix entirely, which is what trains the unconditional branch that classifier-free guidance contrasts against at generation time. `OmniVoiceProcessor.__call__` reproduces that draw and that label layout through `prompt_ratio`, `mask_ratio` and `drop_conditioning`.

**What upstream freezes.** Nothing. There is no `requires_grad_(False)` and no `.eval()` on a submodule anywhere in `omnivoice/`, and `create_optimizer_and_scheduler` puts every parameter with `requires_grad` into one AdamW group. The backbone, the fused audio embedding table and the output head all train together at `learning_rate` 1e-4, `weight_decay` 0.01, `max_grad_norm` 1.0, cosine schedule with `warmup_ratio` 0.03, `steps` 300000, `batch_tokens` 8192, bf16. The audio tokenizer is not part of the model class at all: upstream extracts codes to WebDataset shards before training starts, and here it lives on the processor, so it can never enter the optimizer. The only partial freeze is the optional PEFT path, where `use_lora` puts rank 16 adapters on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj` and keeps `audio_embeddings` and `audio_heads` fully trainable through `modules_to_save`; that path is not migrated, see below.

Since nothing is frozen, the two failure modes that have bitten other migrations here do not arise: there is no frozen module that could keep running dropout for want of an `.eval()` call, and there is no freeze that could fail to survive `from_pretrained` rebuilding `Parameter` objects. The checkpoint's `llm_config` also sets `attention_dropout` to 0.0, so the backbone has no stochastic layer to leak either way. Measured after a real `from_pretrained`: 612577280 of 612577280 parameters trainable, and a backward that leaves no parameter of the backbone, the embedding table or the head without a gradient.

`forward(labels=...)` is checked against an independent reimplementation of the three steps above rather than against its own code, and agrees to 4.768e-07 with a shuffled-target control an order of magnitude higher. See the Verification section.

Packing several samples into one sequence is supported by the model: pass `document_ids` of shape `(batch, sequence)` and positions only attend within their own document. It needs `sdpa` or `flex_attention`, and `forward` raises under flash attention, which takes a padding mask rather than an arbitrary one.


## Lineage

`OmniVoiceAudioEmbeddings` inherits `HiggsAudioV2Embeddings` from transformers, which is the same computation upstream writes by hand: shift each codebook's ids by `codebook_index * audio_vocab_size` into one fused table and sum the eight lookups per frame. That claim is measured rather than argued, at a maximum absolute difference of 0.0 against upstream's own formula from the same weights. `OmniVoicePreTrainedModel` inherits `HiggsAudioV2PreTrainedModel` for the `_init_weights` rule that restores the offsets buffer under meta device init, and the offsets load from the checkpoint as `[0, 1025, 2050, 3075, 4100, 5125, 6150, 7175]`. The backbone is whatever `llm_config` names, `AutoModel.from_config`, a `Qwen3Model` in both released checkpoints.

The `Csm` lineage that breeze_tts and chroma use was checked and does not apply. `Csm` is a backbone plus a depth decoder that walks the codebooks autoregressively within a frame. OmniVoice has no depth decoder and no autoregression of any kind: `audio_heads` is one `nn.Linear(hidden_size, 8 * 1025, bias=False)` read once per position, `use_cache` is off, and frames are filled by iterative unmasking over the whole canvas. The bidirectionality is measured, not assumed: editing the last position of a prompt moves the logits at position 0, which cannot happen under causal masking, and none of the 28 modules exposing `is_causal` is left `True`. Inheriting `Csm` would mean inheriting the very machinery this model replaces. The rejection here is for a different reason than dia2's, which kept the depth decoder and rejected `Csm` only over per depth position weight selection.


## Verification

Two kinds of evidence, kept apart. The measurements below were run on CPU against the real `k2-fsa/OmniVoice` weights. Everything else is a line by line read of the pristine upstream source, which is weaker, and is labelled as such.

**Measured.**

- `from_pretrained` in float32 reports no missing, unexpected or mismatched keys and no errors across all 313 tensors and 612577280 parameters.
- Every one of the 313 checkpoint tensors lands on a target that exists in the loaded model, with the same shape and bit identical values, and no model tensor is left unfilled by the checkpoint. The `WeightRenaming` rules therefore consume every source weight, which is the failure the substitution trap would otherwise hide.
- The fused codebook embedding is bit identical to upstream's own `_prepare_embed_inputs` formula computed from the same weights, at a maximum absolute difference of 0.0. Inheriting `HiggsAudioV2Embeddings` is the same computation, not merely a similar one.
- Attention is bidirectional in fact and not only by intent: editing the final position of a prompt moves the logits at position 0 by 1.535522. A causal backbone gives exactly zero. 28 modules expose `is_causal` and none of them is still `True`.
- `forward(labels=...)` returns 2.16379952 where an independent reimplementation of upstream's three step reduction, written as an explicit per codebook loop over selected valid positions, returns 2.16380000 from the same logits. The difference is 4.768e-07, which is float32 accumulation order. The per codebook means were 0.2822, 0.6609, 1.4248, 2.3689, 4.746, 3.1463, 5.0524 and 7.2855 under weights 0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05 and 0.05, so the codebook ordering the weights assume is visible in the numbers. Scoring the same logits against the same target values shuffled among themselves gives 20.17284393, against a uniform guess reference `ln(1025)` of 6.932448, so the real value is measuring the targets.
- Nothing is frozen, as the upstream read predicted: 612577280 of 612577280 parameters are trainable, and a real backward puts a gradient on every parameter of all three module groups, at grad norms 13.848407 for `model.llm`, 2.291437 for `model.audio_embeddings` and 12.673671 for `audio_heads`, with zero parameters left without a gradient in any of them.
- The processor loads the checkpoint's bundled `HiggsAudioV2TokenizerModel` and `DacFeatureExtractor` at 24000 Hz and 25 frames per second, knows 646 language ids and 646 names, and round trips a 1.2s waveform to `(8, 30)` codes and back to 28800 finite samples.
- A full 32 step generation on CPU, from `"The quick brown fox jumps over the lazy dog, and the dog does not mind at all."` with `language="English"` and `instruct="female, british accent"`, took 94.2s and returned `(1, 8, 113)` codes in the range 1 to 1023, so no frame was left at the mask id. wav2vec2 `facebook/wav2vec2-base-960h` transcribes the decoded waveform as `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG AND THE DOG DOES NOT MIND AT ALL`, which is 17 words against 17 and a WER of 0.000.
- The generated length is the length the model asks for, not a truncation. With no reference pair the duration estimator calibrates against its own fallback, `"Nice to meet you."` at 25 frames, whose measured weight is 14.1, giving 0.564 weight units per frame. The prompt sentence weighs 64.2, so the estimate is `64.2 / 0.564 = 113.83` and `int()` of that is the 113 frames that came out, matching to the frame.
- Frames map to samples at exactly `hop_length` 960, and the written wav is shorter than `frames * 960` only because of upstream's own post-processing. `batch_decode` forwards to `decode` with its defaults, which are upstream's `postprocess_output` behaviour: `remove_silence` at `mid_sil=500, lead_sil=100, trail_sil=100`, peak normalization to 0.5 when no reference RMS is given, then 0.1s of padding on each side. Decoding the same codes with `postprocess=False, pad_duration=0.0, fade_duration=0.0` returns exactly `113 * 960 = 108480` samples, so no frame is lost between the codes and the waveform. Through the default path the stages are 108480 out of the codec, 18720 removed by silence trimming, 4800 added as padding, and 94560 written. Dividing that 94560 by the 113 frames gives 836.8 and looks like a broken ratio, but it divides a post-processed length by a pre-post-processing frame count.
- Net of the 0.2s of padding, the wav holds 3.74s of audio for 17 words, which is 4.55 words per second. Measured over the codec output instead, which still carries the leading and trailing silence, the same utterance is 3.76 words per second. The higher figure is the one that describes the speech: it is fast, at the top of the normal English range rather than the middle of it, and the WER of 0.000 over 17 hypothesis words against 17 reference words shows it is fast speech rather than lost content.
- A second generation from the same prompt, run separately to check the decode accounting, transcribes verbatim as well, so the WER of 0.000 holds over two independent samples rather than one.

**Read, not measured.** The unmasking loop is traced against upstream's `_generate_iterative` step for step: the same shifted timestep schedule, the same per step budget with the remainder forced into the last step, the same guidance form `log_softmax(c + s * (c - u))` over log probabilities rather than logits, the same `-inf` on the mask id, the same `layer_penalty_factor` subtraction by codebook index, the same Gumbel perturbation of the position scores, the same `-inf` on already committed entries, the same flat top-k commit, and the same batching of the conditional rows `[0, B)` against the canvas only rows `[B, 2B)`. The language table, the voice design tables, the abbreviation and punctuation sets and the duration estimator's weight and Unicode range tables are transcribed from upstream unchanged; the language dict occupies the same line span in both files.

The upstream sources traced were `omnivoice/models/omnivoice.py`, `omnivoice/data/processor.py`, `omnivoice/data/collator.py`, `omnivoice/training/trainer.py`, `omnivoice/training/builder.py`, `omnivoice/training/config.py`, `omnivoice/utils/audio.py`, `omnivoice/utils/text.py`, `omnivoice/utils/duration.py`, `omnivoice/utils/voice_design.py`, `omnivoice/utils/lang_map.py`, `omnivoice/utils/lora.py` and `examples/config/*.json`.

**One deliberate numerical deviation.** Upstream's silence handling round trips the waveform through a pydub `AudioSegment`, which quantizes it to 16 bit and back; the inlined replacement computes the same thresholds on the same 16 bit scale but slices the original float array, so the returned samples are not quantized. Everything that decides where to cut is bit compatible with pydub's integer `audioop.rms`, down to the `math.floor` on the square root, but the samples that survive the cut differ from upstream's in the low bits.

**One defect found by measurement and fixed.** `OmniVoiceConfig.audio_tokenizer_id` was inert: a sentinel written into the config never reached the loader while a sentinel passed as an argument did, and the released checkpoint's own `config.json` does not declare the field at all. `OmniVoiceProcessor.from_pretrained` now resolves the fallback from the model config when the caller passes no explicit id, following `voicestudio/models/higgs_tts3`, which established this field and reads it the same defensive way. All three branches are confirmed after the fix: with no argument the config value reaches the loader, an explicit argument still overrides it, and a path carrying no model config falls back to `eustlb/higgs-audio-v2-tokenizer`.


## Not carried over from upstream

Recorded per CLAUDE.md section 2.6. None of these is resolved here.

- **Long form chunked generation.** Upstream's `generate` measures the estimated duration against `audio_chunk_threshold` (30s) and routes anything longer to `_generate_chunked`, which splits the text at punctuation into `audio_chunk_duration` (15s) pieces, generates them chunk by chunk while batching across items at the same chunk index, and, when there is no reference audio, uses the first chunk's own output as the reference for every later chunk so the voice stays fixed. `OmniVoiceProcessor.chunk_text` exposes the splitting and `OmniVoiceProcessor.decode` cross fades a list of chunks, but the loop, the threshold and the first chunk as reference rule are not reproduced anywhere, so a caller has to write them.
- **The `<|denoise|>` training path.** `OmniVoiceSampleProcessor` emits `<|denoise|>` and pins the prompt boundary to `clean_start_token_idx` whenever a sample carries that field, which is how upstream trains the model to recover a clean prompt from a noised one; `omnivoice/scripts/extract_audio_tokens_add_noise.py` is what produces those samples. `OmniVoiceProcessor` emits `<|denoise|>` in generation mode but never in training mode, so the token can be used and cannot be trained.
- **LoRA fine tuning.** `use_lora` and `omnivoice/utils/lora.py` need `peft`, which this project does not depend on, so the adapter path, `merge_lora` and `tests/test_lora.py` are all dropped. Whether `peft` should be added is a decision for a human.
- **Opt-in text normalization.** `generate(normalize_text=True)` routed Chinese and English through WeTextProcessing and other languages through `num2words`, holding out bracketed tags and pinyin tone markers. Both are optional third-party dependencies, so the feature is absent rather than reimplemented, and a caller has to normalize the text itself.
- **The FlashInfer decoding path.** `omnivoice/models/omnivoice_flashinfer.py` packed the conditional and unconditional documents into one row behind flashinfer's ragged attention, replaced Qwen3's RMSNorm, attention and MLP with fused kernels, and optionally captured CUDA graphs per shape. It is a hardware specific accelerator that needs `flashinfer`, and none of it is migrated; the file's name is where `generation_omnivoice.py` comes from, but nothing of its content survives.
- **`VoiceClonePrompt.save` and `load`.** Upstream serializes reference codes, transcript and RMS to a `.pt` file for reuse across sessions. Here `processor.encode_audio` returns the codes and the caller keeps them.
- **The per sample conditioning draws.** `language_ratio`, `use_pinyin_ratio`, `instruct_ratio` and `only_instruct_ratio` decide, per training sample, whether the language marker, the pinyin spelling of the text, and the instruct marker are shown at all, and `only_instruct_ratio` additionally forces `prompt_ratio` to zero. `OmniVoiceProcessor.__call__` takes `language` and `instruct` as given, so a caller assembling a training batch has to make those draws. The `text_pinyin` field itself has no counterpart in the processor.
- **Sequence packing on the data side.** `PackingDataCollator` concatenates samples into a single `[1, C, batch_tokens]` row and emits the `document_ids` the model consumes. The model accepts `document_ids`, but the processor never produces them, so the packing collator has to be written by the caller. `PaddingDataCollator` is subsumed by `OmniVoiceProcessor._collate`.
- **Everything outside the model.** The Gradio demo and the four CLI entry points, the WebDataset readers and length grouped batching, the Accelerate training loop and its checkpoint handling, the offline denoising and token extraction scripts, and the WER, MOS and speaker similarity evaluation harness with its `funasr`, `s3prl`, `jiwer`, `zhconv`, `zhon` and `unidecode` dependencies are all dropped.


## Checkpoints

The CLAUDE.md section 2.3 search found two public checkpoints and no others.

- `k2-fsa/OmniVoice`, the released model this folder targets. Qwen3 backbone at `hidden_size` 1024, 28 layers, `vocab_size` 151676, 8 codebooks of 1025 entries, and a bundled `audio_tokenizer` subfolder holding a `HiggsAudioV2TokenizerModel` at 24 kHz with `hop_length` 960, so 25 frames per second.
- `k2-fsa/OmniVoice-Emilia`, 612.6M parameters, same architecture and same repository layout, trained on Emilia. `examples/config/train_config_emilia.json` is the config that produces it.

Where the search came up empty: the GitHub repository has six releases up to 0.2.1, none of which attaches a weights asset, and its README points back at the two Hugging Face repositories. The Hugging Face Space `k2-fsa/OmniVoice` holds only `app.py`, `requirements.txt`, `README.md` and an `omnivoice/` directory, with no weights file of any kind, so unlike PromptTTS++ there is nothing bundled there. Zenodo returns no record for this model. The paper is arXiv 2604.00688, whose resources section names the same two Hugging Face repositories.


## Repository integration

Three things outside this folder are still needed and were deliberately not touched:

- `voicestudio/models/__init__.py` needs a `from .ommivoice import *` line.
- `PROJECT.md` needs an OmniVoice status entry carrying the gaps listed above, and its "Sibling inheritance map" entry for OmniVoice needs correcting: the backbone plus depth decoder premise it records does not describe this model.
- Nothing in `pyproject.toml` or `uv.lock` changes. The migration removes `pydub`, `soundfile`, `librosa`, `gradio`, `tensorboardX`, `webdataset`, `accelerate`, `flashinfer`, `peft`, `WeTextProcessing`, `num2words`, `jiwer`, `s3prl`, `funasr`, `zhconv`, `zhon` and `unidecode` from what this model needs. What remains is `torch`, `transformers`, `numpy` and, only when a caller passes audio at a sampling rate other than the tokenizer's, `torchaudio`.

`uv.lock` and `pyproject.toml.bak` are upstream's own packaging metadata and are kept, following `voicestudio/models/dia2/`.
