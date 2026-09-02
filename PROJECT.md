# VoiceStudio Migration Project

This document records the plan to migrate VoiceStudio away from the namespace-package
architecture and into a single repository where every supported model is implemented
directly against the `transformers` model API. It exists so the plan survives context
resets across sessions and agents. Update it as work progresses; do not let it go stale.

## Workspace layout

The project checkout lives on the `C:` drive, which does not have room for model
checkpoints. `dep/` (vendored source clones) and `ckpts/` (downloaded checkpoints, used
only for local testing during migration) are relocated to `D:\VoiceWork`:

- `D:\VoiceWork\dep` — vendored source clones, linked at `dep/` in this repo.
- `D:\VoiceWork\ckpts` — downloaded checkpoints for local testing, linked at `ckpts/`.
- `D:\VoiceWork\worktree` — scratch location for any git worktrees created while
  working on a migration in isolation.

`ckpts/` is a temporary scratch folder, not an artifact of the migration. Everything
under it gets deleted once the full migration is complete; do not treat anything placed
there as something to preserve or commit.

Both `dep/` and `ckpts/` are gitignored.

### Downloading models for local testing

- Always download through `hf_xet` (the accelerated Hugging Face download path), never
  the default `huggingface_hub` cache resolution.
- Downloads must land under `ckpts/` (which resolves to `D:\VoiceWork\ckpts`), never the
  default `~/.cache/huggingface` location. Pass an explicit local target, do not rely on
  `HF_HOME`/cache defaults pointing there implicitly without checking.

### GPU verification

The Google Colab CLI (`pip install google-colab-cli`) provisions a remote Colab
runtime and runs scripts on it: `colab new --gpu T4 --auth oauth2` once to
authenticate, then `colab exec -f script.py` or `colab run --gpu T4 script.py` per
run. Use it when a migrated model needs to run on an actual GPU to verify parity with
the original implementation.

## Migration order

1. Qwen3-TTS (depends on the `transformers-tts` merge, see below)
2. Parler-TTS
3. Everything else, in any order after that

## Folders explicitly out of scope

`voicestudio/models/stable_ommivoice/`, `voicestudio/models/stable_parler_tts/`, and
`voicestudio/models/stable_qwen3_tts/` are not part of this migration. Leave them
untouched.

## Background

VoiceStudio originally split each model's code into a separate namespace package
(`voicestudio-parler-tts`, `voicestudio-qwen3-tts`, etc.) to keep each original author's
code attributed and isolated. In practice this made PyPI packaging painful and spread
the codebase across too many repositories to maintain. We are reversing that decision:
model code moves back into this repository, with git history preserved, and is rewritten
to match `transformers` conventions instead of being wrapped.

## Per-model migration procedure

For every model in scope, follow these steps in order:

1. **Vendor the source.** Clone the model's official code repo (or the code shipped in
   its Hugging Face model repo, if that's the only source) into `dep/`.
2. **Merge history into the model's folder.**
   - If an existing namespace implementation already exists under `voicestudio/models/`,
     keep that as the base.
   - Otherwise, merge the cloned repo's git history directly into the target model code
     folder (e.g. via `git subtree`/`git filter-repo` + merge, not a fresh copy) so
     authorship and history are preserved.
3. **Rebase onto the closest transformers model.** Analyze the model architecture, find
   the most structurally similar model already in `transformers`, and inherit from it
   instead of writing the model from scratch. Do not generate a full model implementation
   when a close relative already exists in the library.
4. **Add a README.md** in the model's folder linking back to the original code repository.
5. **Add license headers.** Every source file carries the original repo's license notice,
   formatted the way `transformers` formats its license headers.
6. **Delete the vendored copy** from `dep/` once the model's migration is complete and
   verified.

## Repos to fully migrate then delete

- https://github.com/latentforge/higgs-audio
- https://github.com/latentforge/parler-tts
- https://github.com/latentforge/Qwen3-TTS
- https://github.com/latentforge/CosyVoice
- https://github.com/latentforge/Chroma
- https://github.com/latentforge/Spark-TTS
- https://github.com/latentforge/dia
- https://github.com/latentforge/F5-TTS
- https://github.com/latentforge/promptttspp

Each of these gets deleted from GitHub only after its migration is complete, verified,
and the vendored copy is removed from `dep/`.

## Hugging Face checkpoint sources

This is the exact model list from the `RetentionLabs/latentforge-voicestudio` HF
collection, recorded here so it survives context resets (a prior loss of this list
during a session summary caused PromptTTS++'s entry below to be misdiagnosed as having
no public checkpoint at all).

| Hub repo | Type | Notes |
|---|---|---|
| parler-tts/parler-tts-mini-v1 | Text-to-Speech | |
| parler-tts/parler-tts-mini-v1.1 | Text-to-Speech | |
| parler-tts/parler-tts-large-v1 | Text-to-Speech | |
| Promptttspp | **Agents space** | This is a Gradio Space (demo UI), not a standard downloadable model repo. See `https://huggingface.co/spaces/line-corporation/promptttspp`. Real upstream code/weights: `https://github.com/line/promptttspp`, whose README also links Hugging Face pretrained models/demo. No plain `from_pretrained`-loadable checkpoint repo has been located yet; if one exists it is not this Space entry directly, and finding/verifying it is still open work. |
| Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign | Text-to-Speech | |
| k2-fsa/OmniVoice | Text-to-Speech | |
| bosonai/higgs-tts-3-4b | Text-to-Speech | |
| bosonai/higgs-tts-2-3b-base | Text-to-Speech | |
| FlashLabs/Chroma-4B | Any-to-Any | Gated, requires an HF token with access. |
| Qwen/Qwen3-TTS-12Hz-1.7B-Base | Text-to-Speech | |
| nari-labs/Dia-1.6B-0626 | Text-to-Speech | |
| FunAudioLLM/CosyVoice-300M | Text-to-Speech | CosyVoice v1. |
| FunAudioLLM/CosyVoice2-0.5B | Text-to-Speech | CosyVoice v2. |
| FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | Text-to-Speech | CosyVoice v3. |
| SWivid/F5-TTS | Text-to-Speech | |
| SparkAudio/Spark-TTS-0.5B | Text-to-Speech | |

## Dependencies to remove

- **https://github.com/latentforge/audiotools** — analyze what depends on it, remove the
  dependency, then delete the repo.
- **https://github.com/latentforge/vocos** — dependency was unused anywhere in this repo
  (`F5TTSProcessor.decode` already took a generic vocoder callable), dropped. Repo not
  yet deleted.
- **https://github.com/latentforge/speechbrain** — this fork exists only to support a
  newer torch version. Upstream speechbrain has caught up (v1.1.0 supports
  `torch>=2.1.0` with no upper bound), and the dependency turned out to be unused
  anywhere in this repo, so it was dropped entirely rather than switched to upstream.
  Delete the fork.
- **https://github.com/sarulab-speech/UTMOSv2** — decouple via the `evaluate` library
  using https://huggingface.co/spaces/sarulab-speech/UTMOSv2/ as the reference
  implementation. A new `voicestudio/metrics/` folder may be created if needed for this.

## Rules for the rewritten model code

- Every model must be trainable with cross-entropy loss, matching the standard
  `transformers` convention: `forward()` accepts `labels` and returns a `loss` on the
  output. Inference-only forward passes are not acceptable.
- Follow `transformers` model file conventions strictly: `modeling_<model>.py`,
  `configuration_<model>.py`, standard class inheritance, etc. Do not create files that
  fall outside the `transformers` model layout.
- Where a model already exists in `transformers` itself, only do an import relay — do
  not reimplement it.
- Use `WeightConvert` for any checkpoint conversion work.
- Comments follow `transformers` style: terse and technical, never narrated like a diary
  entry.
- Target `transformers` 5.0 conventions.
- Before writing a model from scratch, find the closest existing model lineage in
  `transformers` and inherit from it.
- Follow the `transformers` "Copied from" / `modular_<model>.py` conventions used
  upstream (see `transformers-tts/.ai/AGENTS.md`) wherever they apply to reduce
  duplication between model files.

## Processor standard

New models must use the tokenizer + audio_tokenizer processor pattern, i.e. all
preprocessing goes through the model's `Processor` — no separate manual preprocessing
step. For audio tokenizer models, follow the Qwen3-TTS and Higgs TTS 2 examples. For
Parler, switch to the `dac` implementation already registered in `transformers` instead
of vendoring DAC.

Target usage shape:

```python
model_id = "eustlb/higgs-audio-v2-generation-3B-base"
model = HiggsTTS2ForConditionalGeneration.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id).to(device)

outputs = model.generate(**inputs)

audio_values, sr = processor.decode(outputs)
```

### Qwen3-TTS specifics

Drop the existing VoiceStudio Qwen3-TTS implementation and depend on the version being
merged into `transformers-tts` instead. Preserve the VoiceStudio `Qwen3TTSProcessor`
task-dispatch behavior if the incoming `transformers-tts` processor implementation is
missing it:

- `processor.encode`: accepts all parameters; raises a runtime error if the task implied
  by the given arguments doesn't match the model's configured task.
  and only allows the arguments valid for that task).
- `processor.encode_<task>` (e.g. `encode_voice_design`): only accepts the arguments
  valid for that specific task.

If the incoming processor is missing this behavior, subclass/extend it inside
VoiceStudio to restore it — don't fork the whole processor.

## Packaging changes

- Remove all namespace-package wiring described in the old README/pyproject (the
  commented-out `voicestudio-*` package entries, the split-repo installation model).
- Flash attention: depend on the `kernels` package (the current standard) instead of
  building/vendoring `flash-attn` wheels.
- Pin the `transformers` dependency to the `transformers-tts` fork
  (https://github.com/latentforge/transformers-tts), not upstream `transformers` or the
  ShahVandit fork.

## Status tracking

Update this table as each model's migration lands. "Code landed" means the folder exists,
imports, and passed a static review pass (architecture, CE-training requirement, CLAUDE.md
conventions) with known issues fixed. It does NOT mean the model has been run against a real
pretrained checkpoint yet; see "Runtime-verified" for that.

| Model | Code landed | Runtime-verified (real checkpoint) | Upstream git history preserved | Notes |
|---|---|---|---|---|
| Qwen3-TTS | Yes | Yes | Yes | Import relays to transformers-tts. Processor subclass adds `encode`/`encode_voice_design`/`encode_custom_voice` task dispatch with `RuntimeError` on task mismatch. `encode_voice_clone` raises `NotImplementedError`. Verified end to end on GPU against `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`: the published checkpoint is upstream's raw format, run `transformers.models.qwen3_tts.convert_qwen3_tts_to_hf.convert_checkpoint` first. Fixed a missing `speech_tokenizer/` subfolder load, a custom `__init__` that broke `ProcessorMixin.from_pretrained`'s constructor introspection, and (in transformers-tts itself) `Qwen3TTSConfig.get_text_config()` not delegating to `talker_config`. |
| Parler-TTS | Yes | No | Yes | transformers 5.0 API-drift pass against the installed `transformers-tts` fork (5.16.0.dev0). Import-time crash fixed: `transformers.pytorch_utils.isin_mps_friendly` no longer exists and the fork now calls `torch.isin` directly everywhere, so `ParlerTTSLogitsProcessor` does too. The vendored `dac_wrapper/` (which imported the standalone PyPI `dac` package and registered a `"dac_on_the_hub"` model_type) was confirmed unreferenced anywhere in the repo and deleted; the audio codec now resolves purely through `transformers.models.dac` (`DacConfig.model_type == "dac"` is in the fork's `CONFIG_MAPPING_NAMES`, and `("dac", "DacModel")` is in `MODEL_MAPPING_NAMES`, so the generic `AutoConfig.for_model`/`AutoModel.from_config` wiring in `configuration_parler_tts.py`/`modeling_parler_tts.py` reaches the real classes). `DacModel`'s real `encode(input_values, n_quantizers, return_dict)` / `decode(quantized_representation, audio_codes, return_dict)` were traced against the fork's source and match the existing call sites: `audio_codes` is 3-dim so the `audio_codes.ndim == 3` branch is taken, `use_audio_scales` correctly resolves to `False` (no `audio_scales` parameter on `decode`), and `use_4dim_audio_codes` stays `False` since `model_type != "encodec"`. Four further drift fixes: `_supports_flash_attn_2` renamed to `_supports_flash_attn` (v5's `_check_and_adjust_attn_implementation` only reads the new name); `ParlerTTSDecoderConfig` now declares `cross_attention_hidden_size` itself, since v5 removed it from the `PreTrainedConfig` base and `ParlerTTSForConditionalGeneration.__init__` dereferences it unconditionally (musicgen's own decoder config took the same fix upstream); the `tie_weights` override was dropped because `config.tie_encoder_decoder` and `PreTrainedModel._tie_encoder_decoder_weights` are both gone from v5 (musicgen dropped the same override); and all `Cache.from_legacy_cache`/`to_legacy_cache` uses were removed, since v5 deleted the legacy tuple-cache format entirely - `ParlerTTSDecoder.forward` now seeds `EncoderDecoderCache(DynamicCache(), DynamicCache())` directly, which is what musicgen does. That last one was on the unconditional forward path (`use_cache` defaults to `True`), so it broke every forward call. Those edits were made without runtime execution and have since been confirmed to import, construct and run: a throwaway small-dims script built a `ParlerTTSConfig`/`ParlerTTSForConditionalGeneration` over a real `DacConfig`+`T5Config`, asserted the audio codec resolves to the real `DacConfig`/`DacModel` (with `use_audio_scales` and `use_4dim_audio_codes` both `False` as traced), and ran a dummy CPU forward+backward with `labels`: loss 3.52, logits `(4, 10, 33)` (batch 2 x 2 codebooks, 7 label steps + delay pattern, `codebook_size + 1` vocab), and real gradients on 58 of 163 parameters. The 105 without gradients are the unused `audio_encoder.*` (DAC only runs when raw `input_values` are passed, not when `labels` are already codes) plus `decoder.model.decoder.embed_positions.weights`, which is a frozen sinusoidal table (`requires_grad = False` at construction, same as musicgen); every trainable text-encoder, projection and decoder parameter received a gradient, so the H5 trainability bar holds. Per H4 this is a confirmation step only, not evidence of architectural correctness; real-checkpoint verification is still outstanding, hence "Runtime-verified: No". Open items found and deliberately not fixed: (1) `ParlerTTSForConditionalGeneration.generate` never applies v5's global generation defaults, so a plain `generate()` call dies with `TypeError: repeat_interleave() received an invalid combination of arguments - got (NoneType, dim=int)` inside `_expand_inputs_for_generation`. Root cause: in v5 every `GenerationConfig` field defaults to `None` and the concrete defaults live in `GenerationConfig._get_default_generation_params()`, which is applied only by `GenerationMixin._prepare_generation_config`; parler's `generate` still hand-rolls the v4 preamble (`deepcopy` + `generation_config.update(**kwargs)`) and so passes `num_return_sequences=None` straight through. Passing `num_return_sequences=1` explicitly makes the whole generation path run clean end to end (greedy, `max_new_tokens=4`, output shape `(2, 12)`), so this single unapplied default is the only thing breaking it. Upstream `MusicgenForConditionalGeneration.generate` has already migrated to `_prepare_generation_config` (alongside the new `_extract_generation_mode_kwargs`/`_validate_generation_mode` calls), while `MusicgenForCausalLM.generate` still carries the identical latent bug, and parler has the same two-method split. This is left open rather than patched because swapping the preamble is more than an API rename: `_prepare_generation_config` additionally *raises* on legacy generation params set on `model.config`, unsets a `"hybrid"` `cache_implementation`, and moves `output_attentions`/`output_hidden_states` into `model_kwargs`, and the surrounding `_extract_generation_mode_kwargs`/`_validate_generation_mode` scaffolding would need a real generation run to validate. (2) `ParlerTTSForConditionalGeneration._get_cache` still reads `cache_to_check.max_batch_size`, which no longer exists on any v5 `Cache`, so `generate(cache_implementation="static")` will `AttributeError`; v5 also removed `_get_cache` from `GenerationMixin` entirely and handles static caches inside `_prepare_cache_for_generation`, so the right fix is probably to delete the override rather than patch the attribute, which needs a real generation run to validate. The confirmation run above used the default dynamic cache and never reached `_get_cache`, so this remains unexercised. (3) `ParlerTTSConfig` declares no `sub_configs`, so `attn_implementation=` passed to `from_pretrained` will not propagate to the text/audio/decoder sub-configs (v5 propagates only through `sub_configs`); serialization is unaffected because `to_dict` recurses over nested configs independently. (4) `self.config.decoder.audio_channels` is dereferenced in `forward` when deriving `decoder_input_ids` from raw `input_values`, but `ParlerTTSDecoderConfig` never defines it and v5's base config does not either - a pre-existing musicgen-stereo leftover, latent in v4 too, reached only on that one branch. (5) `weight_conversion.py` defines its own local `ConversionOps`/`WeightRenaming`/`WeightConverter`/`convert_and_load_state_dict_in_model` that shadow the now-real ones in `transformers.core_model_loading` with different semantics, and its comments/docstrings are in Korean; it is self-contained and works, but it does not go through `WeightConvert` the way the rest of the repo does. Still a ~3300-line near-verbatim port with scattered `# Copied from` comments (several now pointing at methods that no longer exist upstream, e.g. `LlamaModel._update_causal_mask`) rather than a `modular_parler_tts.py`; revisit for a proper modular conversion. |
| Higgs TTS 2 | Yes | Yes | Yes | Import relay to transformers-tts. Verified on GPU against `eustlb/higgs-audio-v2-generation-3B-base`: real weights load with no LOAD REPORT (0 missing/unexpected). A prior static-read pass flagged `HiggsAudioV2ForConditionalGeneration.forward` hardcoding `8` instead of `self.config.num_codebooks` when building `audio_labels_expanded`; confirmed it happened to match this checkpoint's `num_codebooks=8` so it wasn't observably broken, but fixed anyway (transformers-tts `research`, commit `48abe225c3`) since it was still a latent correctness bug. Forward+backward with real `audio_labels` produced logits shape `(1, 299, 8208)` (8 codebooks x 1026 codebook_size), loss 36.67, and a real nonzero gradient norm (10.1) through backprop. `generate()` produced real interleaved audio-codebook tokens (shape `(1, 78, 8)`, correct delay-pattern BOS structure) which decoded to a waveform (66240 samples, RMS 0.091, max abs 0.69) - plausible speech-level amplitude, not degenerate. |
| Higgs TTS 3 | Yes | Yes | No | `bosonai/higgs-tts-3-4b`, weights-only checkpoint (no upstream v3 code, so nothing to graft). Reuses transformers-tts's `Qwen3Model`/`HiggsAudioV2*` classes; `HiggsTTS3Model`/`ForConditionalGeneration` are new. `audio_labels`-only crash and default-off text CE loss fixed. Verified on GPU: the checkpoint's `body.`/`tied.` weight namespace didn't match this model's parameter names, so every real weight loaded as MISSING/UNEXPECTED and the model ran on random init; fixed via a registered checkpoint conversion mapping. The checkpoint also ships no `preprocessor_config.json`/audio tokenizer, which crashed `HiggsTTS3Processor.from_pretrained`; it now degrades to a tokenizer-only processor for text-only use. With real weights loaded, forward logits and a CE loss (labels + audio_labels) with real backprop gradients were confirmed. The real hub repo's `config.json` reports `model_type="higgs_multimodal_qwen3"` (post-rename from `higgs-audio-v3-tts-4b`), not this config's own `higgs_tts3`; `AutoConfig.from_pretrained("bosonai/higgs-tts-3-4b")` previously failed outright since that model_type was unregistered anywhere. Fixed by registering `higgs_multimodal_qwen3` as an alias for `HiggsTTS3Config` directly on `CONFIG_MAPPING` in `__init__.py` (bypasses only `AutoConfig.register()`'s own model_type-consistency assertion, not a real library invariant); confirmed `AutoConfig`/`AutoModel`/`AutoModelForTextToWaveform` now resolve the real repo id to `HiggsTTS3Config`/`HiggsTTS3ForConditionalGeneration`. This is config/class routing only, not a re-verification of weight loading; the checkpoint's `body.`/`tied.` key-namespace conversion above still applies. |
| Chroma | Yes | Yes | Yes | Backbone/decoder reimplemented against transformers-tts's Llama, Qwen2.5-Omni thinker, and Mimi codec classes. Processor kwargs-merging and a labels-dereference guard fixed. Verified on GPU against the real gated `FlashLabs/Chroma-4B` checkpoint: `ChromaForConditionalGeneration.from_pretrained` loads all 1906 real weight keys with 0 missing/0 unexpected/0 mismatched (no weight_conversion.py needed, the checkpoint's own key layout already matches). Getting a real forward+backward pass and `generate()` working surfaced several transformers-tts API-drift/logic bugs invisible to a static read, all fixed (transformers-tts fork itself needed no changes, only this model's code): `ChromaProcessorKwargs` wrongly declared `prompt_text`/`prompt_audio` as `_merge_kwargs` categories (`KeyError: 'prompt_text'`); the processor assumed a `common_kwargs` entry survives in `_merge_kwargs`'s `output_kwargs`, but the real implementation only broadcasts it into the other modality dicts (`KeyError: 'common_kwargs'`); `_prepare_generation_config`/`prepare_inputs_for_generation` called the now-reordered/renamed base `GenerationMixin` methods positionally (`TypeError`s); `_get_initial_cache_position` was removed upstream (replaced with an inline seed of `model_kwargs["cache_position"]`); `ChromaConfig` had no `get_text_config` override, so `PretrainedConfig`'s default name-sniffing returned the top-level config instead of `backbone_config`, crashing `DynamicCache` construction (`AttributeError: 'ChromaConfig' object has no attribute 'num_hidden_layers'`); and `ChromaDecoderForCausalLM.forward`'s `past_codebook_num = past_key_values.get_seq_length() - 1` went negative on the very first decode step now that `generate()` eagerly creates an empty `Cache` before any forward call, producing a large negative embedding-table offset and a CUDA device-side assert. With all of those fixed: a real forward+backward pass (real reference audio re-encoded through the real Mimi codec into real audio codes, not dummy tensors) produced finite, non-degenerate backbone/decoder CE losses (3.47 / 5.13, total 8.59) with a real nonzero gradient norm (18.7) reaching the backbone, decoder, and thinker text-embedding parameters; `generate()` (following the checkpoint's own `example.ipynb` usage pattern: system prompt + question audio through the thinker, reference audio + transcript for voice cloning) produced real in-range interleaved audio-codebook tokens (shape `(1, 50, 8)`, values in `[0, 2047]`) which decoded through the real Mimi codec to a finite, plausible-amplitude waveform (`(1, 96000)` samples, RMS 0.095, max abs 1.05, no NaN/Inf). |
| Spark-TTS | Yes | No | Yes | Config previously had no LLM sub-config, so plain construction silently produced an untrained model; fixed. License header consolidated onto `modeling_spark_tts.py` only; dead code and processor/model duplication cleaned up. |
| Dia | Yes | Yes | Yes | Import relay to transformers-tts's native Dia. Loaded `nari-labs/Dia-1.6B-0626` on GPU and ran `generate()` successfully; no code changes needed. |
| CosyVoice v1 | Yes | No | Yes | LLM (`llm.pt`) verified against FunAudioLLM/CosyVoice-300M: text/LLM encoder was `Wav2Vec2ConformerEncoder` (always has conv module + macaron FFN + no input projection), rewritten to match the real WeNet-style single-FFN/no-conv encoder plus the input `Linear+LayerNorm` it was missing; real weights now load with 0 missing/0 unexpected/0 shape-mismatch keys and a CE forward/backward on real weights produces a finite loss and gradients on all 401 params (see weight_conversion.py `build_llm_weight_conversion_mapping`). HiFTGenerator's `ResBlock` was missing the checkpoint's learnable Snake activation (used fixed leaky_relu) and `F0Predictor` was missing `weight_norm`; both fixed (`build_hift_weight_conversion_mapping`) and now checkpoint-verified end-to-end: also found `CosyVoiceV1HiftConfig.f0_predictor_num_layers` defaulted to 3 but the real `hift.pt`'s `f0_predictor.condnet` has 5 conv layers (`condnet.{0,2,4,6,8}`), fixed the default to 5. With that fix, `hift.pt` now loads into `CosyVoiceV1HiFTGenerator` with 0 missing/0 unexpected keys, and a forward pass on a real-shaped mel (`(2, 80, 50)`) produces a finite, non-silent waveform (RMS 0.486, correctly clamped to the configured `audio_limit=0.99`, no NaNs). A later sibling task's fix to `CosyVoiceV3HiFTGenerator`'s identical `stft_window` meta-init bug (see below) flagged the same pattern as still present, unfixed, in this class; fixed here too (`register_buffer("stft_window", ..., persistent=False)` replaced with a `_hann_window` recomputed fresh on every `_stft`/`_istft` call). `CosyVoiceV2HiFTGenerator` reuses this class directly with no independent `stft_window` definition, so it's covered by the same fix. Verified on GPU against the real `FunAudioLLM/CosyVoice-300M` `hift.pt`: with the pre-fix code, loading the real checkpoint through the actual `from_pretrained` meta-init codepath and running a forward pass reproduced the exact failure mode described in the bug report (`torch.istft` NOLA/window-overlap-add check fails: `window overlap add min: 1`); with the fix, that same codepath's `_stft`/`_istft` round-trip on a real synthetic audio signal (220Hz sine + noise) now reconstructs it with 0.000000 mean absolute error and a non-degenerate RMS (0.357), confirming the recomputed window is correct. Two unrelated pre-existing bugs surfaced during that verification pass and were left out of scope at the time, but have since been fixed and re-verified (commit `068325c2`): (1) `CosyVoiceV1HiFTGenerator.__init__`'s `torch.cumprod(torch.tensor(downsample_rates), dim=0).tolist()` raised under `from_pretrained`'s real meta-device init context regardless of checkpoint content (`NotImplementedError: Cannot copy out of meta tensor; no data!`), so a bare `from_pretrained` call on this class could not succeed without a workaround; replaced with `itertools.accumulate` (the same fix already applied to `CosyVoiceV3HiFTGenerator`, commit `753e431b`). (2) `build_hift_weight_conversion_mapping`'s `resblocks`/`source_resblocks` `weight_g`/`weight_v` renaming rules used two regex capture groups (block idx, conv idx), but `WeightRenaming` only ever substitutes a single `\1` backreference into the target pattern, so both rules silently left a literal `\2` in every renamed key and those conv weights never matched the real module, loading as MISSING/UNEXPECTED and getting randomly reinitialized instead (the same bug class independently fixed in `_rel_position_encoder_renaming`, commit `de541f3d`); fixed by capturing the whole `<block idx>.<conv>.<conv idx>` path as a single group (it is identical on both sides; only the trailing `weight_g`/`weight_v`/`bias` leaf changes) instead of two. Note this means the "0 missing/0 unexpected keys" claim earlier in this cell (from the `f0_predictor_num_layers` fix, commit `39099cfd`) was measured without exercising `from_pretrained`'s real meta-init codepath, which is why it didn't catch either bug; both fixes are now verified specifically through that real codepath. With both fixed, re-verified on a real T4 Colab GPU against `FunAudioLLM/CosyVoice-300M`'s `hift.pt`, calling `CosyVoiceV1HiFTGenerator.from_pretrained` with no workarounds (default meta-init/`low_cpu_mem_usage` behavior): LOAD REPORT shows 0 missing / 0 unexpected / 0 mismatched keys across all 227 checkpoint tensors, and a full forward pass through the complete vocoder (not just an isolated STFT round-trip) on a real-shaped mel (`(2, 80, 50)`) produces a finite, non-degenerate waveform (`(2, 12800)`, no NaN/Inf, min/max correctly clamped to `audio_limit=-0.99/0.99`, mean -0.0002, std 0.516). Also wired the ONNX speech tokenizer (`speech_tokenizer_v1.onnx`) and speaker embedding model (`campplus.onnx`) into `CosyVoiceV1Processor` (`processing_cosyvoice_v1.py`) via `onnxruntime.InferenceSession`, loaded lazily in `from_pretrained` and exposed as `extract_speech_token`/`extract_speaker_embedding`; introspecting the real ONNX graphs confirmed `speech_tokenizer_v1.onnx` expects `(1, 128, T)` whisper-style 16kHz log-mel features (reimplemented `openai-whisper`'s exact `log_mel_spectrogram` using `librosa`'s mel filterbank, since neither VoiceStudio nor transformers-tts depends on the `whisper` package) and `campplus.onnx` expects `(batch, T, 80)` mean-normalized Kaldi fbank features (via `torchaudio.compliance.kaldi.fbank`), matching the real CosyVoice repo's `frontend.py` preprocessing; the ONNX models themselves are run directly, not reimplemented as torch modules. Verified against the real checkpoint's `speech_tokenizer_v1.onnx`/`campplus.onnx` on a real 5-second 16kHz audio sample: speech tokenization produced 250 tokens (25Hz rate, in-range int ids) and speaker embedding extraction produced a finite, non-degenerate `(1, 192)` vector (norm 93.6, no NaNs). Note `FunAudioLLM/CosyVoice-300M`'s hub repo ships no HF-compatible text tokenizer files, so `CosyVoiceV1Processor.from_pretrained`'s `tokenizer` attribute can't load against it end-to-end; that gap is pre-existing/orthogonal to the ONNX wiring and out of scope here. Flow decoder (`flow.pt`) source-traced against `FunAudioLLM/CosyVoice`'s `cosyvoice/flow/decoder.py` (`ConditionalDecoder`) and, since it imports `matcha.models.components.transformer.BasicTransformerBlock`, `Matcha-TTS`'s `matcha/models/components/transformer.py`: the real estimator block is a diffusers-style `BasicTransformerBlock` instantiated with `act_fn="gelu"` and no `num_embeds_ada_norm`/`norm_type` override, so `norm1`/`norm3` are plain `LayerNorm` (not AdaLN as a prior pass had guessed - the flow-matching timestep is injected only into the surrounding `ResnetBlock1D` via FiLM, never into the attention block), `attn1` is a `diffusers.Attention` with separate unbiased `to_q`/`to_k`/`to_v` and a biased `to_out.0` (no `norm2`/`attn2`, since `cross_attention_dim`/`double_self_attention` are unset), and the feed-forward is plain (non-gated) `GELU` (`ff.net.0.proj`+`ff.net.2`), not `GEGLU`. Rewrote `CosyVoiceV1EstimatorBlock`/`CosyVoiceV1EstimatorAttention`/`CosyVoiceV1EstimatorFeedForward` to match exactly, plus three other bugs the trace surfaced: `CosyVoiceV1ConditionalDecoder` never concatenated `cond` into its input (checkpoint's `in_channels=320`=4x mel_dim, code built only 3x), `CosyVoiceV1SinusoidalPosEmb` was missing the real `scale=1000` factor, and `CosyVoiceV1ResnetBlock1D.res_conv`/`CosyVoiceV1Block1D`'s `GroupNorm` used a `min(8, dim_out)`/identity-when-equal shortcut the real `ResnetBlock1D`/`Block1D` don't take (always a `Conv1d`/fixed `groups=8`). Added down/upsample module wrappers (`CosyVoiceV1Downsample1D`/`CosyVoiceV1Upsample1D`) matching `Downsample1D`/`Upsample1D`'s `.conv` nesting, and extended `weight_conversion.py`'s `build_flow_weight_conversion_mapping`/`_estimator_renaming` for the estimator's down/upsample keys; also fixed a latent bug in `_rel_position_encoder_renaming` where `WeightRenaming` only ever substitutes one `\1` backreference, so every two-group `(idx)`/`(weight|bias)` rule was silently leaving a literal `\2` in the renamed key (rewritten to match only the shared parent path and let the leaf suffix pass through unmatched). Verified against the real `FunAudioLLM/CosyVoice-300M` `flow.pt`: `from_pretrained` now loads all 1185 keys with 0 missing/0 unexpected, and a real forward+backward pass (batch of 2, real weights) produces a finite non-degenerate CFM loss (4.12) with finite gradients on all 1185 params, plus a correct-shaped, finite sampled mel from the Euler-solver inference path. ONNX speech-tokenizer/campplus wiring is done for v1's processor (see above); v2/v3 don't yet reuse it. |
| CosyVoice v2 | Yes | No | Yes | Subclasses v1; `Qwen2Model` LLM backbone. Source-traced against `FunAudioLLM/CosyVoice`'s `cosyvoice/llm/llm.py` (`Qwen2LM`/`Qwen2Encoder`) and `cosyvoice/flow/flow.py`+`cosyvoice/transformer/upsample_encoder.py` (`CausalMaskedDiffWithXvec`/`UpsampleConformerEncoder`). The LLM's own math already matched (no separate text encoder, no speaker embedding in the LM input, same sos/task/eos/fill token layout); the mismatch was the checkpoint's weight namespace, since the real `Qwen2Encoder` wraps a full `Qwen2ForCausalLM` (backbone at `llm.model.model.*`, plus an untied, unused-here `llm.model.lm_head.*`) while this model holds the bare `Qwen2Model` directly at `llm.*`; added `build_llm_weight_conversion_mapping` and `_keys_to_ignore_on_load_unexpected` for the dropped `lm_head`. The flow encoder was a real architecture mismatch, not just a naming one: it reused `CosyVoiceV1RelPositionEncoder` (bidirectional Conformer) plus a separate pre-lookahead conv and an interpolate-to-target-length regulator, none of which is what `UpsampleConformerEncoder` actually is (pre-lookahead conv folded inside the encoder, a 6-layer token-rate Conformer stack, a fixed 2x nearest-neighbor upsample via a dedicated `Upsample1D`, then an independently re-embedded 4-layer mel-rate Conformer stack, with the final `LayerNorm` applied once at the very end, not after each stack). Rewrote as `CosyVoiceV2UpsampleConformerEncoder`/`CosyVoiceV2Upsample1D` and dropped the length regulator (the real `CausalMaskedDiffWithXvec` doesn't have one; the token-rate-to-mel-rate ratio is fixed by construction). `CosyVoiceV2CausalConditionalDecoder` reuses `CosyVoiceV1ConditionalDecoder`, which was independently source-traced and fixed to the real diffusers-style estimator this same session, so it now matches `flow.pt`'s estimator too (not re-verified against the v2 checkpoint's estimator weights specifically this pass). Verified against real `FunAudioLLM/CosyVoice2-0.5B` `llm.pt`/`flow.pt`: strict `load_state_dict` on the LLM loads all 294 renamed keys (295 checkpoint keys minus the dropped `lm_head`) with 0 missing/0 unexpected, and a real forward+backward CE pass produces a finite loss (10.97) with nonzero gradients on all 294 params; the flow encoder's 211 non-decoder keys (of 1121 total in `flow.pt`) load with 0 missing/0 unexpected, and a forward+backward flow-matching loss with those real encoder weights is finite (2.18) with real gradients through the encoder. `hift.pt` unchanged from v1, not re-verified this pass. |
| CosyVoice v3 | Yes | Yes | Yes | Subclasses v2. Source-traced against `FunAudioLLM/CosyVoice`'s `cosyvoice/flow/DiT/dit.py`+`DiT/modules.py` (`DiT`/`DiTBlock`) and `cosyvoice/flow/flow.py`'s `CausalMaskedDiffWithDiT`. Found the same category of bug flagged by the cosyvoice_v1 incident: `CosyVoiceV3DiTBlock` used `nn.MultiheadAttention` plus a plain LayerNorm/GELU block, when the real block has separate `to_q/to_k/to_v/to_out` projections with interleaved rotary position embeddings, AdaLN-Zero modulation (6-way pre-attention+pre-ff, 2-way final) recomputed every block, and a plain (non-gated) tanh-GELU feed-forward; rewrote to match, with attribute names mirroring the real classes for a near-identity checkpoint mapping. Also found the flow model wrongly reused `CosyVoiceV1RelPositionEncoder` as a bidirectional Conformer text encoder plus a length regulator; the real `CausalMaskedDiffWithDiT` has neither — it upsamples the pre-lookahead-convolved token embedding straight to the mel rate via `repeat_interleave`. Added `CosyVoiceV3PreLookaheadLayer` matching the real in/out-channel widen-then-narrow bottleneck (`CosyVoiceV2PreLookaheadLayer`'s same-width version has the wrong conv shapes for this checkpoint) and removed the encoder/encoder_proj/length_regulator. Also source-traced `cosyvoice/hifigan/generator.py`: the real checkpoint's vocoder is `CausalHiFTGenerator` (three `[8,5,3]` causal upsample stages built from a `Conv1d`+`Upsample` pair with a different weight layout than `ConvTranspose1d`, causal-padded ResBlocks/conv_pre/conv_post, and a `CausalConvRNNF0Predictor` with a kernel-4 first layer), not the reused `CosyVoiceV1HiFTGenerator`; added `CosyVoiceV3HiftConfig`/`CosyVoiceV3HiFTGenerator` and its causal building blocks. Verified against the real `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` checkpoint (`llm.pt`/`flow.pt`/`hift.pt`, same three-file layout as v1/v2): `from_pretrained` loads all three with 0 missing/0 unexpected/0 mismatched keys (292, 329, 328 keys respectively), after fixing three more checkpoint-only-detectable bugs the random-weight-only prior pass couldn't catch — `CosyVoiceV3FlowConfig`/`CosyVoiceV3LLMConfig` had inherited v1/v2's Conformer-encoder/7B-scale Qwen2 defaults unchanged (real flow `input_size`=80 not 512, `vocab_size`=6561 not 4096; real LLM `hidden_size`=896/`intermediate_size`=4864/`num_hidden_layers`=24/`num_attention_heads`=14/`num_key_value_heads`=2/`rope_theta`=1e6, not Qwen2Config's 4096/22016/32/32/32/1e4 defaults), and `CosyVoiceV3HiFTGenerator`'s `stft_window` was a `register_buffer(..., persistent=False)` computed at `__init__` time, which `from_pretrained`'s meta-device init context silently turns into uninitialized memory instead of a real Hann window since nothing in the checkpoint populates a non-persistent buffer (no shape/dtype error, `torch.istft`'s NOLA check just fails on the first real forward call; recomputed fresh on every `_stft`/`_istft` call instead; the same latent bug in `CosyVoiceV1HiFTGenerator` — out of scope here — was fixed in a later pass, see the CosyVoice v1 row). With those fixed: a real forward+backward CE pass on the LLM produces a finite loss (10.70) with gradients on all 292 params and a plausible `generate()` speech-token sequence; a real forward+backward flow-matching pass on the DiT decoder produces a finite loss (4.02) with gradients on all 329 params and a correct-shaped, finite sampled mel from the Euler-solver inference path; and a full `generate_speech()` (LLM `generate` -> DiT flow decoder -> causal HiFT vocoder, all real weights) produces a finite, correctly-clamped waveform (RMS 0.049, max abs 0.82) — no NaN/Inf, no degenerate silence or clipping-at-limit. |
| F5-TTS | Yes | Yes | Yes | Full reimplementation, DiT flow-matching. Predicts mel spectrograms only; `F5TTSProcessor.decode` needs an external vocoder. `forward()` previously had no `labels`/loss path at all; added. Verified on CPU against `SWivid/F5-TTS` (`F5TTS_v1_Base/model_1250000.safetensors`): no weight converter existed, added `voicestudio/models/f5_tts/weight_conversion.py` (`WeightRenaming` rules registered via `register_checkpoint_conversion_mapping`) translating the original `ema_model.transformer.*` key layout; two bugs fixed along the way (renaming targets missing the `model.` `base_model_prefix`, and `WeightRenaming` only resolving one backreference per rule so a `(weight|bias)` alternation group silently matched nothing). `F5TTSTokenizer` already used a real `vocab.txt`, not `AutoTokenizer`, so no processor change was needed there. `generate()` produced a real mel spectrogram (shape `(1, 699, 100)`, mean -0.63/std 1.77, in the expected log-mel range) and `compute_training_loss` produced a real non-degenerate loss (2.22) with gradient norm ~89 on real weights. |
| PromptTTS++ (`prompt_tts_pp`) | Yes | No (no public checkpoint) | Yes | `FastSpeech2Conformer`/`FastSpeech2ConformerHifiGan` from transformers-tts, conditioned via a BERT-based prompt encoder. `return_dict=False` tuple-indexing bug fixed. Source-traced against `line/promptttspp` (`promptttspp/modules/prompt_encoder.py`'s `PromptEncoder`/`BertWrapper` and `promptttspp/models/prompttts_mdn_v2_final/model.py`'s `PromptTTSMDNDurCFG.forward`/`infer`): the prompt encoder itself (BERT `[CLS]` pooled through a 3-layer Linear/ReLU adaptor MLP) already matched. The style-conditioning mechanism did not: upstream adds the style embedding directly onto the phoneme encoder's output (`x = x + style_emb`, right before the variance adaptor) with no normalization, concatenation, or projection, but the migrated code fed it through `FastSpeech2ConformerModel`'s own `speaker_embedding` argument, which L2-normalizes, concatenates onto hidden states, and projects back down with an extra `Linear`, a different mechanism at the same insertion point. Fixed by overriding the acoustic forward (`PromptTTSppModel._acoustic_forward_with_style`) to add the style embedding directly, and dropped `model_config.speaker_embed_dim` wiring so the built-in concat+projection path is never activated. Not fixed/out of scope: upstream's real training pipeline also freezes all BERT parameters except the last layer's attention, feeds the prompt encoder through an MDN that is trained to match a separate mel-spectrogram-derived GST reference-encoder style embedding (`style_encoder.py`), and at text-only inference samples the style embedding from that MDN rather than using the adaptor output directly; VoiceStudio's implementation intentionally omits the MDN/reference-encoder/GST machinery and uses the adaptor output directly as the style embedding, a simplification that has no real upstream checkpoint to fall back on for the full pipeline either. Still not checkpoint-verified since no public checkpoint exists for either the full MDN pipeline or the simplified variant. |
| OmniVoice | Yes | Yes | Yes | No transformers-tts lineage (closest in spirit to CSM/Moshi); modeling code is new, audio tokenizer reused from transformers-tts's `HiggsAudioV2TokenizerModel`. Training-time sample masking is correctly out of scope: it lives in upstream `omnivoice/data/processor.py`'s `OmniVoiceSampleProcessor` and is training-collator logic, not a processor/model concern, so it is intentionally not in `voicestudio/models/omnivoice/`. Reference-audio auto-transcription (ASR) is real inference-time behavior: upstream `OmniVoice.create_voice_clone_prompt` (`omnivoice/models/omnivoice.py`) runs a Whisper ASR model (default `openai/whisper-large-v3-turbo`, configurable via `asr_model_name`/`asr_device`) to transcribe the reference audio whenever the caller passes `ref_text=None`, then merges the transcript into the rest of prompt construction. This has been ported into `OmniVoiceProcessor.encode_reference` (now `ref_text: Optional[str] = None`) plus a new `OmniVoiceProcessor.transcribe` method, using `transformers`-native `WhisperForConditionalGeneration`/`AutoProcessor` (default `openai/whisper-large-v3-turbo`, lazily loaded, no vendored ASR code). Verified on GPU against `k2-fsa/OmniVoice`: `OmniVoiceForConditionalGeneration.from_pretrained` and the standalone `HiggsAudioV2TokenizerModel` (loaded from the checkpoint's `audio_tokenizer/` subfolder) both load with 0 missing/0 unexpected/0 mismatched keys — no weight_conversion.py needed, the checkpoint's own key layout already matches. `OmniVoiceProcessor.from_pretrained` was broken for this checkpoint layout: `ProcessorMixin`'s generic sub-processor loader only checks preprocessor_config.json at the repo root (this checkpoint ships it only under `audio_tokenizer/`, alongside the codec weights) and routes any attribute whose name contains `"tokenizer"` through `AutoTokenizer` regardless of a hardcoded `*_class` override, so `audio_tokenizer` (a `HiggsAudioV2TokenizerModel`, not a text tokenizer) was never actually instantiated; fixed with an explicit `_get_arguments_from_pretrained` override. With real weights, `generate()` produced a real waveform (shape `(91200,)`, RMS 0.063, max abs 0.5 — plausible speech-level amplitude, not degenerate/silent/clipping), and a forward+backward pass with real audio-token labels (a real utterance re-encoded through the real audio tokenizer, not dummy random tensors) produced a non-degenerate loss (7.35) with real nonzero gradients on all 312/312 params (total grad norm 31.2). `OmniVoiceProcessor.encode_reference(ref_audio, ref_text=None)` was verified end to end with a real reference audio (LibriSpeech dev-clean sample `1272-128104-0000`): the auto-transcription step produced a real, coherent transcript ("He hoped there would be stew for dinner, turnips and carrots and bruised potatoes, and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce."), matching the sample's real content and not empty/garbage, and feeding that transcript into a `VoiceClonePrompt` and `OmniVoiceForConditionalGeneration.generate()` produced a real, non-degenerate waveform (shape `(103200,)`, RMS 0.040, max abs 0.45). |
| audiotools dependency removal | Done | | | No reference to `audiotools` remains in `pyproject.toml` or `voicestudio/`; already dead after the model migrations, nothing to change. |
| vocos dependency removal | Done | | | Dependency was declared in `pyproject.toml` but never imported anywhere in the codebase; `F5TTSProcessor.decode` already took a generic `vocoder` callable rather than importing `vocos` directly. Removed the unused `pyproject.toml` entry and reworded docstrings/README that named `vocos` as if required. No transformers-tts-native vocoder matches F5-TTS's mel config (24kHz, 100 mel channels), so a caller-supplied vocoder is still required at `decode()` time; repo not deleted per task instructions. |
| speechbrain fork removal | Done | | | Unused in repo; dependency dropped entirely (not switched to upstream). |
| UTMOSv2 decoupling | Done | | | `voicestudio/metrics/utmos.py` adapts UTMOSv2 to the `evaluate.Metric` interface; `utmosv2` git dependency dropped from `pyproject.toml`, now an optional runtime import. Upstream model code (five-fold SSL + image-classifier ensemble, hydra config system) not vendored, see module docstring. |

### Upstream git history

Done for the 12 models with real upstream code repos (`git filter-repo
--to-subdirectory-filter` per model, then `git merge --allow-unrelated-histories -X ours`
onto `develop`). The grafted commits are reachable ancestry (`git log --all` shows real
upstream authors), not a checked-out file tree: the actual checked-out files from
`filter-repo` were removed in a follow-up commit after Dependabot flagged hundreds of
long-fixed vulnerabilities in those old dependency manifests. Higgs TTS 3 has no
upstream code repo to graft (weights-only checkpoint).
