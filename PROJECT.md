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
step. For audio tokenizer models, follow the Qwen3-TTS and Higgs v2 examples. For
Parler, switch to the `dac` implementation already registered in `transformers` instead
of vendoring DAC.

Target usage shape:

```python
model_id = "eustlb/higgs-audio-v2-generation-3B-base"
model = HiggsAudioForConditionalGeneration.from_pretrained(model_id).to(device)
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
| Parler-TTS | Yes | No | Yes | Import bug (`isin_mps_friendly`) fixed. Still a ~3300-line near-verbatim port with scattered `# Copied from` comments rather than a `modular_parler_tts.py`; revisit for a proper modular conversion. |
| Higgs-Audio v2 | Yes | No | Yes | Import relay to transformers-tts. Reviewed clean, no issues found. |
| Higgs-Audio v3 | Yes | Yes | No | `bosonai/higgs-tts-3-4b`, weights-only checkpoint (no upstream v3 code, so nothing to graft). Reuses transformers-tts's `Qwen3Model`/`HiggsAudioV2*` classes; `HiggsAudioV3Model`/`ForConditionalGeneration` are new. `audio_labels`-only crash and default-off text CE loss fixed. Verified on GPU: the checkpoint's `body.`/`tied.` weight namespace didn't match this model's parameter names, so every real weight loaded as MISSING/UNEXPECTED and the model ran on random init; fixed via a registered checkpoint conversion mapping. The checkpoint also ships no `preprocessor_config.json`/audio tokenizer, which crashed `HiggsAudioV3Processor.from_pretrained`; it now degrades to a tokenizer-only processor for text-only use. With real weights loaded, forward logits and a CE loss (labels + audio_labels) with real backprop gradients were confirmed. |
| Chroma | Yes | No | Yes | Backbone/decoder reimplemented against transformers-tts's Llama, Qwen2.5-Omni thinker, and Mimi codec classes. Processor kwargs-merging and a labels-dereference guard fixed. |
| Spark-TTS | Yes | No | Yes | Config previously had no LLM sub-config, so plain construction silently produced an untrained model; fixed. License header consolidated onto `modeling_spark_tts.py` only; dead code and processor/model duplication cleaned up. |
| Dia | Yes | Yes | Yes | Import relay to transformers-tts's native Dia. Loaded `nari-labs/Dia-1.6B-0626` on GPU and ran `generate()` successfully; no code changes needed. |
| CosyVoice v1 | Yes | No | Yes | LLM (`llm.pt`) verified against FunAudioLLM/CosyVoice-300M: text/LLM encoder was `Wav2Vec2ConformerEncoder` (always has conv module + macaron FFN + no input projection), rewritten to match the real WeNet-style single-FFN/no-conv encoder plus the input `Linear+LayerNorm` it was missing; real weights now load with 0 missing/0 unexpected/0 shape-mismatch keys and a CE forward/backward on real weights produces a finite loss and gradients on all 401 params (see weight_conversion.py `build_llm_weight_conversion_mapping`). HiFTGenerator's `ResBlock` was missing the checkpoint's learnable Snake activation (used fixed leaky_relu) and `F0Predictor` was missing `weight_norm`; both fixed (`build_hift_weight_conversion_mapping`), not yet checkpoint-verified end-to-end. Flow decoder's `CosyVoiceV1ConditionalDecoder`/`CosyVoiceV1EstimatorBlock` transformer blocks do NOT match `flow.pt`: the real block is diffusers-style (separate `to_q/to_k/to_v/to_out`, GEGLU feed-forward `ff.net.0.proj`/`ff.net.2`, `norm1`/`norm3` AdaLN time-conditioning, no `norm2`) vs. the current `nn.MultiheadAttention` + plain-GELU + `norm1`/`norm2`/`time_proj` implementation; needs a rewrite before `flow.pt` can load, same as ONNX speech-tokenizer/campplus wiring in the processor and v2/v3 extension - none of that is done yet. |
| CosyVoice v2 | Yes | No | Yes | Subclasses v1; `Qwen2Model` LLM backbone. |
| CosyVoice v3 | Yes | No | Yes | Subclasses v2. DiT attention-mask shape bug (crashed on padded batches) fixed. |
| F5-TTS | Yes | Yes | Yes | Full reimplementation, DiT flow-matching. Predicts mel spectrograms only; `F5TTSProcessor.decode` needs an external vocoder. `forward()` previously had no `labels`/loss path at all; added. Verified on CPU against `SWivid/F5-TTS` (`F5TTS_v1_Base/model_1250000.safetensors`): no weight converter existed, added `voicestudio/models/f5_tts/weight_conversion.py` (`WeightRenaming` rules registered via `register_checkpoint_conversion_mapping`) translating the original `ema_model.transformer.*` key layout; two bugs fixed along the way (renaming targets missing the `model.` `base_model_prefix`, and `WeightRenaming` only resolving one backreference per rule so a `(weight|bias)` alternation group silently matched nothing). `F5TTSTokenizer` already used a real `vocab.txt`, not `AutoTokenizer`, so no processor change was needed there. `generate()` produced a real mel spectrogram (shape `(1, 699, 100)`, mean -0.63/std 1.77, in the expected log-mel range) and `compute_training_loss` produced a real non-degenerate loss (2.22) with gradient norm ~89 on real weights. |
| PromptTTS++ (`prompt_tts_pp`) | Yes | No (no public checkpoint) | Yes | `FastSpeech2Conformer`/`FastSpeech2ConformerHifiGan` from transformers-tts, conditioned via a BERT-based prompt encoder. `return_dict=False` tuple-indexing bug fixed. Source-traced against `line/promptttspp` (`promptttspp/modules/prompt_encoder.py`'s `PromptEncoder`/`BertWrapper` and `promptttspp/models/prompttts_mdn_v2_final/model.py`'s `PromptTTSMDNDurCFG.forward`/`infer`): the prompt encoder itself (BERT `[CLS]` pooled through a 3-layer Linear/ReLU adaptor MLP) already matched. The style-conditioning mechanism did not: upstream adds the style embedding directly onto the phoneme encoder's output (`x = x + style_emb`, right before the variance adaptor) with no normalization, concatenation, or projection, but the migrated code fed it through `FastSpeech2ConformerModel`'s own `speaker_embedding` argument, which L2-normalizes, concatenates onto hidden states, and projects back down with an extra `Linear`, a different mechanism at the same insertion point. Fixed by overriding the acoustic forward (`PromptTTSppModel._acoustic_forward_with_style`) to add the style embedding directly, and dropped `model_config.speaker_embed_dim` wiring so the built-in concat+projection path is never activated. Not fixed/out of scope: upstream's real training pipeline also freezes all BERT parameters except the last layer's attention, feeds the prompt encoder through an MDN that is trained to match a separate mel-spectrogram-derived GST reference-encoder style embedding (`style_encoder.py`), and at text-only inference samples the style embedding from that MDN rather than using the adaptor output directly; VoiceStudio's implementation intentionally omits the MDN/reference-encoder/GST machinery and uses the adaptor output directly as the style embedding, a simplification that has no real upstream checkpoint to fall back on for the full pipeline either. Still not checkpoint-verified since no public checkpoint exists for either the full MDN pipeline or the simplified variant. |
| OmniVoice | Yes | No | Yes | No transformers-tts lineage (closest in spirit to CSM/Moshi); modeling code is new, audio tokenizer reused from transformers-tts's `HiggsAudioV2TokenizerModel`. Training-time sample masking and reference-audio auto-transcription (ASR) not ported. |
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
long-fixed vulnerabilities in those old dependency manifests. Higgs-Audio v3 has no
upstream code repo to graft (weights-only checkpoint).
