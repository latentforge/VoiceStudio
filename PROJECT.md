# VoiceStudio Migration Project

This document records the plan to migrate VoiceStudio away from the namespace-package
architecture and into a single repository where every supported model is implemented
directly against the `transformers` model API. It exists so the plan survives context
resets across sessions and agents. Update it as work progresses; do not let it go stale.

## Workspace layout

Model weights live under `.cache/` in the repository checkout, which is gitignored:

- `.cache/huggingface` is `HF_HOME`, holding everything `huggingface_hub` downloads.
- `.cache/torch` is `TORCH_HOME`.

Both are mapped in two places so they apply everywhere, not only inside the editor.
`.vscode/settings.json` sets them for VS Code integrated terminals, and `.env` sets
them for every other shell, which is what agent tool calls and `uv run` actually use.
`.env` is gitignored, so recreate it if it is missing; without it a download silently
falls back to `~/.cache/huggingface` outside the editor.

`ckpts/` is the old location and is gitignored. Nothing new goes there.

### Downloading models for local testing

- Always download through `hf_xet`, the accelerated Hugging Face download path.
- Let `huggingface_hub` resolve its own cache under `HF_HOME`. Do not pass an explicit
  local target that writes outside `.cache/`.
- Nothing under `.cache/` is an artifact of the migration. Do not treat anything there
  as something to preserve or commit.

Do not create git worktrees. This harness bases a new worktree on `main`, which is far
behind `develop`, and that has silently blocked several tasks here.

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
| Breeze TTS 2 | Yes | Yes | Yes | `2cc89500`. Inherits CSM for the depth decoder, Qwen3 for the backbone; native `T5Gemma2TextEncoder` and `MimiModel`. wav2vec2 transcribed the generated speech; 1115 tensors load clean. Gaps in the folder README. |
| Chroma | Yes | Yes | Yes | `76f1d989`. CSM lineage plus a `Qwen2_5OmniThinker` reasoner. wav2vec2 transcribed a coherent answer; 1906 tensors, 5.92B parameters load clean. Its processor inherits `Qwen2_5OmniProcessor`, so `pillow` and `torchvision` arrive through the `omni` extra (`34df981c`). |
| Dia | Yes | Yes | Yes | Import relay; the model ships in transformers-tts. `08168e8b` transcribed nine of nine generations. Scripts must open with `[S1]` and run five to twenty seconds. |
| Dia2 | Yes | Yes | Yes | `26aee709`. Llama/Qwen3 lineage with a per-depth-position weight schedule that ruled out csm and higgs_audio_v2. `3338f278` replaced upstream's `whisper-timestamped` prefix alignment with native Whisper `return_token_timestamps`. |
| Dia2 alignment refinement | Yes | Yes | n/a | `3338f278` then `ac769d10`. Upstream's `whisper-timestamped` is a wrapper around OpenAI's own Whisper weights, not a model, so it was replaced rather than ported: native `return_token_timestamps` for the coarse DTW pass, and the second windowed pass inlined from `whisper_timestamped/transcribe.py` v1.15.9 step by step. Only the efficient path applies, because upstream's `load_model` converts the HF weights into a native openai-whisper model, which leaves `naive_approach` False. Four deviations are traced in the folder README, the largest being that `max_duration` masking is dead code on this path since the mel never ends in an exact zero, measured at -0.7350 and -0.5438 on the two test clips. Refinement moves 25 of 26 and 43 of 44 boundaries, every shift inside the margin. Word grouping still uses OpenAI's `_combine_tokens_into_words` rather than upstream's punctuation-aware split; recorded as open. |
| F5-TTS | Yes | Yes | Yes | `455b73f3`. Backbone checked numerically against a reimplementation of the upstream forward: max divergence 1.2e-05 to 3.0e-04 on |max| 13, growing smoothly from layer 0, so float32 accumulation order rather than structure. Three checkpoints transcribe word for word. `torchdiffeq` replaced by an inlined fixed-step solver that matches it bit for bit. |
| Higgs TTS 2 | Yes | Yes | Yes | `62bf46ab`. Both prompts transcribe verbatim; 5.38B parameters load clean. `max_new_tokens`, `add_generation_prompt` and moving the audio tokenizer to the model device are all load-bearing. |
| Higgs TTS 3 | Yes | Yes | Yes | `dfeb00d3` after the prompt-format and delay-pattern fix in `b4b18e5b`. Both prompts transcribe verbatim. 528 unexpected keys, all the bundled codec copy, logged as open. |
| Parler-TTS | Yes | Yes | Yes | DAC conversion coverage confirmed: all 301 source `audio_encoder.model.*` tensors are consumed with zero reported unused, including 54 of 54 `in_proj`/`out_proj` keys, which `convert_dac_checkpoint`'s `apply_weight_norm` and its single `quantizer.quantizers.*` wildcard cover without special casing. Round trip 11.46 dB against the 8.82 dB baseline, and re-randomising just those 36 projection tensors collapses it to -1.10 dB, which calibrates the check against the failure mode it was looking for. `992d149c` replaced the vendored `descript-audio-codec` with native `DacModel`. `b58b94cf` added the missing `ParlerTTSProcessor` and declared `sub_configs`, without which transformers 5 skipped the decoder during dtype resolution and half-precision loads crashed. Transcribes verbatim in float16 and float32. |
| PromptTTS++ (`prompt_tts_pp`) | Yes | Yes | Yes | `6fabba05`. The section 2.7 gap is closed: MDN, GST reference encoder and `GaussianDiffusion` decoder are all implemented and no FastSpeech2Conformer path remains. The checkpoint is bundled in the Space `line-corporation/promptttspp`, which the section 2.3 search confirmed is the only source. wav2vec2 WER 0.222 and 0.286. The BERT freeze not surviving `from_pretrained` was fixed and verified by parameter count. |
| Qwen3-TTS | Yes | Yes | Yes | Import relays plus a task-dispatching processor. `af7968ea` fixed voice design running in streaming mode: 17 of 17 prompts now transcribe verbatim. `encoder.upsample.conv.weight` reports MISSING, which is inert and upstream; see the open item above. |
| Spark-TTS | Yes | Yes | Yes | `c0be998c`. WER 0.000 across voice cloning, attribute creation and prompt continuation. `freeze_semantic_model` never called `.eval()`, so dropout, layerdrop and SpecAugment kept running in the frozen feature source and step-0 loss was irreproducible; fixed. |
| Spark-TTS BiCodec (`spark_tts_bicodec`) | Yes | Yes | Yes | `b2e78da3` split it into its own folder, following `higgs_audio_v2_tokenizer`. Round trip transcribes identically to the source clip. Objective taken from SparkVox's `loss_lambdas`, not the inference repo. |
| VoxInstruct | Yes | Yes | Yes | `75800c0e`. Both stages transcribe verbatim. Teacher forcing gives ar_loss 2.48 and nar_loss 3.50 against 8.25 and 8.09 for shuffled targets, and gradients land only on the LoRA adapters, both decoders and the drawn residual head. The Vocos vocoder is not yet ported; see the folder README. |
| BigVGAN | Yes | Yes, independently | n/a (NVIDIA source traced, no upstream tree was vendored) | `687502ad`. Its own model folder rather than a subclass of Qwen2.5-Omni's copy, which would have made the general case depend on one consumer. Re-verified from scratch on a remote GPU after the status row was found to rest on the migration commit's own prose: clean load with all 783 source tensors accounted for, 565 consumed and 218 resampling filters rebuilt from config, copy synthesis log mel L1 0.0887 against the 0.0886 calibration point in `d35f867f`, collapsing to 3.13 and 4.35 under negative controls on `conv_post` and `resblocks[0]`, and both consumers reloading clean and transcribing verbatim (`f2041f4c`). `1abdcdc4` reparents PromptTTS++'s BigVGAN, AMP block and Snake classes onto it, the first sibling inheritance in the repo, and `d35f867f` pairs `F5TTS_Base_bigvgan` with it: the old pairing shipped Vocos, which transcribed word for word while running 37 times too quiet at a copy-synthesis log mel distance of 3.895 against 0.0886. |
| CosyVoice v1 | Yes | Yes | Yes | `61bc1224`, then `86a9fa18` and `4bec4b53` took the folder from 166 files to a flat eight, every deletion named in the README File map. The 25 Hz tiktoken tokenizer is migrated as `tokenization_cosyvoice_v1.py`, built on transformers' `TikTokenConverter` reading the rank file with stdlib base64, matching a reference BPE on 12 of 12 strings; `git log --follow` needs `-M20%` to cross that rename, since 199 of 327 lines changed. `CosyVoiceV1ResBlock` inherits `HifiGanResidualBlock` from speecht5 with checkpoint keys unmoved, and the objective's mel comes from `voicestudio/models/bigvgan/` rather than a second copy. The generator itself is not inherited: all four transformers HiFi-GANs lack the f0 predictor, source module and iSTFT head that reach this checkpoint. Copy synthesis gives a log mel distance of 0.105986 against 3.905116 for same-energy noise, and `compute_f0` is exactly equal to upstream's, which needs the interpolation kept in float64. `diffusers`, `matcha-tts`, `einops`, `omegaconf`, `hyperpyyaml` and `openai-whisper` removed; `onnxruntime` could not be, because upstream publishes the v1 speech tokenizer and speaker encoder as ONNX graphs only, so it is reported and lazily imported rather than added to pyproject. Two meta-device buffers came back uninitialised after a reload and turned the fox sentence into `HADNIN YOUR DET SSFUL I SA YO TO DEING HEE`; both now build on first use. |
| CosyVoice v2 | Yes | Yes | Yes | `08e20c74`, special tokens fixed in `d888c4ab`. Its converter passed a bare `AutoTokenizer`, so 17 of upstream's 19 markers were absent and split into ordinary pieces: the model spoke `[laughter]` aloud, transcribing as `LAUTYR`, and no longer does. Only `<|im_start|>` and `<|im_end|>` were already present, as base Qwen2 chat tokens. No embedding resize is needed; the table is 151936 rows against a fixed tokenizer length of 151663. Subclasses v1. Every flow matching component is bit exact against upstream's own classes with streaming on and off, as are the sine generator and the neural source filter; the vocoder differs by 1.2e-05, the scipy against torch Hann window floor. |
| CosyVoice v3 | Yes | Yes | Yes | `961705ce`. Subclasses v2, and takes `F5TTSTimestepEmbedding`, `F5TTSDecoderLayer`, `F5TTSAdaLayerNormFinal` and `F5TTSRotaryEmbedding` from f5_tts, which is the DiT lineage the sibling map predicted. |
| OmniVoice (`ommivoice`) | Yes | Yes | Yes | `a452fcd5`. 90 files to 9. All 313 tensors and 612,577,280 parameters load with every source weight consumed, the fused embedding is bit identical to upstream's formula, and `forward` matches an independent reimplementation to 4.8e-07 against 20.17 on shuffled targets. wav2vec2 WER 0.000. The processor absorbs Whisper transcription of a missing reference transcript, which ends the pydub, soundfile and librosa dependencies. |
| Vocos | Yes | Yes | n/a (lifted from f5_tts, no upstream tree was vendored) | `f3e409be` and `cf5b12bf`. Own model folder covering both published frontends. Checked against upstream's own classes from the same weights: 1.5e-08 mel, 1.8e-07 encodec, and bit-for-bit on real VoxInstruct codes. `charactr/vocos-encodec-24khz` carries `feature_extractor.codebook_weights`, a trained 16384x128 parameter that the inherited blanket `feature_extractor.` ignore rule would have discarded silently. The adversarial half of the objective is not implemented; see the open item below. |
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

## Salvaged from the abandoned `wip/ai-migration-2026-08-19` branch

Recorded here before `/home/work/voice_research/VoiceStudio-backup`, that branch's worktree, was
removed. The branch itself, its `origin` copy, and `922f3969` in develop's own history all still
carry the code, so nothing below needs the worktree to stay on disk.

Findings that had not reached develop:

- **Parler-TTS half-precision load.** `from_pretrained(dtype=torch.bfloat16)` casts only
  `text_encoder` and `embed_prompts`; `decoder` and `audio_encoder` stay at the checkpoint's
  float32, because all three submodels are built and loaded as independent `PreTrainedModel`s.
  Cross-attention then mixes dtypes and crashes. The `from_pretrained` override sets `_fast_init`
  and nothing else. Not caught here because verification ran in float32 on a T4.
- **`get_text_config()` and composite configs.** Transformers 5's implementation probes only the
  names `decoder`, `generator`, `text_config` and `text_encoder`, and never consults `sub_configs`.
  Any composite model whose sub-config is named something else needs an override. Check this for
  ommivoice, cosyvoice and spark_tts before their migrations land.
- **OmniVoice reference transcription.** Upstream `OmniVoice.create_voice_clone_prompt` transcribes
  the reference audio with Whisper when `ref_text` is `None`, defaulting to
  `openai/whisper-large-v3-turbo`, and folds the result into the prompt. That is inference-time
  behaviour, so it belongs in the processor.
- **CosyVoice conversion facts**, useful when that migration starts: `f0_predictor.condnet` holds
  five convolutions at indices 0, 2, 4, 6 and 8; HiFT's `stft_window` as a non-persistent buffer is
  left uninitialised under meta-device init, so `torch.istft`'s NOLA check fails with
  `window overlap add min: 1`; `torch.cumprod(torch.tensor(...))` raises
  `NotImplementedError: Cannot copy out of meta tensor` in the same context and needs
  `itertools.accumulate` instead.
- **`WeightRenaming` substitutes only `\1`.** A two-group rule leaves a literal `\2` in the renamed
  key, which then goes MISSING without an error. This applies to every converter in the repo, not
  only CosyVoice.

Checked and already handled, recorded so they are not re-investigated: f5-TTS's converter reads
`ema_model_state_dict`, strips the `ema_model.` prefix and discards the `initted` and `step`
bookkeeping tensors, which is the EMA-only mapping the branch flagged; higgs_tts3's `body.`/`tied.`
conversion mapping, tokenizer-only fallback and missing-`preprocessor_config` tolerance all landed;
`Qwen3TTSConfig.get_text_config()` delegating to `talker_config` is in transformers-tts 5.16.0.dev0.

Three items left open:

- **CosyVoice's vocoder objective is half implemented.** `CosyVoiceV1HiFTGenerator.compute_loss`
  now returns the 45 times mel term and the f0 term, which is what could be written without a
  discriminator. Upstream's generator turn is `generator_loss` plus 2.0 times `feature_loss`, which
  doubles again inside, plus 45 times `mel_loss` plus 1.0 times `tpr_loss` at tau 0.04 plus the f0
  L1, alternating with a discriminator turn of `discriminator_loss` plus `tpr_loss` under its own
  optimizer. The three adversarial terms and that whole second turn stay open for the same reason
  Vocos's do: transformers carries no discriminator inside a model class, and none appears in a
  published checkpoint. `discriminator.py` was deleted rather than renamed, with every dropped class
  named in the folder README.
- **`convert("base", ...)` crashes for CosyVoice v2 and v3.** Both converters compute
  `resolved = PUBLISHED_CHECKPOINTS.get(source, source)` and then use it to locate the tokenizer
  directory, but for a shorthand key it stays the bare repo id rather than the local directory
  `load_upstream_checkpoints` downloaded to, so the tokenizer load raises `HFValidationError`.
  Passing a local directory, which is what the READMEs show, works. Found while verifying the
  special-token fix and not touched, since it is unrelated to it.
- **Vocos trains without its discriminators.** Upstream's generator loss is five terms: MPD hinge
  loss over periods 2, 3, 5, 7 and 11, `mrd_loss_coeff` times MRD hinge loss over FFT sizes 2048,
  1024 and 512, feature matching for each, and `mel_loss_coeff` times mel reconstruction, with a
  separate optimizer on the discriminators. `VocosModel.forward(labels=...)` implements the mel
  reconstruction term alone. `pretrain_mel_steps` is 0 in both released configs, so no phase of
  upstream training optimizes that term by itself, and a run against this loss is therefore not
  upstream's run. The stated reasons for leaving it out are that the discriminators are
  training-only modules absent from every published checkpoint, that no transformers model carries
  a discriminator in its model class, and that they would pull in `einops`. That is a scope
  decision, not a finding, and it needs a human.
- **Qwen3-TTS audio tokenizer reports `encoder.upsample.conv.weight` MISSING.** The key is real but
  the weight is not: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`'s `speech_tokenizer/model.safetensors`
  holds 496 tensors, none named `encoder.upsample.*` and none of the required `(512, 1, 4)` shape,
  so there is nothing to patch in the way `weight_conversion.py` patches the code predictor's
  `lm_head`. The cause is a divergence in transformers-tts between
  `modular_qwen3_tts_tokenizer_multi_codebook.py`, which nulls `decoder`, `decoder_transformer` and
  `upsample` after `super().__init__(config)`, and the generated
  `modeling_qwen3_tts_tokenizer_multi_codebook.py`, which nulls `upsample` at line 1999 and then
  rebuilds it at line 2012 inside the `frame_rate != encodec_frame_rate` branch without nulling it
  again. Its own comment still says it nulls upsample, so the generated file contradicts its stated
  intent. The module is dead either way: forward hooks show it is called zero times during both
  `encode()` and `decode()`, and overwriting the weight with random values leaves the encoded codes
  bit-identical and the decoded waveform bit-exact, against a positive control where the same
  treatment of `encoder.downsample.conv.weight` changes 99.9 percent of the codes. H7 forbids
  hand-editing a generated file, so the fix belongs in transformers-tts's modular source; the
  warning is harmless until then.
Nothing was salvageable for generation defaults. The branch's `PROJECT.md` records no
`temperature`, `top_k`, `top_p`, `guidance`, `do_sample` or `max_new_tokens` value for any
checkpoint, which follows from its verification standard: a plausible waveform RMS passes even when
the sampling parameters and the generation length are wrong. It cleared Higgs TTS 2 at
"RMS 0.091, max abs 0.69, plausible speech-level amplitude" on the same `max_length=53` truncation
that this repo later caught by transcribing the audio.

## Sibling inheritance map

Principle 1 asks for inheritance between models inside `voicestudio/models/`, not only from
transformers. No cross-folder import exists yet. This is the measured list of candidates, taken by
normalising every class name in every `modeling_*.py` against its model prefix and keeping the names
that appear in more than one folder. It binds the migrations still to come: a migration that
reimplements something on this list has to say why inheriting was rejected.

Actionable once `voicestudio/models/bigvgan/` exists:

- `PromptTTSPPBigVGan`, `PromptTTSPPAmpBlock`, `PromptTTSPPAmpLayer`, `PromptTTSPPSnakeActivation`
  should inherit it. Upstream's `F0AwareBigVGAN` is stock BigVGAN plus an NSF source module, so the
  F0 path is the only part that stays local.

Gated on the CosyVoice migration, which still has to check these before writing anything:

- `FeedForward` and `TimestepEmbedding` against f5_tts. Both are flow-matching DiT models, and
  f5_tts's DiT and fixed-step solver are numerically verified against upstream, so they are the
  stronger base.
- `Encoder` and `EncoderLayer` against prompt_tts_pp, which inherits the conformer submodules from
  `FastSpeech2Conformer` in transformers rather than reimplementing them.
- `SourceModule` against prompt_tts_pp. `Snake` against bigvgan was measured and rejected: bigvgan's
  activation carries an anti-aliasing sandwich and a different `layers.<n>.conv1` key layout.

Gated on the OmniVoice migration:

- Its backbone-plus-depth-decoder shape is closest to breeze_tts and chroma, both of which inherit
  `Csm*`. Note that dia2 evaluated the same lineage and rejected it, because its depth decoder
  selects weights per depth position, so the check is real work rather than a formality.

Measured and rejected, recorded so it is not re-litigated:

- `SparkTTSConvNeXtBlock` and `VocosConvNeXtBlock` compute the same thing, but their AdaLayerNorms
  do not: Vocos looks scale and shift up in an `nn.Embedding` keyed by a discrete bandwidth id,
  while Spark projects them from a continuous conditioning vector with `nn.Linear`. Sharing the
  block would mean growing a second conditioning mode into the Vocos model to serve Spark's codec.
  The blocks stay separate.
