# VoiceStudio Migration Project

This document records the plan to migrate VoiceStudio away from the namespace-package
architecture and into a single repository where every supported model is implemented
directly against the `transformers` model API. It exists so the plan survives context
resets across sessions and agents. Update it as work progresses; do not let it go stale.

## Workspace layout

Model weights live under `.cache/` in the repository checkout, which is gitignored:

- `.cache/huggingface` is `HF_HOME`, holding everything `huggingface_hub` downloads.
- `.cache/torch` is `TORCH_HOME`.
- `.cache/huggingface/converted` holds checkpoints converted from a published layout, keyed by the
  source repository and its resolved commit. Delete it to force a reconversion; nothing else reads it.

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
| Promptttspp | **Agents space** | Settled: the Space is the checkpoint source and `voicestudio/models/prompt_tts_pp/` loads from it. `https://huggingface.co/spaces/line-corporation/promptttspp` is a Gradio Space rather than a model repo, so the weights sit in the Space repository itself; the section 2.3 search found no separate `from_pretrained`-loadable repo and confirmed this is the only one. Upstream code: `https://github.com/line/promptttspp`. Verified in `6fabba05` at wav2vec2 WER 0.222 and 0.286. |
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

## Rules for the rewritten model code

- Every model must be trainable with cross-entropy loss, matching the standard
  `transformers` convention: `forward()` accepts `labels` and returns a `loss` on the
  output. Inference-only forward passes are not acceptable.
- Follow `transformers` model file conventions strictly: `modeling_<model>.py`,
  `configuration_<model>.py`, standard class inheritance, etc. Do not create files that
  fall outside the `transformers` model layout.
- Where a model already exists in `transformers` itself, only do an import relay, and do
  not reimplement it.
- Convert at load time, per CLAUDE.md section 9.2 and H17. A published repository id must load with no manual conversion call.
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
preprocessing goes through the model's `Processor`, with no separate manual preprocessing
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
VoiceStudio to restore it rather than forking the whole processor.

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
| F5-TTS | Yes | Yes | Yes | `455b73f3`. Backbone checked numerically against a reimplementation of the upstream forward: max divergence 1.2e-05 to 3.0e-04 on |max| 13, growing smoothly from layer 0, so float32 accumulation order rather than structure. All five published checkpoints load directly with no conversion call (`a0855a9d`) and transcribe word for word: `F5TTS_v1_Base`, `F5TTS_v1_Base_no_zero_init`, `F5TTS_Base`, `E2TTS_Base` and `F5TTS_Base_bigvgan`. The mel front end is recorded in no checkpoint file, so it is carried per checkpoint name alongside the vocoder, which closes the `d35f867f` pairing trap by construction; the bigvgan entry reproduces its calibration at copy synthesis log mel L1 0.0884 against Vocos's 3.8544 on the same clip. The old table also looked for a vocabulary file in `SWivid/E2-TTS`, which 404s, since only `SWivid/F5-TTS` ships one. `torchdiffeq` replaced by an inlined fixed-step solver that matches it bit for bit. `4189b2a3` records the output level. On the Vocos path the family peaks past full scale, 19 of 20 draws over five seeds, `F5TTS_Base` highest at 1.92 and rising with `guidance_scale` (0.95 at 0.0, 2.36 at 3.0), which `sf.write`'s `PCM_16` default clips. Upstream computes the same array through the same two conditional RMS operations, peak normalizes nothing and writes 16 bit through the same call, so the behaviour is left alone and the folder README carries the caveat, the same shape as the Spark-TTS continuation finding. `F5TTS_Base_bigvgan`'s exact 1.0000 is BigVGAN's `torch.clamp` under `use_tanh_at_final: false`, not headroom. |
| Higgs TTS 2 | Yes | Yes | Yes | `62bf46ab`. Both prompts transcribe verbatim; 5.38B parameters load clean. `max_new_tokens`, `add_generation_prompt` and moving the audio tokenizer to the model device are all load-bearing. |
| Higgs TTS 3 | Yes | Yes | Yes | `dfeb00d3` after the prompt-format and delay-pattern fix in `b4b18e5b`. Both prompts transcribe verbatim. 528 unexpected keys, all the bundled codec copy, logged as open. |
| Parler-TTS | Yes | Yes | Yes | DAC conversion coverage confirmed: all 301 source `audio_encoder.model.*` tensors are consumed with zero reported unused, including 54 of 54 `in_proj`/`out_proj` keys, which `convert_dac_checkpoint`'s `apply_weight_norm` and its single `quantizer.quantizers.*` wildcard cover without special casing. Round trip 11.46 dB against the 8.82 dB baseline, and re-randomising just those 36 projection tensors collapses it to -1.10 dB, which calibrates the check against the failure mode it was looking for. `992d149c` replaced the vendored `descript-audio-codec` with native `DacModel`. `b58b94cf` added the missing `ParlerTTSProcessor` and declared `sub_configs`, without which transformers 5 skipped the decoder during dtype resolution and half-precision loads crashed. Transcribes verbatim in float16 and float32. |
| PromptTTS++ (`prompt_tts_pp`) | Yes | Yes | Yes | `6fabba05`. The section 2.7 gap is closed: MDN, GST reference encoder and `GaussianDiffusion` decoder are all implemented and no FastSpeech2Conformer path remains. The checkpoint is bundled in the Space `line-corporation/promptttspp`, which the section 2.3 search confirmed is the only source. wav2vec2 WER 0.222 and 0.286. The BERT freeze not surviving `from_pretrained` was fixed and verified by parameter count. |
| Qwen3-TTS | Yes | Yes | Yes | Import relays plus a task-dispatching processor. `af7968ea` fixed voice design running in streaming mode: 17 of 17 prompts now transcribe verbatim. `encoder.upsample.conv.weight` used to report MISSING; the backport closed it, and a fresh conversion loads at 0 MISSING and 0 UNEXPECTED. See above. |
| Spark-TTS | Yes | Yes | Yes | `c0be998c`. WER 0.000 under `whisper-large-v3-turbo` across voice cloning and attribute creation, and on two of three sentences under prompt continuation before `82893e09`, whose third lost its leading word to a fused token; all three are verbatim after it, see above. `freeze_semantic_model` never called `.eval()`, so dropout, layerdrop and SpecAugment kept running in the frozen feature source and step-0 loss was irreproducible; fixed. |
| Spark-TTS BiCodec (`spark_tts_bicodec`) | Yes | Yes | Yes | `b2e78da3` split it into its own folder, following `higgs_audio_v2_tokenizer`. Round trip transcribes identically to the source clip. Objective taken from SparkVox's `loss_lambdas`, not the inference repo. |
| VoxInstruct | Yes | Yes | Yes | `75800c0e`. Both stages transcribe verbatim. Teacher forcing gives ar_loss 2.48 and nar_loss 3.50 against 8.25 and 8.09 for shuffled targets, and gradients land only on the LoRA adapters, both decoders and the drawn residual head. `cf5b12bf` wired in the native `VocosModel` from `voicestudio/models/vocos/`, which closed that gap: `generate` takes `vocoder="vocos"` by default, matching upstream's `infer.sh`, with `"encodec"` as the alternative. |
| BigVGAN | Yes | Yes, independently | n/a (NVIDIA source traced, no upstream tree was vendored) | `687502ad`. Its own model folder rather than a subclass of Qwen2.5-Omni's copy, which would have made the general case depend on one consumer. Re-verified from scratch on a remote GPU after the status row was found to rest on the migration commit's own prose: clean load with all 783 source tensors accounted for, 565 consumed and 218 resampling filters rebuilt from config, copy synthesis log mel L1 0.0887 against the 0.0886 calibration point in `d35f867f`, collapsing to 3.13 and 4.35 under negative controls on `conv_post` and `resblocks[0]`, and both consumers reloading clean and transcribing verbatim (`f2041f4c`). `1abdcdc4` reparents PromptTTS++'s BigVGAN, AMP block and Snake classes onto it, the first sibling inheritance in the repo, and `d35f867f` pairs `F5TTS_Base_bigvgan` with it: the old pairing shipped Vocos, which transcribed word for word while running 37 times too quiet at a copy-synthesis log mel distance of 3.895 against 0.0886. |
| CosyVoice v1 | Yes | Yes | Yes | `61bc1224`, then `86a9fa18` and `4bec4b53` took the folder from 166 files to a flat eight, every deletion named in the README File map. The 25 Hz tiktoken tokenizer is migrated as `tokenization_cosyvoice_v1.py`, built on transformers' `TikTokenConverter` reading the rank file with stdlib base64, matching a reference BPE on 12 of 12 strings; `git log --follow` needs `-M20%` to cross that rename, since 199 of 327 lines changed. `CosyVoiceV1ResBlock` inherits `HifiGanResidualBlock` from speecht5 with checkpoint keys unmoved, and the objective's mel comes from `voicestudio/models/bigvgan/` rather than a second copy. The generator itself is not inherited: all four transformers HiFi-GANs lack the f0 predictor, source module and iSTFT head that reach this checkpoint. Copy synthesis gives a log mel distance of 0.105986 against 3.905116 for same-energy noise, and `compute_f0` is exactly equal to upstream's, which needs the interpolation kept in float64. `diffusers`, `matcha-tts`, `einops`, `omegaconf`, `hyperpyyaml` and `openai-whisper` removed; `onnxruntime` too, once `5bc89a95` ported both ONNX components to PyTorch: the CAM++ speaker encoder, and the speech tokenizer, whose weights are read straight out of the graph's protocol buffer rather than through a dependency on `onnx`. `CosyVoiceV1SpeechTokenizerLayer` inherits `WhisperEncoderLayer`, the graph being Whisper-large's first six encoder blocks with q and k each scaled by `head_dim**-0.25`. Speaker embeddings match the graph to 8.106e-06 and every speech token id is identical, 1158 of 1158 for v1 and 580 of 580 for v2 and v3, and zero-shot cloning now transcribes back word for word on all three. Two meta-device buffers came back uninitialised after a reload and turned the fox sentence into `HADNIN YOUR DET SSFUL I SA YO TO DEING HEE`; both now build on first use. |
| CosyVoice v2 | Yes | Yes | Yes | `08e20c74`, special tokens fixed in `d888c4ab`. Its converter passed a bare `AutoTokenizer`, so 17 of upstream's 19 markers were absent and split into ordinary pieces: the model spoke `[laughter]` aloud, transcribing as `LAUTYR`, and no longer does. Only `<|im_start|>` and `<|im_end|>` were already present, as base Qwen2 chat tokens. No embedding resize is needed; the table is 151936 rows against a fixed tokenizer length of 151663. Subclasses v1. Every flow matching component is bit exact against upstream's own classes with streaming on and off, as are the sine generator and the neural source filter; the vocoder differs by 1.2e-05, the scipy against torch Hann window floor. |
| CosyVoice v3 | Yes | Yes | Yes | `961705ce`. Subclasses v2, and takes `F5TTSTimestepEmbedding`, `F5TTSDecoderLayer`, `F5TTSAdaLayerNormFinal` and `F5TTSRotaryEmbedding` from f5_tts, which is the DiT lineage the sibling map predicted. |
| OmniVoice (`ommivoice`) | Yes | Yes | Yes | `a452fcd5`. 90 files to 9. All 313 tensors and 612,577,280 parameters load with every source weight consumed, the fused embedding is bit identical to upstream's formula, and `forward` matches an independent reimplementation to 4.8e-07 against 20.17 on shuffled targets. wav2vec2 WER 0.000. The processor absorbs Whisper transcription of a missing reference transcript, which ends the pydub, soundfile and librosa dependencies. |
| Vocos | Yes | Yes | n/a (lifted from f5_tts, no upstream tree was vendored) | `f3e409be` and `cf5b12bf`. Own model folder covering both published frontends. Checked against upstream's own classes from the same weights: 1.5e-08 mel, 1.8e-07 encodec, and bit-for-bit on real VoxInstruct codes. `charactr/vocos-encodec-24khz` carries `feature_extractor.codebook_weights`, a trained 16384x128 parameter that the inherited blanket `feature_extractor.` ignore rule would have discarded silently. The adversarial half of the objective is not implemented, which follows the `transformers` convention and is settled; see below. |
| audiotools dependency removal | Done | | | No reference to `audiotools` remains in `pyproject.toml` or `voicestudio/`; already dead after the model migrations, nothing to change. |
| vocos dependency removal | Done | | | Dependency was declared in `pyproject.toml` but never imported anywhere in the codebase; `F5TTSProcessor.decode` already took a generic `vocoder` callable rather than importing `vocos` directly. Removed the unused `pyproject.toml` entry and reworded docstrings/README that named `vocos` as if required. No transformers-tts-native vocoder matches F5-TTS's mel config (24kHz, 100 mel channels), so a caller-supplied vocoder is still required at `decode()` time; repo not deleted per task instructions. |
| speechbrain fork removal | Done | | | Unused in repo; dependency dropped entirely (not switched to upstream). |
| UTMOSv2 decoupling | Done | | | `voicestudio/metrics/utmos.py` no longer exists on this branch and `voicestudio/metrics/__init__.py` is empty, so nothing here consumes UTMOSv2. The `utmosv2` git source and its `eval` entry were both removed in `6b407586`. The adapter survives on `wip/ai-migration-2026-08-19` if it is ever wanted back. |

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

One item, now closed:

- **Qwen3-TTS audio tokenizer reports `encoder.upsample.conv.weight` MISSING.** The key is real but
  the weight is not: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`'s `speech_tokenizer/model.safetensors`
  holds 496 tensors, none named `encoder.upsample.*` and none of the required `(512, 1, 4)` shape,
  so there is nothing to patch in the way the conversion mapping registered in `modeling_qwen3_tts.py` fuses the code predictor's
  `lm_head`. The cause is a divergence in transformers-tts between
  `modular_qwen3_tts_tokenizer_multi_codebook.py`, which nulls `decoder`, `decoder_transformer` and
  `upsample` after `super().__init__(config)`, and the generated
  `modeling_qwen3_tts_tokenizer_multi_codebook.py`, which nulls `upsample` at line 1999 and then
  rebuilds it at line 2012 inside the `frame_rate != encodec_frame_rate` branch without nulling it
  again. Its own comment still says it nulls upsample, so the generated file contradicts its stated
  intent. The module is dead either way: forward hooks show it is called zero times during both
  `encode()` and `decode()`, and overwriting the weight with random values leaves the encoded codes
  bit-identical and the decoded waveform bit-exact, against a positive control where the same
  treatment of `encoder.downsample.conv.weight` changes 99.9 percent of the codes. The modular source is correct and the generated file is not, so neither editing it, which H7
  forbids, nor editing the modular file, which is already right, is the fix. The generator collapsed
  two identical looking `self.upsample = None` statements that had different positions: `MimiModel.__init__`
  nulls it before conditionally building it, and the modular subclass nulls it again afterwards. Inlining
  the parent merged the two, so the second null landed before the build.

  **Closed by the backport in `voicestudio/backport/`.** The generated file now clears the module with
  `setattr(self, "upsample", None)` rather than a plain assignment, which the modular converter does not
  fold into the parent's earlier null, so the module is gone rather than rebuilt unweighted. Measured on
  `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`: `encoder.upsample` is `None`, the load reports 0 MISSING and 0
  UNEXPECTED, and the parameter count is unchanged at 145,293,281. The stale conversion cache had to be
  deleted to see it. The cache was written before the backport, when the module still existed, so
  `save_pretrained` had written its randomly initialised weight into the converted directory: 513 tensors
  against the source's 496. A fresh conversion writes 512 and carries no `encoder.upsample.*` at all.
Nothing was salvageable for generation defaults. The branch's `PROJECT.md` records no
`temperature`, `top_k`, `top_p`, `guidance`, `do_sample` or `max_new_tokens` value for any
checkpoint, which follows from its verification standard: a plausible waveform RMS passes even when
the sampling parameters and the generation length are wrong. It cleared Higgs TTS 2 at
"RMS 0.091, max abs 0.69, plausible speech-level amplitude" on the same `max_length=53` truncation
that this repo later caught by transcribing the audio.

## Future direction: inference performance

No model here has been tuned for inference speed, and the migrations deliberately did not carry
upstream's optimized paths across. Breeze TTS 2 is the clearest case, since its upstream README
advertises 40 ms to first audio and a 0.32 real time factor measured on a hand written CUDA graph
capture (`models/cudagraph/`, `models/fast_streaming.py`, `models/warmup_profile.py`,
`configs/fast.json`), all of which `2cc89500` removed. Those graphs replay the same kernels this
code already runs, so nothing about accuracy changed, but that headline number does not reproduce
today. The same gap applies to every autoregressive codec language model here, which spend most of
their wall clock launching many small kernels per audio frame rather than computing.

The route back is the `transformers` one, not the vendored code. A static cache and CUDA graph
capture are selected by the caller through `GenerationConfig`, not implemented per model:
`cache_implementation` takes `"static"` or `"offloaded_static"` and is validated against
`ALL_CACHE_IMPLEMENTATIONS`, `generation/utils.py` branches on it at line 1980, and `compile_config`
takes a `CompileConfig`. So a model's obligation is to stay compatible with that path, with static
shapes and no per-step Python branching on tensor values, rather than to ship a capture of its own.

What is owed, when inference speed becomes a requirement rather than a nice to have: measure the
real time factor of each model under `cache_implementation="static"` with a `compile_config`, find
which ones fall out of the static path and why, and fix those. Restoring a vendored capture is a
last resort. Breeze TTS 2's `models/stream_runtime/` is not a candidate at all, since it is built
entirely on the third party `qwen_tts` package and H11 forbids taking that dependency back.

## Decided: the code predictor's rope base is 1000000

The published `config.json` sets `talker_config.code_predictor_config.rope_theta` to 1000000 under
the pre-5.0 field name, and transformers 5 replaced it with the class default of 500000:
`Qwen3TTSTalkerCodePredictorConfig.__init__` assigns `rope_parameters` after `super().__init__()`
has already migrated the old spelling, so the migrated value is overwritten.

A source trace settled which base is right rather than a preference. `qwen_tts` 0.1.1 pins
transformers 4.57.3, whose `_compute_default_rope_parameters` opens with `base = config.rope_theta`,
and `Qwen3TTSTalkerCodePredictorModel.__init__` builds its rotary embedding from the config that
carries the field (`modeling_qwen3_tts.py:991`, `configuration_qwen3_tts.py:237` and `:439`). So
upstream serves the code predictor at 1000000 and the dropped field was the defect.

The difference is real. Over the code predictor's 17 positions the rotary angles differ by up to
0.264 radians, and replaying identical inputs at the two bases moves 26 percent of the residual
codebook tokens, which shifts utterance length by up to 8 frames at a fixed seed. Both bases still
transcribe 17 of 17 verbatim, because the words come from the talker's first-layer token and the
talker was already at the right base; the code predictor only fills the residual codebooks. So the
earlier calibration stands and the fix does not regress it.

`b32a0a18` deletes the talker-only special case and normalizes the pre-5.0 rope spelling at any
depth instead. `transformers`'s own `convert_qwen3_tts_to_hf` still drops the field, firing only on
`rope_scaling`, so checkpoints written by that script stay at 500000; that fix belongs upstream.

## Audited: route 2 and the generation config

`from_pretrained` reads `generation_config.json` off the path it is handed, and passing `state_dict=`
requires that path to be `None` (`modeling_utils.py:4153` raises otherwise), so a route 2 load keeps
only the `GenerationConfig.from_model_config(config)` built in `__init__`. In parler_tts that lost the
published `max_length` of 2580 and produced 0.139 seconds of audio that transcribed to nothing, with
a clean load report, the right parameter count and no NaN. `e04b36f7` closed it structurally instead: a route 2 load now converts once into a cached
directory and hands `from_pretrained` that directory, so the ordinary path resolves
`generation_config.json` as usual and the `read_generation_config` workaround was removed.

Every other folder was then audited, and none needed one. `vox_instruct`, `f5_tts`, `prompt_tts_pp`,
`cosyvoice_v1`, `vocos` and `bigvgan` take the route 2 shape; `dia2` does not, since its override
builds only the config and a registered mapping does the tensor work inside the loading path but their published repositories ship no
`generation_config.json` at all; `vocos`, `bigvgan` and both PromptTTS++ classes report
`can_generate()` false, so the question does not arise for them. `breeze_tts`, `qwen3_tts`, `dia2`
and `spark_tts` keep the path and resolve theirs normally.

CosyVoice v2 and v3 do drop one, and it is deliberately left dropped. The only
`generation_config.json` in those repositories is `CosyVoice-BlankEN/generation_config.json`, the
base text language model's directory, and it is byte identical to `Qwen/Qwen2-0.5B-Instruct`'s stock
file. Upstream loads that body with `Qwen2ForCausalLM.from_pretrained` and only ever calls `forward`
on it; its real decode parameters live in `cosyvoice2.yaml` as `ras_sampling(top_p=0.8, top_k=25,
win_size=10, tau_r=0.1)`, which are already the defaults of the migrated `generate_speech_tokens`.
Its `eos_token_id` values are text-vocabulary ids, outside the speech head's 6564. Instrumenting
`model.generation_config` with a recording subclass logged zero attribute reads during generation,
and attaching the file changed the audio by 0.000003 max absolute against a same-seed control of
0.000003, which is GPU nondeterminism. Attaching it would graft a chat model's stop ids onto a
speech-token model for no behavioural gain.

Worth knowing if a repository ever does publish a root one:
`from_pretrained(None, state_dict=..., generation_config=<GenerationConfig>)` is accepted and
applied through `adjust_generation_fn`, which is a cheaper fix than parler's post-assignment and
preserves a model's own `generation_config_class`.

## Decided: the converted directory is the artifact, not a cache beside one

A successful conversion reclaims the source it converted from, through
`huggingface_hub`'s own cache API, so a model is stored once rather than twice. That is safe because
a cache hit resolves only a small file of the same revision and the key comes from the snapshot
path, `models--owner--repo@commit`, rather than from any file's bytes. Getting there took a pass per
folder, since most of them resolved their weights before consulting the cache and so re-downloaded
on every hit what the previous conversion had just reclaimed. CosyVoice v2's entry went from
3,549,561,596 bytes on a hit to 7,991, and its second load from 84.67 seconds to 2.19. Parler-TTS was
the last, and the odd one: its `converted_checkpoint` was already keyed on `config.json`, and the
re-download came from `ParlerTTSProcessor.from_pretrained` rebuilding the codec from the raw shards
instead of reading the conversion the model had already cached. Both now share one entry.

Reclamation covers only what the conversion actually fetched, not the whole revision, which
`3086274a` narrowed it to by listing the hub cache before and after the write and taking the
difference. The broad rule deleted files the conversion never opened: CosyVoice's
`speech_tokenizer_v1.onnx` is read by the processor and not by the conversion, so a processor built
before the model and used after it raised `FileNotFoundError`. A file already resident when a
conversion starts is exactly the file some earlier caller may hold a path to, and those now always
stay. The consequence is that a large file a processor fetches but no conversion reads now
persists.

`CACHE_VERSION` was removed with it. Versioning made sense for a derived copy sitting beside an
original; it does not for the only copy, and a constant somebody has to remember to bump is not
worth carrying.

The cost, which is real and was accepted rather than overlooked: a converter whose output changes
keeps serving what it wrote before, and with the source gone there is nothing to compare against.
Changing a converter means clearing `$HF_HOME/converted` by hand and letting the source download
again. Several converters changed today, so this is not hypothetical.

## Decided: the Auto classes are left alone

Five published repositories cannot be opened through an Auto class: `charactr/vocos-*` ships a
`config.yaml` and no `config.json` at all, while `nvidia/bigvgan_*` and `nari-labs/Dia2-*` ship a
`config.json` carrying the upstream project's own schema with no `model_type` key, and PromptTTS++'s
weights live in a Space. `AutoConfig.from_pretrained` picks a config class from `model_type` alone,
so it fails before any of this repository's code can run its `is_published_layout` probe.

A fallback is possible: catch that specific failure and ask each registered model whether it claims
the repository. It is deliberately not built. The concrete classes already load every one of these
directly, which is what their READMEs document, and a dispatcher that second-guesses
`AutoConfig` adds a code path that runs only when transformers has already given up. Models whose
published repository does carry a `transformers` config, which is most of them, work through the
Auto classes as usual.

## Fixed: Spark-TTS continuation lost a leading word at a fused token

`SparkTTSProcessor` joins the reference transcript and the text to speak as `prompt_text + text`
with no separator inside `<|start_content|>`, and the prompt it builds is token for token identical
to upstream `cli/SparkTTS.py::process_prompt` on the same codes, for every sentence and both
`add_special_tokens` settings. With a reference transcript ending `...his gospel.`, the seam becomes
one BPE token: `gospel.` plus `Actions` tokenizes as `Ġgospel`, `.Actions`, id 72044, so the first
word of the text is not a token of its own and that is the word that goes missing.

Measured over eight seeds under `whisper-large-v3-turbo`: no separator 0 of 8, one space added to
`prompt_text` 8 of 8. SparkVox's own `prepare_train.py` puts a single utterance's transcript between
the content markers and never concatenates a reference transcript, so continuation is inference only
and these fused sentence-boundary tokens are out of distribution. Copy synthesis of the reference
through BiCodec transcribes verbatim, so the semantic prefix is sound and the failure is on the text
side.

`82893e09` separates the join, which is a deliberate divergence from upstream rather than a match to
it. The separator is chosen by the script of the text being appended, because one rule does not cover
both: a space fixes English, and in Chinese it is the failure. Joined directly, `。` then `行动` reproduces the
second text's own tokenization exactly, since Qwen2's BPE carries no merge spanning a CJK boundary;
inserting a space instead produces `Ġ�`, `�`, `动`, splitting one character across two tokens. So a single
space goes in unless `prompt_text` already ends in whitespace or `text` opens on a Chinese character,
which also gets the mixed cases right in both directions. Re-measured at 8 of 8 verbatim on the
sentence that was 0 of 8, with the other two sentences and the other two layouts re-run at 70 of 72
over the same eight seeds.

One caveat on the upstream comparison: upstream was compared by token ids rather than executed, since
running it needs `einops` and `einx`, which H11 forbids installing.

## A calibration that is a single draw is not a calibration

Three of the models here sample at generation time, so a recorded transcript from one seed is a
point in a spread rather than a property to reproduce. PromptTTS++ was recorded at WER 0.222 and
0.286; re-run over eight seeds it scatters from 0.000 to 0.667 under wav2vec2 while
`whisper-large-v3-turbo` hears fifteen of sixteen clips correctly, so the model is healthy and the
two recorded numbers were never reproducible as stated.

The transcriber matters as much as the seed. wav2vec2 heard VoxInstruct's 1.6 second clip as
`FIRE A HOLE PLATOON MAJOR`, which reads as a failure until `whisper-large-v3-turbo` returns the
same clip verbatim, and the folder README had already recorded the wav2vec2 string. A number without
its transcriber named is not a result.

Where a model samples, record a seed sweep and name the transcriber, not one draw.

## Settled: GAN discriminators stay unimplemented

This one is closed. It follows the `transformers` convention, the convention is measured below, and
it is not an open scope decision in any folder. Do not reopen it.

**What is missing, per model.** Vocos is missing the MPD and MRD hinge generator losses and their
feature matching, leaving the mel reconstruction term. CosyVoice v1, v2 and v3 are missing
`generator_loss`, the feature matching loss and `tpr_loss`, plus the whole discriminator turn,
leaving the 45 times mel term and the f0 term. BigVGAN is missing `loss_gen_f`, `loss_gen_s`,
`loss_fm_f` and `loss_fm_s`, leaving the multiscale mel term. Spark-TTS BiCodec has the same shape.
PromptTTS++ is a different item and this does not settle it: its vocoder objective was never traced,
so it returns no loss at all rather than the non adversarial half of one.

**The convention, measured.** Over the 510 model directories of the installed `transformers-tts`
5.16.1 tree:

- `grep -rl "Discriminator" models/ --include="*.py"` returns exactly two files.
  `ElectraDiscriminatorPredictions` at `electra/modeling_electra.py:465` and
  `FunnelDiscriminatorPredictions` at `funnel/modeling_funnel.py:653` are ELECTRA style replaced
  token detection heads, a dense layer over hidden states, not adversaries over a waveform. Adding
  `-i` finds only `electra/configuration_electra.py` on top of those two.
- `MultiPeriodDiscriminator`, `MultiResolutionDiscriminator`, `MultiScaleDiscriminator` and
  `MultiScaleSTFT` return 0 hits over the whole package.
- `feature_loss`, `discriminator_loss`, `generator_loss`, `adversarial` and `hinge_loss` return 0
  files over the whole package. `transformers/loss/` holds eleven loss modules, all of them object
  detection, segmentation, RNNT, TDT or the `LOSS_MAPPING` cross entropy family, and none of them is
  adversarial or feature matching.
- Every vocoder ships generator only and inference only. `SpeechT5HifiGan.forward(spectrogram)`,
  `FastSpeech2ConformerHifiGan.forward(spectrogram)`,
  `VitsHifiGan.forward(spectrogram, global_conditioning)`,
  `SeamlessM4THifiGan.forward(inputs_embeds)`, `SeamlessM4Tv2HifiGan.forward(inputs_embeds)` and
  `Qwen2_5OmniToken2WavBigVGANModel.forward(mel_spectrogram)` each return a bare waveform tensor,
  take no `labels` and compute no loss. `SeamlessM4TCodeHifiGan` and `SeamlessM4Tv2CodeHifiGan` are
  the same with a speaker and language id. `VitsModel.forward` does take `labels`, and the first
  thing it does with them is `raise NotImplementedError("Training of VITS is not supported yet.")`
  at `vits/modeling_vits.py:1305`, even though VITS is adversarially trained upstream.
  `FastSpeech2ConformerModel` is the one audio model that computes a real synthesis loss, through
  `self.criterion` over `spectrogram_labels`, `duration_labels`, `pitch_labels` and `energy_labels`,
  and that is the acoustic model; the paired `FastSpeech2ConformerWithHifiGan.forward` passes those
  through and adds no vocoder term.
- Every codec ships the same way. `EncodecModel`, `MimiModel`, `XcodecModel`, `Xcodec2Model`,
  `HiggsAudioV2TokenizerModel` and `VibeVoiceAcousticTokenizerModel` take no `labels` and compute no
  loss, and the VibeVoice file contains neither word anywhere. `DacModel` is the sharpest case: DAC
  is trained upstream against a multi scale STFT discriminator with mel, feature matching and
  adversarial terms, and `DacOutput.loss` is
  `commitment_loss_weight * commitment + codebook_loss_weight * codebook` at
  `dac/modeling_dac.py:602` and nothing else. So the convention is not that a GAN trained model
  carries no objective. It is that it carries the part of its objective that needs no
  discriminator, and carries no discriminator.

**What follows.** `voicestudio/models/bigvgan` and `voicestudio/models/vocos` return the mel
reconstruction term from `forward(labels=...)`, and CosyVoice's vocoder returns the mel and f0 terms
from `compute_loss`, so these folders offer more training support than any vocoder in `transformers`
offers, not less. Nothing about inference or about fine tuning the rest of a model is affected.

Two facts stay on the record and neither reopens this. Training a vocoder from scratch through
`forward` alone would not reproduce the released weights, and there is no phase of upstream training
that optimizes the reconstruction term alone to fall back on, since `pretrain_mel_steps` is 0 in
both released Vocos configs and `--freeze_step` is 0 in BigVGAN's. And the checkpoint asymmetry: the
Vocos and CosyVoice discriminators appear in no published checkpoint, but BigVGAN's do, since every
`nvidia/bigvgan*` repository ships a `bigvgan_discriminator_optimizer.pt`, so weights to verify a
BigVGAN discriminator against do exist.

The folder READMEs name every dropped class.

## Settled: the remaining scope decisions

Taken together with the GAN discriminator decision above, these close the scope items the folder
READMEs had been carrying as needing a human. None of them is open any more.

**Direct preference optimization, CosyVoice v2 and v3: not implemented, closed.** `Qwen2LM.forward_dpo`
and `DPOLoss` need a second model instance held as a frozen reference and a batch carrying a rejected
sequence beside the chosen one, so they do not fit inside a single `forward`. Two things settle it
rather than one. No released CosyVoice checkpoint is preference tuned, so there is nothing to run it
against. And upstream averages its log probabilities over the positions where the target **is**
`IGNORE_ID`, which reads as a sign error, so the reference implementation is not one to copy.

**Vocos ResNet backbone and IMDCT heads: out of scope, closed, and there is nothing to leave behind.**
`configs/vocos-resnet.yaml` trains a HiFi-GAN style dilated residual backbone in place of the ConvNeXt
one, and `configs/vocos-imdct.yaml` a head predicting modified discrete cosine transform coefficients
instead of STFT coefficients. Neither has a published checkpoint. This is a boundary of what
`VocosModel` is rather than an omission from it: `VocosConfig` exposes one switch, `feature_extractor_type`,
both of whose values are implemented, and it raises on anything else. No backbone or head switch exists,
so there is no dead setting and no path that silently misbehaves. Should a checkpoint for either appear
later, its config would load through `PreTrainedConfig.from_dict` and then fail loudly on weight shapes
rather than quietly producing the wrong model.

**Dia2 loss term weights: unknowable, and the exposure is named.** The section 2.3 search is recorded in
`voicestudio/models/dia2/README.md` and came up empty everywhere: the 36 blobs and full 19 commit history
of the `nari-labs/dia2` tree hold no trainer, loss module, collator or eval script; both checkpoint repos
carry no `training_args.bin`, optimizer state, `trainer_state.json` or scheduler at any revision, their
safetensors headers no `__metadata__`, and their `config.json` no loss field at any revision; the Space
bundles the same inference package; arXiv, Hugging Face papers, Zenodo and the nari-labs blog return
nothing, and two open requests for a fine-tune recipe have no reply.

What that leaves is not the identity of the three terms, which the checkpoint and the decode loop fix. It
is how loudly each counts, and the sibling models here do not agree. Dia2 computes the depth term as one
pooled cross-entropy over all 31 remaining codebooks, which is `CsmForConditionalGeneration`'s convention
and matches the lineage Dia2 already follows down to CSM's frame-dropping hook.
`HiggsAudioV2ForConditionalGeneration` instead sums one mean per codebook, which on 31 codebooks makes the
acoustic term roughly 31 times heavier. That factor of 31 is the whole exposure: it would not crash, would
not produce a NaN and would not show in a gradient norm, and would surface only as a fine-tune that drifts
in audio detail or loses word timing. So the equal-weight sum is a defensible inherited-lineage choice, not
a fact about Dia2, and the README says so rather than calling it upstream's objective.

## Open: one item that needs a decision, not code

**Settled: the text normalizer reaches `wetext` through an optional extra.** The user granted an H11
exception for it, conditional on the measurement, and the measurement carried it. `e0efb9b9`, `6ae69b3e`,
`040bed9a` and `02413ef2` land `wetext>=0.1.7` as its own `frontend` extra, Apache-2.0, pulling `kaldifst`
and `contractions`, 35 MB of grammar on disk bundled in the wheel with no download at first use. It is in
neither `all` nor `research`, and with it absent `load_text_normalizer` returns `None` and every existing
path behaves exactly as before, so the default is unchanged.

Chinese is what justified it, not English. On CosyVoice v2 under `openai/whisper-large-v3-turbo`, three
seeds, character error rate against the original sentence: dates 0.545/0.545/1.000 to 0.000/0.045/0.182,
currency 0.444/2.056/0.722 to 0.000 three times, phone numbers 0.522/0.609/0.391 to 0.000/0.000/0.087.
Without it v2 reads a Chinese date back as `会议定在疫情意识五年三个支指` and one seed collapses to `好`.
v3 is the outlier: it already reads unnormalized Chinese dates and phone numbers back perfectly and only
currency improves. The digit-free control is identical seed for seed on all three models, which is what
rules out collateral damage.

`transformers`' own `EnglishNormalizer` from `models/clvp/number_normalizer.py` was the preferred route
under section 9.1 rule 1, measured, and rejected on content rather than on score. Its abbreviation table
maps `ft.` to `fort`, so `5 ft.` transcribes as `FIVE FORT` where `wetext` gives `FIVE FEET`; it also maps
`mrs.` to `misess`, `st.` to `saint` inside street names, and every ordinal wrongly, `1st` to `onest` and
`21st` to `twenty-onest`. It posted the best `units` figures while reading the wrong words.

Two things about it are worth knowing rather than discovering later. `wetext` reads `1234` as "twelve
thirty four" where this repository's own `number_to_words` reads it in full, and it drops the "one" from
`$1,234.50`; that is upstream's behaviour reproduced rather than corrected, it is invisible in a word error
rate scored against what each setting produced, and the better reading stays available by leaving the extra
out. And `wetext` raises `AssertionError` on 45 of v3's 276 markup tokens on the English branch while
silently rewriting those same 45 on the Chinese branch, `[AA1]` to `[AA一]`; `6ae69b3e` extends the markup
skip across both branches so all 276 survive. Upstream carries both defects.

Ordering, since two components now expand numbers: on the English path `wetext` runs first and
`number_to_words` reads only the digit runs it leaves behind, which is upstream's own ordering, and
`I paid 1234 dollars in 2025 for 7 books.` reaches `spell_out_number` with no digits left. On the Chinese
path `number_to_words` is never reached, sitting in the `else` of `contains_chinese`. The 41,821-string
`inflect` parity was re-run after the change at zero mismatches.

**Still open: `ttsfrd`.** It is the closed-source Alibaba wheel whose rules ship as a separate
`CosyVoice-ttsfrd` resource pack, and it was deliberately not touched even under the H11 exception. The one
concrete case naming it is `Dept.`, which is in neither `wetext`'s tables nor CLVP's, so
`Dr. Smith works at the U.S. Dept. of Energy.` is still not read correctly by anything available here.

One correction to the record it replaces: the old entry said the frontend emits the ARPAbet and pinyin
markup the 278 tokens exist for. Nothing in the open upstream source emits it. `ttsfrd` is its only
producer, and upstream's own example writes it inline as a caller-supplied pronunciation hotfix.

**CosyVoice v3's `llm.rl.pt`.** A second full checkpoint, not a delta: 293 tensors whose key set is exactly
`llm.pt`'s, no optimizer state or metadata, loading cleanly into a stock `Qwen2ForCausalLM`. Zero tensors
are bit-identical and 0.999999 of its 642,283,136 parameters differ, at overall relative L2 0.657, with
every embedding and head moved and `cos(a - base, b - base)` of 0.07 to 0.37 ruling out a small aligned RL
delta. Nothing upstream loads it: `grep -rn "llm\.rl\|rl\.pt"` is empty and the only path is a hardcoded
`llm.pt`.

**Decided: the base checkpoint stays the default and the RL one is not wired in.** Both are intended
releases, the roadmap listing "base model, rl model and its training/inference script", and the model
card benchmarks both across six measures. The trade is systematic rather than noise, better on all three
accuracy measures and worse on all three similarity measures: Chinese CER 1.21 to 0.81, English WER 2.24
to 1.68, hard-set CER 6.71 to 5.44, against Chinese speaker similarity 78.0 to 77.4, English 71.8 to 69.5
and hard-set 75.8 to 75.0. The largest single move is the 2.3 point drop in English speaker similarity,
which matters most for the voice cloning this model is usually reached for. Upstream hardcodes `llm.pt`
and its card recommends neither, so following the default is also following upstream. Nothing is
implemented for the RL checkpoint; this entry is the record of why.

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
