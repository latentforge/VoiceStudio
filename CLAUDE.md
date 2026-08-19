# VoiceStudio Agent Guidance

This file governs how coding agents (Claude Code and others) operate in this repository.
See `PROJECT.md` for the current migration plan and status; keep that file updated as
work progresses instead of tracking migration state only in conversation.

## Commit rules

- Do not add Claude, or any AI agent, as a co-author on commits. No
  `Co-Authored-By: Claude ...` trailers.
- Do not use em dashes anywhere in commit messages, code comments, or generated
  documentation. Use a comma, a period, or restructure the sentence instead.

## Model code conventions

- Never write a submodule (attention block, FFN, normalization, encoder layer, ...) by
  matching it to a vague architecture category ("this is a conformer", "this looks like a
  U-Net", "this is roughly a diffusion transformer") and substituting a similarly-labeled
  existing class. Open the actual upstream source file for that submodule and trace its
  class definition and forward method line by line: exact attention projections (fused
  qkv vs separate q/k/v/out), exact FFN shape (single vs macaron/double, GEGLU vs GELU vs
  SwiGLU), presence or absence of extra submodules (depthwise conv, gating, adapters),
  and the exact normalization/conditioning scheme (plain LayerNorm vs AdaLN, pre-norm vs
  post-norm). Two components with the same one-line description can have materially
  different internals; only a line-by-line reading of the real source catches that.
  Loading real pretrained weights afterward (clean `from_pretrained` LOAD REPORT, no
  MISSING/UNEXPECTED keys) is a confirmation step for this, not a substitute for it:
  passing a dummy-tensor forward/backward smoke test proves nothing about architectural
  correctness and must never be reported or treated as verification.
- Every migrated model must be trainable, not inference-only. Its top-level
  `*ForConditionalGeneration`/`*ForCausalLM` `forward()` must accept `labels` and return
  a cross-entropy loss (the standard `transformers` pattern: `ModelOutput` with a `loss`
  field, computed the same way the model it inherits from computes it). A model whose
  `forward()` only supports `generate()`/inference is not a valid migration.
- Follow `transformers` model file conventions: `modeling_<model>.py`,
  `configuration_<model>.py`, standard class inheritance from existing `transformers`
  base classes. Do not add source files that fall outside this layout.
- Before implementing a model from scratch, find the closest existing model lineage in
  `transformers` and inherit from it. Full from-scratch model implementations are only a
  last resort.
- Where a model already ships in `transformers` itself, only add an import relay, not a
  reimplementation.
- Use the `transformers` "Copied from ..." and `modular_<model>.py` mechanisms to avoid
  duplicating code between model files, the same way `transformers` itself does. Do not
  hand-edit a generated file behind a `modular_<model>.py` source, and do not edit inside
  a `# Copied from ...` block; edit the source it copies from instead.
- Comments follow `transformers` style: short, technical, explaining non-obvious
  runtime behavior of the code that is there right now (an invariant, a workaround for a
  specific bug, a constraint the reader could not otherwise infer). A comment must never
  explain what changed, why a change was made, what used to be there, what alternative
  was rejected, or reference the task/migration/PR that produced the code (no "instead
  of X", "previously did Y", "not needed here", "this replaces Z", "see PROJECT.md for
  why"). That kind of information belongs in the commit message, never in the file. If
  a line only makes sense as a note to whoever is reading the diff, delete it; write no
  comment on that line at all rather than a softened version of it. Docstrings describe
  what a function/class does and its parameters, never the history of how it got that
  way.
- Target `transformers` 5.0 conventions for anything newly written.
- Checkpoint conversions go through `WeightConvert`.
- Preprocessing is processor-only: every model exposes a single `Processor` combining
  tokenizer and audio_tokenizer behavior. Do not add a separate manual preprocessing
  step outside the processor.
- The `transformers` dependency in `pyproject.toml` points at the
  `latentforge/transformers-tts` fork, not upstream `transformers`.
- Flash attention support goes through the `kernels` package, not vendored/prebuilt
  `flash-attn` wheels.

## Licensing

Only `modeling_<model>.py` carries the original repository's license header, formatted
the way `transformers` formats its license headers. `configuration_<model>.py`,
`processing_<model>.py`, `tokenization_<model>.py`, `__init__.py`, and any other file in
a model's folder do not get a license header. Each model's folder also gets a
`README.md` linking back to the original code repository it was migrated from.

## Docstring format

Match the exact docstring shape `transformers` itself uses, not a paraphrase of it:

- Module docstring: one line, e.g. `"""Processor class for Qwen3-TTS."""` /
  `"""Configuration class for Qwen3-TTS."""`. No prose paragraphs at module level.
- Class docstring: `r"""` block starting with "Constructs a ..." / "This is the
  configuration class to store the configuration of a ...", followed by an `Args:`
  section documenting `__init__` parameters, each as `` name (`type`, *optional*):
  `` on its own line with the description indented below it. Cross-reference other
  classes/methods with `` [`ClassName`] `` / `` [`~ClassName.method`] ``.
- Method docstring: `Args:`, `Returns:`, and `Raises:` sections in that shape, not a
  single descriptive sentence.

Use `AGENTS.md` in the `transformers-tts` checkout and its `modeling_*.py`/
`processing_*.py`/`configuration_*.py` files as the reference for exact formatting.

## Import relay files

A file that only re-exports names from `transformers` (an import relay, used when a
model already ships in `transformers` itself) has no module docstring. Do not write a
line like `"""Import relay: ..."""` or any other sentence stating that the file is a
relay, that the model ships elsewhere, or where the code lives. The imports and
`__all__` speak for themselves.

## Repository hygiene

- `dep/` is a staging area for vendored upstream source during a migration. A model's
  vendored copy is deleted from `dep/` once its migration into `voicestudio/models/` is
  complete and verified, not before.
- Do not reintroduce the namespace-package split (`voicestudio-<model>` packages). All
  model code lives in this repository.
- Do not touch `voicestudio/models/stable_ommivoice/`, `stable_parler_tts/`, or
  `stable_qwen3_tts/`. They are out of scope for this migration.

## Workspace and checkpoint downloads

- The repo checkout lives on `C:`, which has no room for model checkpoints. `dep/` and
  `ckpts/` are symlinked to `D:\VoiceWork\dep` and `D:\VoiceWork\ckpts`. If a git worktree
  is needed for isolated work on a model, create it under `D:\VoiceWork\worktree`, not on
  `C:`.
- Any model downloaded from Hugging Face for local testing must be fetched through
  `hf_xet`, and must land under `ckpts/`, never the default `~/.cache/huggingface`
  resolution path. Pass an explicit local download target rather than relying on cache
  defaults.
- `ckpts/` is scratch space. Nothing under it survives past the end of the migration;
  never treat it as a place to keep deliverables.

## Migration order

Work through models in this order: Qwen3-TTS first, then Parler-TTS, then the rest of
`PROJECT.md`'s repo list in any order.
