# VoiceStudio Agent Guidance

This file governs how coding agents (Claude Code and others) operate in this
repository. See `PROJECT.md` for the current migration plan and status; keep that
file updated as work progresses instead of tracking migration state only in
conversation.

---

## 0. Hard Rules — Read First

These are non-negotiable. If a task would require breaking one of these, stop and
ask instead of proceeding.

| # | Rule |
|---|------|
| H1 | Never add `Co-Authored-By: Claude ...` or any AI co-author trailer to a commit. |
| H2 | Never use an em dash anywhere in commit messages, code comments, or generated docs. |
| H3 | Never write a submodule by architecture-category guessing. Always trace the real upstream source line by line first (§3.2). |
| H4 | Never treat a clean `from_pretrained` load or a dummy-tensor forward/backward pass as proof of architectural correctness. It is a confirmation step only. |
| H5 | Every migrated top-level `*ForConditionalGeneration`/`*ForCausalLM` must support training: `forward()` accepts `labels`, returns a `loss` via the standard `transformers` `ModelOutput` pattern. Inference-only is not a valid migration. |
| H6 | Never conclude "no public checkpoint exists" without the exhaustive search in §3.3, and never silently omit a real submodule found during a source-trace. Log gaps in `PROJECT.md`; do not resolve them unilaterally (§3.6). |
| H7 | Never hand-edit a generated file behind `modular_<model>.py`, and never edit inside a `# Copied from ...` block. Edit the source it copies from. |
| H8 | Comments must never narrate history, diffs, or rationale ("instead of X", "previously did Y", "see PROJECT.md"). See §6.1 for what's allowed. |
| H9 | Every source file in a model folder gets a license header. Import-relay files get no module docstring at all (§8). |
| H10 | Never write a migration as new files added beside an untouched vendored tree, and never delete that tree to make room. `git mv` the real upstream file and edit it in place (§2.4). |
| H11 | Never add a third-party dependency to make a migration work, and never `pip install` one into the environment to get past a blocker. Removing the upstream model's dependencies is part of the migration; if one cannot be removed, report it and let a human decide (§9.1). |
| H12 | Never rewrite shared history. No `git reset`, `rebase`, `commit --amend`, or force push, and never move the branch. Undo your own work with `git revert` only, and never touch a commit you did not create. Other agents commit to this branch concurrently (§1.3). |
| H13 | Never report a model verified without transcribing its generated audio back. A waveform with plausible amplitude and no NaN is not evidence (§2.5). |
| H14 | Never verify on the local GPU. Real-checkpoint runs go through the `colab` CLI (§2.5). |
| H15 | Every git command carries an explicit pathspec, and never create a git worktree (§1.4). |
| H16 | Model weights go under `.cache/` (`HF_HOME`), never `ckpts/` or any other ad hoc path. |

---

## 1. Commit Conventions

### 1.1 Prohibited content
- No AI co-author trailers (H1).
- No em dashes (H2). Use a comma, a period, or restructure the sentence.

### 1.2 Subject line format
`<Type>: <Short Title Case Description>`

- `Type` ∈ `Feat` / `Fix` / `Chore` / `Refactor` / `Docs` / `Style` / `Test` / `Merge`,
  capitalized, followed by `: `.
- Subject itself: a few words, Title Case, not a full sentence.
- Match existing repo history style, e.g. `Feat: Add Higgs Tokenizer`, `Fix: Device`,
  `Chore: Update Gitignore`, `Refactor: Prepare For Merging`, `Docs: Update Readme`,
  `Merge: Parler TTS`.
- A longer body explaining what changed and why is fine below the subject line when
  the change needs it; the format constraint applies to the subject line only.

### 1.3 Working on a shared branch

Several agents commit to this branch at the same time, so treat its history as
append-only. Undo your own work with `git revert`, which adds a commit; never with
`git reset`, `rebase`, `commit --amend`, or a force push, all of which drop other
people's commits along with yours. Do not move the branch, and do not check content
out of another branch into this one.

This is not hypothetical. A `git reset` meant to undo one agent's own migration
silently discarded four commits from three other authors, including a CLAUDE.md
update and a revert that had removed code from an abandoned branch, which the reset
brought back.

---

### 1.4 Sharing a working tree

Agents run concurrently in one checkout and one git index, so every git command
carries an explicit pathspec naming only your own folder:
`git commit voicestudio/models/<folder>/ -m "..."`. A bare `git commit` or
`git add -A` commits whatever else is staged. That has happened here: one agent's
commit swallowed 152 file deletions belonging to two others, which cost nothing but
made the commit boundary a lie.

Do not reach for a git worktree to get isolation. This harness bases a new worktree on
`main`, which is far behind `develop`, and agents that tried it either worked against
stale code or spent their run merging.

---

## 2. Model Migration Workflow

Follow these steps **in order** for every new model migration. Do not skip ahead to
implementation before completing the source-trace and checkpoint search.

### 2.1 Step 1 — Find the closest lineage
Before implementing anything from scratch, find the closest existing model lineage
in `transformers` and inherit from it. A full from-scratch implementation is a last
resort, not a default.

If a model already ships in `transformers` itself, add an import relay only (§8) —
never a reimplementation.

Inheritance is not limited to `transformers`. When a model already migrated under
`voicestudio/models/` is structurally close, inherit from that too rather than
writing a parallel copy. Models sharing a codec family are the usual case: several
models here decode Mimi codebooks, and several are multi-codebook codec LMs with
the same backbone-plus-depth-decoder shape. Check the sibling folders before
writing a class, and say in the migration report which lineage was chosen and why
the alternatives were rejected.

Inherit layers, compose models. A codec or vocoder that is independently published
with its own checkpoint is its own model folder, and the models that use it hold it as
a composed sub-model or as the processor's `audio_tokenizer`, never as a base class.
`transformers-tts` does the same, pairing `qwen3_tts` with
`qwen3_tts_tokenizer_multi_codebook` and `higgs_audio_v2` with
`higgs_audio_v2_tokenizer`, and `vocos`, `bigvgan` and `spark_tts_bicodec` follow it
here. Inheriting a *layer* across folders is the opposite case and is encouraged, the
way `spark_tts_bicodec` inherits DAC's `Snake1d` and `DacResidualUnit` and xcodec2's
`Xcodec2FiniteScalarQuantization`.

Never inherit a general model from a specific one's copy of it. Qwen2.5-Omni carries
its own BigVGAN, but a `bigvgan` model that subclassed it would make the general case
depend on one consumer. Trace the original author's source instead.

`PROJECT.md` carries a measured map of which classes appear in more than one folder.
Check it before writing a class, and if you reimplement something on that list, say in
the migration report why inheriting was rejected.

### 2.2 Step 2 — Trace the real upstream source, line by line
Never match a submodule (attention block, FFN, normalization, encoder layer, ...) to
a vague architecture category ("this is a conformer", "this looks like a U-Net",
"this is roughly a diffusion transformer") and substitute a similarly-labeled
existing class.

Instead, open the actual upstream source file for that submodule and trace its class
definition and `forward` method line by line, checking specifically:
- Exact attention projections (fused qkv vs. separate q/k/v/out)
- Exact FFN shape (single vs. macaron/double, GEGLU vs. GELU vs. SwiGLU)
- Presence or absence of extra submodules (depthwise conv, gating, adapters)
- Exact normalization/conditioning scheme (plain LayerNorm vs. AdaLN, pre-norm vs.
  post-norm)

Two components with the same one-line description can have materially different
internals. Only a line-by-line reading of the real source catches that.

### 2.3 Step 3 — Exhaustive checkpoint search
Never conclude "no public checkpoint exists" without checking **all** of:
- The Hugging Face model hub
- The upstream GitHub repo's README/releases
- Hugging Face Spaces (a demo Space frequently bundles real weights directly in its
  own repo even when no separate model repo exists)
- Zenodo
- The paper's own resources/appendix section

Record exactly what was checked and where it came up empty in `PROJECT.md` — not
just the negative conclusion. A wrong "no checkpoint" conclusion is not harmless: it
licenses skipping real-weight verification and can lead to silently simplifying or
omitting submodules on the mistaken belief nothing will ever catch the divergence.

### 2.4 Step 4 — Implement
- Inherit from the lineage found in Step 1.
- Follow the trainability requirement (§4).
- Follow file/module conventions (§5).

Implement by **transforming the vendored upstream source in place**, not by writing
new files next to it. Every model folder here already contains the real upstream
code, merged in with its git history; that code is the starting material, not a
reference to read once and set aside.

Concretely: `git mv` the upstream file that holds a component to its
`<kind>_<model>.py` name, then edit it into shape. Do not author a fresh
`modeling_<model>.py` alongside an untouched upstream tree, and do not delete the
upstream tree to make room for one. A migration whose diff is "N files added, M
files deleted" is wrong on its face; the diff should show the upstream files being
renamed and modified.

Two reasons this is a rule and not a preference. History: a rename keeps
`git log --follow` working back into the original author's commits, which is the
whole point of having merged that history. Fidelity: editing the real code makes
divergence from it visible in the diff, while writing alongside makes it invisible,
which is exactly how §2.7 happened.

### 2.5 Step 5 — Verify
A clean `from_pretrained` LOAD REPORT (no MISSING/UNEXPECTED keys) with real
pretrained weights is a **confirmation step for Step 2**, not a substitute for it.

A dummy-tensor forward/backward smoke test proves nothing about architectural
correctness and must never be reported or treated as verification.

The standard is intelligibility, not signal. Generate speech from a real checkpoint,
transcribe it back with wav2vec2 or Whisper, and report that transcript against the
prompt you asked for. Waveform RMS, peak amplitude and absence of NaN are not
evidence: a truncated half-second generation and a model ignoring its script both
pass every one of them. Two real cases here were caught only by transcription, a
`max_length` fallback that cut a sentence in half and a script below the model's
documented length floor that produced fluent unrelated speech.

Transcription has its own blind spot, so reach past it when the failure mode is not
lexical. Pairing a checkpoint with the wrong vocoder transcribed word for word while
the waveform was 37 times too quiet; level, spectrum and a copy-synthesis log mel
distance caught it. Where a numeric check against upstream's own classes is possible,
that is the strongest evidence available, and several models here match upstream to
1e-07 or bit for bit.

Verification does not run on the local GPU. Use the `colab` CLI for real-checkpoint
runs. `colab exec` can time out client-side while the remote script keeps running, so
poll for the result rather than reading a timeout as a failure.

### 2.6 Handling gaps found during the trace
Reading the real upstream source correctly is not the same as deciding it's fine to
skip part of it. If the source-trace finds a submodule or training-time mechanism
(an MDN, a reference encoder, a diffusion decoder, sample masking,
auto-transcription, anything) that the migrated code does not implement:

- That is a **scope decision**, not a finding — it does not get resolved by the same
  pass that found it.
- Do not land a commit or `PROJECT.md` status update that quietly omits a known real
  submodule on a self-supplied justification ("no checkpoint exists to verify it
  anyway", "out of scope for now", "this is a simplification").
- Record the gap in `PROJECT.md` exactly as found, with the omission and its
  justification called out as **still open**, and let a human decide whether it's
  acceptable.

### 2.7 Why this workflow exists: PromptTTS++
This process was learned from a real failure. A source-trace pass on PromptTTS++
correctly identified that the real model uses an MDN, a GST reference encoder, and a
`GaussianDiffusion` decoder in place of the migrated `FastSpeech2Conformer` path —
then unilaterally decided to leave the gap in place, believing no checkpoint existed
to check against. That belief was wrong: the checkpoint was bundled inside the
model's Hugging Face Space, and this was never independently verified before the
decision was made.

---

## 3. Trainability Requirement

Every migrated model must be trainable, not inference-only. Its top-level
`*ForConditionalGeneration`/`*ForCausalLM` `forward()` must:
- Accept `labels`
- Return a cross-entropy loss, computed the same way the model it inherits from
  computes it, via the standard `transformers` `ModelOutput` pattern with a `loss`
  field

A `forward()` that only supports `generate()`/inference is not a valid migration.

Derive the loss from the upstream repo's own training and evaluation code, not from
guesswork about what a model of this shape usually optimizes. Nearly every vendored
repo here ships a trainer, a loss module, a data collator or an eval script, and
those files are the authority on what the real objective is: which terms it sums,
how each is weighted or masked, which inputs are teacher-forced, what is frozen,
and which targets come from the codec rather than the raw waveform. Read them the
same way §2.2 requires the modeling code to be read.

Two of those details are the usual source of silent error, so state both explicitly
in the migration report: what the upstream loss actually computes term by term, and
which modules upstream freezes during training. A migration that runs `backward()`
without crashing but optimizes a different objective than upstream is a failed
migration, and nothing about a finite loss or a nonzero gradient norm will reveal
it.

---

## 4. File & Module Organization

### 4.1 Naming convention
Per-model source files are named `<kind>_<model>.py`, following `transformers` model
file conventions (standard class inheritance from existing `transformers` base
classes).

### 4.2 Determining which `<kind>` prefixes a model needs
The set of `<kind>` prefixes a given model needs is whatever the real
`transformers`/`transformers-tts` convention actually uses for a model with that
shape — **not** a fixed shortlist.

Besides `modeling_` / `configuration_`, real examples already present in
`transformers-tts` include:
- `generation_` (e.g. `csm`, `dia`, `higgs_audio_v2`, `qwen3_tts`, `whisper` — for a
  model with a custom `GenerationMixin` override worth splitting out)
- `processing_`
- `tokenization_`
- `image_processing_`
- `feature_extraction_`
- `modular_`

**Before** assuming a migrated model must be squeezed into a smaller file set than
its real upstream source used, check `transformers-tts` for the closest precedent
(`grep` its `src/transformers/models/` tree) rather than inferring the allowed set
from this document's examples.

- Do not invent a `<kind>` prefix with no precedent in `transformers-tts`.
- Do not merge multiple real upstream files together (e.g. folding a model's own
  `generation_<model>.py` into `modeling_<model>.py`) just because this document's
  examples didn't happen to name that file.

### 4.4 Claiming a convention

Before asserting that `transformers` does or does not do something, `grep` its
`models/` tree for the pattern and cite what you found. Do not generalize from one
subsystem, and do not infer the allowed set from this document's examples.

This is a rule because the inference is wrong often enough to matter. `loss_utils.py`
holds only functions, which reads as "losses are functions here", but seven models
including `fastspeech2_conformer` hold a structured loss as an `nn.Module` attribute
named `self.criterion`, defined in their own `modeling_*.py`. A migration was nearly
changed to match a convention that does not exist.

### 4.3 Copied-from / modular mechanism
Use the `transformers` "Copied from ..." mechanism to avoid duplicating code between
model files, the same way `transformers` itself does.

- Do not hand-edit a generated file behind a `modular_<model>.py` source.
- Do not edit inside a `# Copied from ...` block, edit the source it copies from.

`modular_<model>.py` itself does not apply to this repository, and its absence here is
not a gap. Inside `transformers`, the modular file is where inheritance happens and
`modeling_<model>.py` is generated from it with everything flattened, which is why
`modular_qwen3.py` declares `Qwen3Attention(LlamaAttention)` while the generated
`modeling_qwen3.py` inherits across no model at all. A project that consumes
`transformers` as a library has no generation step, so its `modeling_<model>.py` plays
the modular role and inherits directly. That is what `cosyvoice_v3` does from `f5_tts`
and `spark_tts_bicodec` from `xcodec2`, and it is correct.

---

## 5. Code Style

### 5.1 Comments
Comments follow `transformers` style: short, technical, explaining **non-obvious
runtime behavior of the code that is there right now** — an invariant, a workaround
for a specific bug, a constraint the reader could not otherwise infer.

A comment must **never**:
- Explain what changed
- Explain why a change was made
- Describe what used to be there
- Reference an alternative that was rejected
- Reference the task/migration/PR that produced the code

No `"instead of X"`, `"previously did Y"`, `"not needed here"`, `"this replaces Z"`,
`"see PROJECT.md for why"`.

That kind of information belongs in the commit message, never in the file. If a line
only makes sense as a note to whoever is reading the diff, delete it — write no
comment on that line at all rather than a softened version of it.

Docstrings describe what a function/class does and its parameters, never the history
of how it got that way.

### 5.2 Docstring format
Match the exact docstring shape `transformers` itself uses, not a paraphrase of it.

**Module docstring** — one line only, no prose paragraphs:
```python
"""Processor class for Qwen3-TTS."""
"""Configuration class for Qwen3-TTS."""
```

**Class docstring** — `r"""` block starting with "Constructs a ..." / "This is the
configuration class to store the configuration of a ...", followed by an `Args:`
section documenting `__init__` parameters, each as:
```
name (`type`, *optional*):
    Description indented below it.
```
Cross-reference other classes/methods with `` [`ClassName`] `` / ``
[`~ClassName.method`] ``.

**Method docstring** — `Args:`, `Returns:`, and `Raises:` sections in that shape, not
a single descriptive sentence.

Reference: use `AGENTS.md` in the `transformers-tts` checkout and its
`modeling_*.py` / `processing_*.py` / `configuration_*.py` files as the exact
formatting reference.

---

## 6. Licensing & Headers

- Every source file in a model's folder carries the original repository's license
  header, formatted the way `transformers` formats its own. That is what
  `transformers` does: `configuration_llama.py`, `configuration_qwen3.py` and
  `configuration_dac.py` all open with one, not only their modeling files.
- No per-model `LICENSE` or `INFO.md` file. The license lives in the modeling header
  and one `LICENSE` sits at the repository root, which is how `transformers` does it.
  An upstream `README.md` carried in during vendoring is replaced, not kept alongside.
- Each model's folder gets a `README.md` in the format of
  `voicestudio/models/higgs_tts2/README.md`: the model name, a paragraph on what it
  does architecturally, then a line reading
  `Original model and code: [owner/repo](https://github.com/owner/repo)` with the
  repository hyperlinked, then `## Usage`. Where a source-trace found something the
  migration does not implement, the README also carries a
  `## Not carried over from upstream` section listing it.

---

## 7. Import Relay Files

A file that only re-exports names from `transformers` (used when a model already
ships in `transformers` itself) gets **no module docstring at all**.

Do not write a line like `"""Import relay: ..."""` or any other sentence stating
that the file is a relay, that the model ships elsewhere, or where the code lives.
The imports and `__all__` speak for themselves.

---

## 8. Preprocessing

Preprocessing is processor-only: every model exposes a single `Processor` combining
tokenizer and audio_tokenizer behavior. Do not add a separate manual preprocessing
step outside the processor.

---

## 9. Dependencies & Infra

- Target `transformers` 5.0 conventions for anything newly written.
- Checkpoint conversions go through `WeightConvert`.
- The `transformers` dependency in `pyproject.toml` points at
  `latentforge/transformers-tts`, not upstream `transformers`. Refer to it by name;
  it is not "the fork".
- Flash attention support goes through the `kernels` package, not
  vendored/prebuilt `flash-attn` wheels.

### 9.1 Removing a model's external dependencies

A migration removes the upstream model's third-party dependencies; it never adds
them. If the vendored source imports a library to do its work, the migration's job
is to end that import, in this order of preference:

1. **Use what `transformers-tts` already ships.** Codecs, vocoders and encoders are
   usually already there (`dac`, `encodec`, `mimi`, `xcodec`, `xcodec2`, `hubert`,
   `wav2vec2`, and others). An upstream import of a standalone codec package is
   almost always replaceable by the native class.
2. **Inline the part actually used.** Where the dependency supplies a small piece of
   math rather than a model, port that piece into the model file. A fixed-step ODE
   solver, a beta schedule, or a base class the upstream inherits for a handful of
   methods are all a few dozen lines, and inlining them costs less than carrying a
   package.
3. **Ask before keeping one.** If a dependency genuinely cannot be removed, do not
   add it to `pyproject.toml` on your own judgement. Report what it is, what needs
   it, and what removing it would cost, and let a human decide.

Never `pip install` a package into the environment to make a migration work without
raising it first. Discovering mid-task that a model wants a new library is a finding
to report, not a step to take.
