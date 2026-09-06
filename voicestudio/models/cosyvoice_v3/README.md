# CosyVoice v3

CosyVoice v3 keeps v2's three network layout and replaces two of the three. A Qwen2 0.5B decoder
turns text into 6561 supervised semantic speech tokens at 25 Hz. A conditional flow matching model
turns those tokens into an 80 bin mel spectrogram at 24 kHz, and it has **no encoder at all**: a
lookahead convolution runs over the embedded tokens, a repeat interleave carries them to the mel
frame rate, and a diffusion transformer predicts the vector field. A causal HiFTNet vocoder turns the
mel spectrogram into a waveform, padding every convolution on one side only so a chunk can be
rendered before the frames that follow it exist.

Against v2 the language model moves its start of sequence, end of speech and task vectors out of a
separate two entry table and into the speech token table, which grows from `6561 + 3` to
`6561 + 200`. Because that is the only change to how a sequence is built, every packing routine and
the decode loop are inherited from v2 rather than copied.

The three networks are trained separately upstream and are released as three separate `.pt` files.

Original model and code: https://github.com/FunAudioLLM/CosyVoice

## Usage

```python
from voicestudio.models.cosyvoice_v3 import CosyVoiceV3ForConditionalGeneration, CosyVoiceV3Processor

model = CosyVoiceV3ForConditionalGeneration.from_pretrained("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
processor = CosyVoiceV3Processor.from_pretrained("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")

inputs = processor(text="<|endofprompt|>The quick brown fox jumps over the lazy dog.")

waveform = model.generate(input_ids=inputs.input_ids, speaker_embedding=speaker_embedding)
```

v3 requires the end of prompt token, id 151646, to appear in the text or the prompt text, and
`generate` raises if it does not. Upstream uses it to separate an instruction prefix from the text to
synthesize, as in `"用四川话说<|endofprompt|>扁担长..."`, so an empty instruction leaves the marker at
the front. The same conditioning modes as v2 are available, and passing a speaker embedding alone is
upstream's sft mode, the only one that needs no reference waveform.

The released directory holds one `.pt` file per network rather than a single checkpoint, beside the
`CosyVoice-BlankEN` directory the language model is built from. `from_pretrained` reads that layout
directly: it merges the three files under the name of the submodule each belongs to into a directory
under `HF_HOME`, keyed on the repository and the commit it resolved to, and the `WeightRenaming` rules
registered in `modeling_cosyvoice_v3.py` turn upstream's module names into this model's as that
directory loads. Later loads reuse it, and resolve nothing but the `cosyvoice3.yaml` and the
`CosyVoice-BlankEN/config.json` that name the revision. Once the merge is written, the three `.pt`
files are dropped from the `huggingface_hub` cache. The processor takes the same repository id and
picks up the text tokenizer, the speech tokenizer and the speaker encoder.

## The text frontend

`CosyVoiceV3Processor.normalize_text` is upstream's `CosyVoiceFrontEnd.text_normalize`, and it runs
before the tokenizer. Both branches open with the `wetext` text normalizer, which the `frontend`
extra installs and which is skipped when it is absent; see "The text normalizer". A Chinese sentence
then loses the spaces that do not sit inside an embedded English word, has its corner marks spelled
out, its brackets removed, its full stops and dashes replaced by their Chinese counterparts and a
trailing run of commas turned into a full stop. Any other sentence has its remaining digit runs read
out in English. Either way the result is split into the pieces upstream
synthesizes one at a time, on punctuation, with each piece grown to at most 80 units and a trailing
piece shorter than 20 merged into the one before it, a Chinese piece measured in characters and any
other in tokens. `text_frontend=False` turns the whole thing off, which is what upstream passes to
reproduce the samples of its demonstration pages.

```python
pieces = processor.normalize_text("I paid 1234 dollars in 2025 for 7 books.")
# ['I paid one thousand, two hundred and thirty-four dollars in two thousand and twenty-five for seven books.']
```

One thing is v3's rather than v1's: the whole rewrite is skipped inside the markup of the added
vocabulary, on both branches, because `[AA1]` is one token whose trailing `1` is a stress mark rather
than a number. Without that, upstream's own English branch rewrites it to `[AAone]` and the token is
gone, and the text normalizer raises on it outright. The
reading itself is `number_to_words` in `voicestudio/models/cosyvoice_v1/`, inherited through v2. It
is upstream's `inflect` call inlined, so nothing has to be installed for it and v1, v2 and v3 read a
number the same way.

That markup is what the 278 added embedding rows are for. It is written inline by the caller, to
override a pronunciation, as in upstream's own `'...对报道[j][ǐ]予好评。'`, and `'[T][AH0][M][EY1][T][OW2]'`
for English. Nothing in the open upstream source emits it: the only producer is `ttsfrd`, which is
closed source, so a caller supplies it.

Text arrives as a `str`. Passing a generator of `input_ids` tensors to `generate` instead selects
upstream's `inference_bistream`, the interleaved decode that reads text as it arrives and emits
speech tokens between the groups, at the same `mix_ratio` `[5, 15]` the interleaved training layout
uses. v3 requires the end of prompt token in the prompt text there, since the text is not complete
when the sequence opens.

```python
def stream():
    for chunk in ["The quick ", "brown fox ", "jumps over ", "the lazy dog."]:
        yield processor(text=chunk).input_ids

waveform = model.generate(input_ids=stream(), speaker_embedding=speaker_embedding,
                          prompt_input_ids=prompt.prompt_input_ids,
                          prompt_speech_token_ids=prompt.prompt_speech_token_ids,
                          prompt_speech_feat=prompt.speech_feat)
```

Switching on `isinstance(input_ids, GeneratorType)` is what `transformers` itself does for a
streamed input: `nemotron_asr_streaming/generation_nemotron_asr_streaming.py` selects its streaming
path with `isinstance(input_features, GeneratorType)` in both `_prepare_model_inputs` and
`generate`, and `voxtral_realtime` does the same in four places. Those are the only two occurrences
of `GeneratorType` in `transformers`, and both are input side. The `BaseStreamer` family in
`generation/streamers.py` was rejected because it is an output side protocol, `put` and `end` called
by `generate` as it produces tokens, with no input side counterpart; a `TextIteratorStreamer` driving
this model is its consumer side, and reaches it as `(chunk for chunk in streamer)`.

## Training

`CosyVoiceV3ForConditionalGeneration.forward(labels=...)` returns the language model objective only,
because upstream trains the three networks one at a time. The other two objectives are on the
submodules.

- **Language model.** `LabelSmoothingLoss` over `speech_vocab_size + 200` classes with
  `smoothing=0.0` and `normalize_length=True`. The two sequence layouts are v2's unchanged, the
  unistream one and the interleaved bistream one at `mix_ratio` `[5, 15]`, and upstream draws between
  them per sample with probability one half when the speech to text ratio allows; `forward` takes
  that draw as its `bistream` argument so it stays deterministic. The only difference from v2 is that
  the control vectors come out of the speech token table.
- **Flow matching.** `CosyVoiceV3FlowModel.forward` returns
  `CosyVoiceV3ConditionalCFM.compute_loss`, which is v1's optimal transport conditional flow matching
  loss unchanged, with v2's unified streaming training on top: upstream draws `streaming` once per
  batch with probability one half and threads it into the estimator.
- **Vocoder.** `CosyVoiceV3HiFTGenerator.forward` returns the waveform and the predicted f0.
  `compute_loss`, inherited from v1, scores the two terms that need no discriminator, at the mel
  resolution `CosyVoiceV2Config` sets: `cosyvoice3.yaml` gives `mel_spec_transform1` the same
  `n_fft` 1920, `hop_size` 480 and `win_size` 1920 at 24 kHz that `cosyvoice2.yaml` does, so v3
  needs no override of its own. The rest of the objective is not implemented; see "Not carried over
  from upstream".

Upstream freezes nothing, for the same reason as v1 and v2: the only `requires_grad` write in the
tree is `Snake.alpha.requires_grad = alpha_trainable`, whose default is `True`, and the only
`.eval()` on a training path belongs to the direct preference optimization recipe's separate
reference model.

## Lineage

The base is `voicestudio/models/cosyvoice_v2/`, mirroring upstream, where `CosyVoice3LM` subclasses
`Qwen2LM` and `CosyVoice3Model` subclasses `CosyVoice2Model`. What is inherited and what is not:

| Class | Base | Why |
|---|---|---|
| `CosyVoiceV3Config` | `CosyVoiceV2Config` | Same flat field set with the v3 geometry, plus an `estimator_config` sub configuration. The flow encoder fields are inherited and unused, since v3 has no flow encoder. |
| `CosyVoiceV3SpeechTokenLM` | `CosyVoiceV2SpeechTokenLM` | `llm_embedding` becomes a read only property returning the speech token table, which is exactly what v3 does, so both training layouts and the decode loop are inherited rather than copied. Only the table widths and the head losing its bias are overridden. |
| `CosyVoiceV3ConditionalCFM` | `CosyVoiceV2ConditionalCFM` | Same objective, same guided Euler solver, same fixed initial noise; only the estimator changes. |
| `CosyVoiceV3ResBlock` | `CosyVoiceV1ResBlock` | Same residual algebra with left padded convolutions. |
| `CosyVoiceV3HiFTGenerator` | `CosyVoiceV1HiFTGenerator` | Same synthesis filter and inverse transform head, rebuilt from causal convolutions. |
| `CosyVoiceV3FeatureExtractor`, `CosyVoiceV3Processor` | the v2 pair | The mel geometry is identical; only the tokenizer's added vocabulary and the speech tokenizer file name change. |
| `CosyVoiceV3GenerationMixin` | `CosyVoiceV2GenerationMixin` | The language model loop is inherited. `token2wav` is written out, because v3 carries no vocoder cache and re-renders the whole accumulated mel each chunk. |

### The estimator comes from f5_tts, and this is where the sibling map paid off

PROJECT.md's sibling inheritance map named `FeedForward` and `TimestepEmbedding` against f5_tts and
gated the check on the CosyVoice migration. v1 rejected both with concrete reasons, because v1's
estimator is a Matcha one dimensional UNet. **v3 is the model the map was pointing at**, and the
rejection is reversed: `cosyvoice/flow/DiT/dit.py` and `modules.py` are the F5-TTS diffusion
transformer, and `F5TTSConfig`'s stock defaults are v3's geometry exactly, `hidden_size` 1024,
`num_hidden_layers` 22, `num_attention_heads` 16, `head_dim` 64, `ff_mult` 2, `dropout` 0.1.

Inherited unchanged: `F5TTSDecoderLayer`, `F5TTSAdaLayerNormFinal`, `F5TTSTimestepEmbedding`,
`F5TTSSinusPositionEmbedding` through it, `F5TTSAttention` and `F5TTSRotaryEmbedding`. The estimator
configuration is an actual `F5TTSConfig`, so the reuse is checkable rather than asserted.

Two classes could not be inherited. `CosyVoiceV3CausalConvPositionEmbedding` is left padded and holds
two separate convolution sequences where `F5TTSConvPositionEmbedding` is centre padded and holds one,
so the parameter paths differ. `CosyVoiceV3InputEmbedding` concatenates a speaker vector that f5_tts
has no equivalent of, in the order noised speech, conditioning speech, encoded tokens, speaker, which
is not f5_tts's order.

**`pe_attn_head=1` is the setting that makes this correct, and it is not obvious.** Upstream calls
x-transformers' `apply_rotary_pos_emb` on the **unreshaped** `(batch, length, 1024)` projection, and
that function rotates only the leading `freqs.shape[-1]` channels, which is `dim_head`, 64. After the
head reshape those 64 channels are the first attention head and nothing else, so the rotary reaches
one head of sixteen. Measured against upstream's own `DiT`, the estimator with the rotary on **every**
head differs by **3.356** where head 0 only differs by 4.702e-03, three orders apart, so the setting
does real work and matching at it is a result rather than a coincidence.

The bridge between the two rotary conventions is exact. x-transformers lays its frequencies out as
interleaved pairs and rotates with `rotate_half` over `(d r)`; llama uses the half split layout.
f5_tts's `deinterleave_head_dim` is precisely the permutation between them, and since query and key
both receive it the attention dot product is invariant. Measured: with the rotary applied by hand
both ways, the fifteen heads that receive none differ by **exactly 0.0** and head 0 by 7.324e-04 on
scores of magnitude 8323, a relative 8.8e-08, which is below one float32 unit in the last place.

Other entries on the sibling map are rejected for v3 for the same reasons as v1 and v2: `Encoder` and
`EncoderLayer` against prompt_tts_pp are moot, since v3 has no encoder; `SourceModule` against
prompt_tts_pp, because v3 uses upstream's interpolating causal generator, which is a third variant
again; and `Snake` against bigvgan, because `BigVGANSnakeActivation` unconditionally resamples
through a Kaiser windowed sinc and `anti_alias_ratio=1` is not an identity.

## The estimator's disagreement with upstream, characterised

The estimator is **not bit exact** against upstream and cannot be, and the reason is entirely in two
library behaviours rather than in the model. Both are places where `transformers` forces float32
inside an otherwise higher precision computation, which is invisible at float32 and visible above it.

- **`eager_attention_forward` computes the softmax in float32.** Its last lines are
  `nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)`. Upstream calls
  `F.scaled_dot_product_attention`, which uses the input dtype.
- **`LlamaRotaryEmbedding.forward` computes its frequencies in float32**, under an explicit autocast
  disable, and casts only the result. x-transformers computes them in whatever dtype its buffer
  carries.

Crossing the two in float64, where rounding is about 1e-16 and anything visible is a real difference
in operations:

| attention implementation | rotary | relative difference from upstream |
|---|---|---|
| eager | transformers | 7.710e-05 |
| eager | matched to upstream | 7.921e-05 |
| sdpa | transformers | 1.606e-06 |
| sdpa | matched to upstream | **1.570e-12** |

Five orders of collapse, with the eager softmax as the large term and the rotary as the small one.
The worst single layer falls from 2.678e-04 to 6.834e-15. Neither alone accounts for the difference,
which is why removing only the rotary looked like a failed hypothesis.

**None of this is visible at float32**, which is what the model is run at. Measured against a float64
ground truth, upstream's own float32 sits at 2.745e-04 relative and this implementation at
2.391e-04, an error ratio of **0.871**: neither is the wrong one, and if anything the migrated path
is marginally closer to the truth. The reason a 1e-07 seed reaches 1e-04 is amplification, roughly a
thousandfold through 22 residual layers whose activations grow from 333 to 1436.

There is no bit exactness claim here, and the right statement about the v3 estimator is a relative
agreement of that order.

### A trap worth knowing about

The estimator is a plain `nn.Module`, not a `PreTrainedModel`, so nothing runs the attention
implementation autoselection over its sub configuration and `_attn_implementation` stays `None`,
which dispatches to the eager path and its float32 softmax. `CosyVoiceV3Config` therefore sets it
explicitly. Anyone nesting a `transformers` sub configuration under a non `PreTrainedModel` container
has the same hazard.

## Upstream does not run correctly on transformers 5

Both defects v2 found reproduce on v3, since `CosyVoice3LM` inherits the methods that carry them.
Neither is an omission in this folder and neither is a choice this migration made.

- **`Qwen2Encoder.forward_one_step` mishandles its attention mask.** It passes a padding mask
  covering the current step alone while the cache already holds the prefix. Measured for v2 against
  an uncached full sequence reference, upstream's cached logits are wrong by 9.762 where this
  implementation is at 9.5e-06. On v3, from the same seed and the same text, stock upstream produces
  **52** tokens and this implementation **97**; with the one step mask dropped the two are
  **identical, 97 against 97**, and both end on a stop token. This model deliberately does not
  reproduce upstream's sampled output on transformers 5, because reproducing it would mean
  reproducing the defect.
- **`Qwen2Encoder.__init__` leaves the backbone dtype to `from_pretrained`.** Under upstream's pinned
  4.51.3 that gives float32; under transformers 5 it follows the checkpoint and gives bfloat16 while
  the sibling modules stay float32, so upstream's own module is internally mixed and raises. This
  model is uniformly float32, confirmed by survey.

## Deliberate deviations from upstream

Three, all of which reproduce the computation and change only a side effect or a source of
irreproducibility.

- **The 29 MB `uv` noise tensor is not built.** Upstream's `SourceModuleHnNSF` draws it and returns
  it, and the vocoder discards it with `s, _, _`. It never reaches the output, so not building it
  removes a meta device hazard by construction rather than by fixing it.
- **The fixed sine tensors are seeded.** Upstream draws the per harmonic phase offsets and a 259 MB
  noise tensor from whatever global random state is current when the module is built, which makes its
  v3 vocoder **irreproducible across processes**. This is a property of upstream, not a defect in the
  migration. Here they come from a private generator seeded with `source_noise_seed`, built lazily on
  first use and outside inference mode. One consequence has to be disclosed: a parity comparison
  cannot be made by seeding alone, so the harness points the migrated generator at upstream's two
  tensors, and every vocoder number below is measured under that substitution.
- **`inference` restores the f0 predictor's dtype.** Upstream casts it to float64 and leaves it cast,
  so calling `inference` permanently changes the module. Measured: after upstream's `inference` its
  predictor is float64 and its own training path `forward` raises
  `RuntimeError: Input type (float) and bias type (double) should be the same`; after the migrated
  `inference` the predictor is float32 and `forward` succeeds. The computation is identical, with the
  f0 contour at **exactly 0.0**.

## Verification

Everything below ran on CPU in float32 against the real
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` weights, in the project venv except the two runs that close
this section, which needed `onnxruntime` and ran on a Colab T4 through the `colab` CLI with the model
on CPU.

**Checkpoint coverage.** 951 source tensors convert to 949 over **859,185,455 parameters**,
506,148,480 under `llm`, 332,257,088 under `flow` and 20,779,887 under `hift`, with zero MISSING and
zero UNEXPECTED keys. Two tensors are dropped and each is accounted for rather than assumed:
`llm.model.lm_head.weight`, shown by a bit identity search to equal `llm.model.embed_tokens.weight`,
which is the tie the Qwen2 configuration declares; and
`decoder.estimator.rotary_embed.inv_freq`, x-transformers' persistent frequency buffer, which the
migrated estimator computes for itself. That second one has no bit identical counterpart, so it was
compared directly instead: **0.0 against the frequencies the migrated estimator computes, and 0.0
again on the cosines derived from each**.

Every one of those source tensors is bit identical to the model tensor it lands on, and no model
tensor is left unfilled, so the registered `WeightRenaming` rules account for the checkpoint exactly
rather than merely producing a clean report. `save_pretrained` followed by `from_pretrained` returns
the same tensors with zero missing, unexpected or mismatched keys, which is what keeps the reverse
of the mapping honest.

**Tokenizer.** Upstream's special token list takes the released tokenizer from 151646 to 151924
entries against 151936 embedding rows, so all 278 additions land inside the table with twelve rows to
spare and nothing is resized. Upstream adds the same list, with its own comment that the rows stay
randomly initialised. There is an independent check that the 281 entry transcription is correct in
both content and order: upstream hardcodes 151646 for `<|endofprompt|>` in an assertion, and that id
is only right if this exact list is added in this exact order to this exact tokenizer. It lands on
151646.

**Numeric parity against upstream's own classes.** Reported as `max|diff|` with the relative figure,
since the estimator's activations make an absolute number unreadable on its own:

| Component | Upstream class | difference |
|---|---|---|
| Rotary frequency buffer and its cosines | `RotaryEmbedding` | 0.0 |
| Lookahead layer, with and without context | `PreLookaheadLayer` | 0.0 |
| Fixed initial noise | `CausalConditionalCFM.rand_noise` | 0.0 |
| Flow estimator, full and chunked attention | `DiT` | 9.582e-04 and 9.327e-04 relative |
| Euler solver, full and chunked | `CausalConditionalCFM` | 7.178e-04 and 3.313e-04 relative |
| Flow matching loss, full and chunked | `CausalConditionalCFM.compute_loss` | 1.257e-05 and 1.198e-05 relative |
| Flow inference, all four streaming and finalize settings | `CausalMaskedDiffWithDiT.inference` | 1.024e-04 to 2.914e-04 relative |
| f0 predictor, both finalize settings | `CausalConvRNNF0Predictor` | 0.0 |
| Sine generator waves and voiced mask | `SineGen2` | 0.0 |
| Neural source filter | `SourceModuleHnNSF` | 0.0 |
| Vocoder excitation, both finalize settings | `CausalHiFTGenerator.inference` | 0.0 |
| Vocoder waveform, both finalize settings | `CausalHiFTGenerator.inference` | 1.149e-05 relative |
| Vocoder f0 and waveform, training entry point | `CausalHiFTGenerator.forward` | 0.0 and 1.475e-05 relative |
| Language model text embedding and logits | `Qwen2Encoder` | 0.0 |
| Language model loss, unistream and bistream | `CosyVoice3LM.forward` | 0.0 |

Everything that can be bit exact is. The vocoder's 1.1e-05 is the analysis window floor v1 pinned to
`scipy.signal.get_window("hann", 16, fftbins=True)` against `torch.hann_window(16, periodic=True)`,
and the excitation and f0 being exactly 0.0 places it at the short time Fourier transform and nowhere
earlier. The flow differences are the characterised estimator divergence: `flow inference` at
2.914e-04 relative sits on the 2.391e-04 and 2.745e-04 measured against float64 truth, and the ten
Euler steps **reduce** the estimator's own 9.58e-04 rather than compounding it.

**Why those differences are rounding and not bias.** The flow matching loss is a scalar reduction
over 80 by 120 elements. If the per element differences are independent and zero mean they should
average down by the square root of the count, and a systematic bias would carry through at close to
full size. The prediction and the measurement:

```
9.582e-04 / sqrt(9600) = 9.78e-06        measured 1.257e-05        ratio 1.29
```

Within a factor of 1.3. The lookahead layer at exactly 0.0 pins the divergence to the estimator and
nothing before it, and the fixed initial noise at 0.0 rules out the solver's starting point.

**The autoregressive loop** was checked as a loop. Against upstream with the one step mask corrected,
`CosyVoice3LM.inference` and `generate_speech_tokens` produce the **same 97 speech tokens** from the
same seed, and both end on a stop token rather than the length cap. Against stock upstream they
differ, 52 against 97; see the transformers 5 section. The silence thinning was checked against
upstream's rule on a scripted stream: 19 tokens in, 15 out, a nine token silent run cut to five and a
four token run left alone.

**Training objectives.** Both language model losses match upstream at 0.0 on both sequence layouts.
Gradients from that objective reach 292 parameter tensors, all under `llm`, none under `flow` or
`hift`.

**Round trip through the hub format.** This found a real defect, and it is worth reading as two
separate results rather than one.

*The defect.* `transformers`' rotary embedding computes its frequencies in its constructor and
registers them as two **non persistent** buffers, `inv_freq` and `original_inv_freq`. Under the meta
device initialisation transformers 5 uses, a non persistent buffer is materialised as uninitialised
memory rather than by rerunning the computation that produced it, and `is_meta` is `False`
afterwards, so checking for meta tensors does not catch it. Measured on a reloaded model, the two
buffers came back holding values of **1.77e21** and **5.24e22**, and flow inference differed from the
model it had been saved from by **2.744, a relative 2.5e-01**, which is a materially different output
rather than a precision difference.

The cause is structural and it generalises: `CosyVoiceV1PreTrainedModel._init_weights` enumerates its
own module types and stops, with no `else: super()._init_weights(module)` fallback, and v2 and v3
inherit it. **Any `_init_weights` in this repository that enumerates module types without falling
through to `super()` has the same hole.** It only bites when a module carrying constructor computed
non persistent buffers sits under a plain `nn.Module` rather than a nested `PreTrainedModel`, which
is why v2 is unaffected: its only rotary lives inside `Qwen2Model`, which initialises itself, while
v3's estimator rotary lives inside `CosyVoiceV3ConditionalDecoder`, a plain module reachable only
through `_init_weights`. `CosyVoiceV3PreTrainedModel._init_weights` now rebuilds both buffers from
the configuration, recomputing rather than copying and mirroring the constructor's own selection of
the initialiser including `attention_scaling`. A blanket fallback to `PreTrainedModel._init_weights`
was **not** added, because it would reach v1's implementation, which is inert for unknown types, and
making it reach the base implementation would re-initialise types v1 deliberately leaves alone.
v1 and v2 are one nested plain module away from the same defect and neither is currently reached by a
case that needs it; that is recorded for a human rather than changed here.

*The residual, after the fix.* Every parameter, every buffer including both rotary ones, every tensor
held as a plain attribute, the rotary cosines, the vocoder waveform and the sampled speech token
sequence are all **bit identical**. Flow inference is **not**: it agrees to `1.013e-03`, a relative
**9.314e-05**. This is weaker than v2's `0.000e+00` and should be read as weaker.

The cause is storage layout, demonstrated rather than argued. `from_pretrained` memory maps its
weights and a freshly built model allocates its own, and on layer 19's query projection both are
contiguous with the same stride, dtype and storage offset while the base pointers differ in
alignment, **`data_ptr mod 64` of 0 against 56**. Different alignment leads the matmul to a different
blocking and so a different accumulation order, at roughly float32 epsilon per operation, which the
same thousandfold amplification through 22 residual layers carries to 1e-04. Loading the reloaded
weights into a third, freshly allocated model as contiguous clones, changing the layout and nothing
else, reproduces the built model at **exactly 0.0**, while the memory mapped one stays at 1.577e-03:

| comparison | estimator | flow inference |
|---|---|---|
| built against reloaded | 1.577e-03 | 1.013e-03 |
| built against a contiguous clone of the reloaded weights | **0.000e+00** | **0.000e+00** |
| reloaded against that same clone | 1.577e-03 | |

Two candidate causes were eliminated by measurement rather than by argument. The attention
implementation is `sdpa` on both sides, so the eager float32 softmax path is not involved. Both
models are in eval mode, `training` `False` on the model and on the estimator, so dropout at p=0.1 is
not firing on one side; `same model twice: 0.000e+00` had already excluded that, since dropout in
train mode consumes randomness and two consecutive calls could not have agreed bit for bit.

Three configuration leaves differ after a round trip and none has any effect: `_name_or_path` is `''`
against the save path, and `estimator_config.dtype` and `text_config.dtype` are `None` against
`'float32'`, which is serialisation recording the dtype that was already in force. Both models'
parameters are `torch.float32` throughout, measured. This is the mirror image of the Parler-TTS trap
PROJECT.md records rather than an instance of it: there a sub configuration dtype created a split,
here it records a dtype that is uniform on both sides.

**Generated speech, transcribed back.** Two utterances were synthesized and transcribed with
`facebook/wav2vec2-base-960h`. Model and processor were both built by
`from_pretrained("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")` with no conversion step before it, and the
load report carried no missing, unexpected or mismatched key over 859,185,455 parameters.

| Prompt text | Transcript |
|---|---|
| `The quick brown fox jumps over the lazy dog.` | `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG` |
| `She sells sea shells by the sea shore.` | `SHE SELLS SEASHELLS BY THE SEASHORE` |

Both are word for word, with the connectionist temporal classification decoder merging the two
compounds of the second. The waveforms are 2.84 s at RMS 0.1430 and 2.44 s at RMS 0.1134.

The conditioning was upstream's sft mode, `sft, speaker embedding only, no reference waveform`,
selected by the geometry test rather than hardcoded: a 192 dimensional campplus speaker vector taken
from the CosyVoice 1 speaker table, no prompt speech tokens and no prompt mel spectrogram. The vector
is in distribution because `campplus.onnx` is byte identical at 28,303,423 bytes across v1, v2 and
v3. The text carried the end of prompt token at the front, which is what upstream's assertion
requires and what an empty instruction prefix produces.

That exercises the language model, the flow matching model and the vocoder end to end, and it is a
meaningful test of all three because the v3 language model does not read the speaker embedding at
all, so every word came out of the text path.

That run predates the ONNX port and used upstream's sft mode only. Zero shot voice cloning from a
reference clip, which is the path that derives speech tokens, a mel spectrogram and a speaker
embedding from a waveform, is exercised by the run recorded under "The ported ONNX components"
below.

**The ported ONNX components, against the graphs they replace.** Three LibriSpeech clips of 2.9, 2.5
and 6.5 seconds went through `onnxruntime` and through the PyTorch port, on the same features. The
speaker embedding, 192 dimensions reaching 2.83 in magnitude, differs by at most 8.106e-06. The
speech tokenizer produced 580 token ids over the three clips and every one is identical; its encoder
output, reaching 8.61 in magnitude, differs by at most 1.627e-05. That residual is float32
reassociation, not a difference in the computation.

**Zero shot voice cloning, end to end.** A 5.86 s LibriSpeech clip, which
`facebook/wav2vec2-base-960h` transcribes as `MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND
WE ARE GLAD TO WELCOME HIS GOSPEL`, became 146 speech tokens and a 192 dimensional speaker embedding,
and both were passed to the language model and to the flow matching model together with the prompt
mel spectrogram and the clip's transcript, with the end of prompt marker at the front of the text.

| Asked | Heard back | Waveform |
|---|---|---|
| `The quick brown fox jumps over the lazy dog.` | `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG` | 4.00 s at 24000 Hz, RMS 0.0787, peak 0.5275 |
| `She sells seashells by the seashore.` | `SHE SELLS SEASHELLS BY THE SEASHORE` | 3.00 s at 24000 Hz, RMS 0.0754, peak 0.5809 |

Both are word for word.

The prompt transcript has to be written the way a caller would write it. Passing the same clip's
LibriSpeech transcript verbatim, upper case and unpunctuated, collapsed the same two calls to a 1.52 s
`LAZY DOG` and to a second waveform at RMS 0.0025 that transcribes as nothing at all. Lower casing it
and adding the full stop is the only difference between that and the table above. v1 and v2 generate
the whole sentence either way, so this sensitivity is v3's.

**The text frontend, against `inflect` and against the added vocabulary.** The English number
reading lives in `voicestudio/models/cosyvoice_v1/processing_cosyvoice_v1.py` and is measured there,
against `inflect` 7.3.1, the version upstream's `requirements.txt` pins: 41,821 digit strings with
zero mismatches, and 1,600 strings of 34 to 41 digits agreeing on which side of the largest scale
word they fall. That folder's README carries the corpus.

All 278 tokens the released tokenizer gains are reached. Of the 280 entries in `SPECIAL_TOKENS`, 278
are new to the tokenizer, which grows from 151,646 to 151,924, and none is unknown to it afterwards.
276 are inline markup, and every one of them survives `normalize_text` and encodes to its own single
id through **both** branches, the English one carrying it in `the word @ here.` and the Chinese one
in `这是@的读音。`. The remaining 4 are the `<|...|>` markers, which trip upstream's own guard and come
back verbatim, again each on its own id. Without the markup skip the English branch loses **all 45**
markup tokens that carry a digit, rewriting `[AA1]` to `[AAone]`; measured by running the same 45
through the reading with the skip removed, none survives.

Splitting was checked at the boundary rather than asserted. A 122 token English paragraph splits into
two pieces of 71 and 53 tokens, while the same paragraph cut to 81 tokens stays one piece, which is
`token_max_n` 80 and `token_min_n` 60 together with the rule that a trailing piece shorter than
`merge_len` 20 is merged back. A Chinese paragraph of three sentences splits into two pieces on
character count. With `text_frontend=False` all of them come back as one piece, untouched.

**The text frontend, generated and transcribed back.** See "The text normalizer" below for the full
four way grid on five sentences. The two rows this README carried before were re-run inside it and
both hold, and the older figures are superseded rather than contradicted.

**Streaming input text.** The same clip, the same three seeds, the same transcriber, against the
sentence `The quick brown fox jumps over the lazy dog.` fed as the four chunks `The quick `,
`brown fox `, `jumps over `, `the lazy dog.`:

| Input | WER by seed | Heard back |
|---|---|---|
| the whole text at once | 0.000 / 0.000 / 0.000 | `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG` |
| a generator of the four chunks | 0.333 / 0.333 / 0.333 | `WELCOME HIS GOSPEL THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG` |

Every seed of the streamed decode carries the whole sentence word for word, preceded by the last
three words of the prompt utterance. That prefix is the interleaved layout rather than a defect, and
a redraw shows why: dropping the prompt speech tokens from the language model, which is upstream's
cross lingual shape, leaves the prompt transcript interleaved with no speech to stand in for it and
the model speaks **the whole prompt sentence** first, at 9.5 to 14.8 s and WER 1.889 to 2.444 against
the same three seeds. The prompt text is part of the interleaved stream by construction, so the
decode continues out of it.

The decode was also checked as a decode rather than only by its audio: from seed 0 it yields 118
speech tokens with a maximum id of 6459 against a `speech_vocab_size` of 6561, so every fill token
and the end of speech token were consumed inside the loop and none leaked into the output.

**The interleaved decode against upstream's own `inference_bistream`.** Upstream's method was run
unmodified on the same weights, through an adapter exposing this model's language model under the
attribute names it reads, with both sides drawing from `repetition_aware_sampling` off the same
seed. It agrees **token for token**: 179 tokens against 179, no first difference, from a generator
yielding one text token at a time and again from one yielding five. That covers the end of prompt
split as well, since upstream takes its v3 branch on the class name the adapter carries.

## The text normalizer

`normalize_text` is v2's with v3's markup skip on top, and how it composes `wetext` with
`number_to_words` is described in `voicestudio/models/cosyvoice_v1/README.md`. Two things are v3's
own: the interaction with the inline markup, and the fact that v3's Chinese needs the normalizer far
less than v1's and v2's do.

**The markup skip now covers both branches, and it has to.** `normalize_text` applies the normalizer
span by span, leaving the spans that hold a token of the added vocabulary alone, which is a
deliberate deviation from upstream and is measured rather than argued. Of the 276 markup tokens, 45
carry a stress digit, and `wetext` **raises `AssertionError`** on all 45 of them on the English
branch: `Normalizer().normalize("the word [AA1] here.")` does not return a mangled string, it
crashes. On the Chinese branch it does not crash but rewrites those same 45, turning `[AA1]` into
`[AA一]` and destroying the token. With the skip, **all 276 survive both branches**; without it, the
English branch raises on 45 and the Chinese branch loses 45. Upstream normalizes the whole string on
both branches and has the same two defects, which is one more reason nothing in its open source can
emit this markup.

**English**, zero shot from the 5.86 s LibriSpeech clip, three seeds, `facebook/wav2vec2-base-960h`,
word error rate against the text each setting handed to the model. `off` is `text_frontend=False`,
`today` is this repository without the `frontend` extra, `clvp` is `today` plus `EnglishNormalizer`
from `transformers.models.clvp`, which was measured and rejected, and `wetext` is the extra
installed.

| Case | off | today | clvp | wetext |
|---|---|---|---|---|
| `I paid 1234 dollars in 2025 for 7 books.` | 0.667 / 0.833 / 0.833 | 0.053 / 0.579 / 0.000 | 0.053 / 0.579 / 0.000 | 0.000 / 0.000 / 0.538 |
| `Dr. Smith works at the U.S. Dept. of Energy.` | 1.000 / 0.200 / 0.800 | 1.000 / 0.200 / 0.800 | 0.800 / 0.100 / 0.800 | 0.778 / 0.778 / 0.778 |
| `The book costs $1,234.50 and weighs 2.5 kilograms.` | 2.333 / 1.833 / 1.333 | 0.467 / 0.733 / 0.733 | 0.158 / 0.000 / 0.053 | 0.056 / 0.056 / 0.111 |
| `The St. Louis Co. Ltd. shipped 5 ft. of cable.` | 0.778 / 0.778 / 0.889 | 0.700 / 0.700 / 1.300 | 0.000 / 0.000 / 0.100 | 0.100 / 0.100 / 0.200 |
| `The quick brown fox jumps over the lazy dog.` | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |

Four things to read out of it.

The **digits** row shows `today` and `clvp` **identical seed for seed**, waveform for waveform,
because CLVP rewrites nothing in that sentence: this repository's `number_to_words` had already
consumed every digit. That identity is what makes CLVP's other columns a fair comparison rather than
a coincidence. It also shows `wetext` scoring lower while saying something less correct, because it
reads `1234` as `twelve thirty four`, a year reading of a quantity, and the model then says that
faithfully. A lower figure in this table means the model matched its own reference, not that the
reference was right.

The **money** row is where the frontend as it stands is worst and where it is not obvious from the
text alone: `today` hands the model `The book costs $one,two hundred and thirty-four.fifty and
weighs two.five kilograms.`, and `$1,234.50` comes back as `TWO BLANK QUELL BLADS TWO HUNDRED THIRTY
FOUR FIFTY`. Both normalizers fix the synthesis. `clvp` is the semantically better of the two here,
reading `one thousand, two hundred and thirty-four dollars, fifty cents`, where `wetext` drops the
leading word and reads `thousand two hundred and thirty four point five dollars`.

The **units** row is the reverse and it is why CLVP was rejected. Its figures are the best in the
table, 0.000 / 0.000 / 0.100, and its transcripts read `THE SAINT LOUIS COMPANY LIMITED SHIPPED FIVE
FORT OF CABLE`. CLVP's abbreviation table maps `ft.` to `fort`, so the model is faithfully speaking
a wrong word. `wetext` maps it to `feet` and its transcripts read `FIVE FEET`. Everything CLVP wins,
`wetext` wins as well or better, and CLVP additionally gets `mrs.` wrong as `misess`, `st.` wrong as
`saint` in a street name, and every ordinal wrong, `1st` to `onest` and `21st` to `twenty-onest`.

The **abbreviations** row is a negative result under all four settings, and it is the sentence this
whole investigation started from. `Dr.` and `U.S.` are read correctly by v1 and v2 unaided; what
fails is `Dept.`, which is absent from CLVP's 18 entry table and from `wetext`'s grammar alike. On
v3 specifically the sentence also truncates, most seeds returning `OF ENERGY` regardless of setting,
which is a decode failure rather than a text one. **This case is not closed and no setting closes
it.**

**Chinese**, upstream's sft mode with the `中文女` speaker vector from
`FunAudioLLM/CosyVoice-300M-SFT`, no reference waveform, three seeds,
`openai/whisper-large-v3-turbo`, character error rate against the original sentence rather than each
setting's own text, since the two settings write a number differently and the transcriber writes it
a third way:

| Case | today | wetext |
|---|---|---|
| `会议定在2025年3月8日下午3点30分开始。` | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |
| `这本书卖1234.50元，比原价便宜25%。` | 0.333 / 0.000 / 0.333 | 0.000 / 0.000 / 0.000 |
| `他跑了5公里，用了30分钟，体重是65kg。` | 0.421 / 0.263 / 0.474 | 0.421 / 0.421 / 0.105 |
| `电话号码是13800138000，房间号是302。` | 0.000 / 0.000 / 0.043 | 0.043 / 0.261 / 0.000 |
| `今天天气很好，我们一起去公园散步吧。` | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 |

**v3 is the outlier, and this is worth knowing before assuming the extra is needed.** Where v1 and
v2 fall apart on an unnormalized Chinese date or phone number, v3 reads both back perfectly with no
normalizer at all: all three seeds return `会议定在2025年3月8日下午3点30分开始` and
`电话号码是13800138000，房间号是302`. Its units transcripts already say `五公里`, `三十分钟` and
`六十五千克` unaided; the figure is inflated only because the transcriber writes Chinese numerals
where the original has digits. The one case that genuinely improves is the decimal currency amount,
where `today` reads `1234.50` as `一二三四五十`, digit by digit with a stray fifty, on two of three
seeds. Compare v2's table, where four rows collapse without the normalizer.

## The second language model checkpoint

The released directory ships `llm.rl.pt` beside `llm.pt`, and nothing in this folder reads it. What
follows is what it is, measured rather than inferred from the name, against commit
`29e01c4e8d000f4bcd70751be16fa94bf3d85a18` of `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`.

**It is a second full checkpoint, not a delta.** `llm.pt` is 2,024,669,519 bytes and `llm.rl.pt` is
2,024,682,701, a difference of 13,182. Both load with `torch.load(weights_only=True)` to a plain
dictionary of 293 tensors, the key sets are exactly equal, all 293 shapes and dtypes match, and
neither file carries a single non tensor entry: no optimizer state, no step counter, no metadata.
Both load into a stock `Qwen2ForCausalLM` built from the shipped `CosyVoice-BlankEN/config.json`
with zero missing and zero unexpected keys. It is not a LoRA, an adapter or a trainer state.

**The weights have moved much further than the name suggests.** Per tensor figures are
`‖b − a‖₂ / ‖a‖₂` and elementwise `max|a − b|` with `a` the base file and `b` the RL one; a group
figure is `sqrt(Σ‖b − a‖² / Σ‖a‖²)` over the group's tensors, not a mean of ratios.

| module group | relative L2 | max abs diff |
|---|---|---|
| `layers.*.self_attn.o_proj` | 1.602 | 1.612 |
| `layers.*.mlp.down_proj` | 1.577 | 1.347 |
| `layers.*.mlp.up_proj` | 1.561 | 0.586 |
| `layers.*.mlp.gate_proj` | 1.550 | 0.866 |
| `llm_decoder`, the speech token output head | 1.413 | 0.4205 |
| `lm_head` and `embed_tokens`, which are tied | 1.033 | 0.2195 |
| `layers.*.self_attn.q_proj` | 0.751 | 65.55 |
| `layers.*.self_attn.k_proj` | 0.594 | 88.29 |
| `speech_embedding` | 0.585 | 2.655 |
| `model.norm` | 0.541 | 8.921 |

Zero of the 293 tensors are bit identical, and 0.999999 of the 642,283,136 parameters differ; the
898 exactly equal elements all sit in `speech_embedding.weight`. Overall relative L2 over the whole
checkpoint is 0.657. **Every embedding and every head moved**, so nothing can be shared between the
two. The speech token input embedding is the one that moved in scale rather than direction, cosine
0.9919 with its root mean square rescaled from 0.8907 to 0.3753, while the speech token output head
is at cosine 0.3415.

**It is the same lineage, but it is not `llm.pt` plus a small aligned delta.** Against the shipped
`CosyVoice-BlankEN/model.safetensors`, cosines stay positive throughout and the final norm channel
scale is preserved in all three at cosine at least 0.9995, so both descend from the same base. But
`llm.rl.pt` sits strictly further from that base than `llm.pt` does, `embed_tokens` at 1.239 relative
against 0.783 and layer 23's `o_proj` at 2.633 against 1.391, and the delta direction test
`cos(a − base, b − base)` is only 0.33 to 0.37 at layer 0 and 0.07 to 0.09 in the upper layers.
Either the reinforcement learning run was long with no effective anchor, or the branch went through
further supervised fine tuning first. Weights alone cannot separate those two.

**What the file is for comes from the repository card, not from the code.** Its evaluation table
carries two rows, and the second is the characteristic signature of reinforcement learning against a
recognition derived reward, large error rate gains with a small speaker similarity regression:

| model | test-zh CER | test-zh SIM | test-en WER | test-en SIM | test-hard CER | test-hard SIM |
|---|---|---|---|---|---|---|
| Fun-CosyVoice3-0.5B-2512 | 1.21 | 78.0 | 2.24 | 71.8 | 6.71 | 75.8 |
| Fun-CosyVoice3-0.5B-2512_RL | 0.81 | 77.4 | 1.68 | 69.5 | 5.44 | 75.0 |

**Nothing in upstream's code loads it.** `grep -rn "llm\.rl\|rl\.pt"` over the whole vendored tree
returns no hit. The only load path is hardcoded, `cosyvoice/cli/cosyvoice.py:213`
`self.model.load('{}/llm.pt'.format(model_dir), ...)`, with no flag, argument or configuration field
that selects another file. Upstream's own recipes never produce this name either: the direct
preference optimization recipe averages its result back into a plain `llm.pt`
(`examples/libritts/cosyvoice2/run_dpo.sh`), and the GRPO recipe writes a Hugging Face format
directory (`examples/grpo/cosyvoice2/run.sh`). The upstream `README.md` roadmap mentions releasing a
"rl model and its training/inference script"; the script is not in the tree.

**What it would take to use it here.** The registered conversion mapping needs no change at all:
because the key sets are identical, `WeightRenaming(r"^llm\.llm\.model\.model\." → r"llm\.model\.")`
in `modeling_cosyvoice_v3.py` matches this file exactly as it matches `llm.pt`. Four things are
missing, and none of them is in this folder alone:

1. `CHECKPOINT_FILES` in `voicestudio/models/cosyvoice_v1/weight_conversion.py` is a module constant
   that both `load_checkpoint` and `write_checkpoint` iterate directly, so which language model file
   the merge reads cannot be selected.
2. The same constant is the `allow_patterns` list `resolve_checkpoint` passes to `snapshot_download`,
   so the file would never be fetched.
3. `converted_checkpoint` keys the cache on the model type and the source snapshot revision alone.
   Both variants come from one repository at one commit, so they would hash to the same directory and
   whichever converted first would silently serve both.
4. `from_pretrained` forwards its keyword arguments verbatim, and neither `CosyVoiceV3Config` nor
   `PUBLISHED_CHECKPOINTS` carries a variant field, so there is no caller facing knob to add one to.

None of that is a large change, but it reaches outside this folder, and no verification of this
checkpoint through the migrated model exists. The decision to wire it in is left open.

## Not carried over from upstream

Recorded per CLAUDE.md section 2.6.

- **The vocoder's adversarial objective.** `cosyvoice/hifigan/hifigan.py` is unchanged from v1 and
  v2: `generator_loss` plus 2.0 times feature matching plus 45 times mel reconstruction plus 1.0
  times `tpr_loss` at tau 0.04 plus an L1 on the f0, with the discriminator optimized in an
  alternating turn. `MultipleDiscriminator` and the three losses from `matcha.hifigan.models` are not
  implemented, so the vocoder cannot be trained the way upstream trained it. Leaving them out follows
  the `transformers` convention on GAN trained vocoders, measured over the 494 `modeling_*.py` files
  in the 510 model folders of `transformers` 5.16.1. `Discriminator` appears in two of those files,
  `electra/modeling_electra.py` and `funnel/modeling_funnel.py`, and both are pretraining heads over
  token logits, `ElectraDiscriminatorPredictions` and `FunnelDiscriminatorPredictions` under
  `FunnelForPreTraining`, not adversaries over a waveform. `adversarial` appears in none of them.
  Every vocoder shipped takes a spectrogram or codes and returns a bare tensor, and none takes
  `labels`:

  | Class | `forward` signature |
  |---|---|
  | `SpeechT5HifiGan` | `(self, spectrogram, **kwargs) -> torch.FloatTensor` |
  | `FastSpeech2ConformerHifiGan` | `(self, spectrogram, **kwargs) -> torch.FloatTensor` |
  | `VitsHifiGan` | `(self, spectrogram, global_conditioning=None) -> torch.FloatTensor` |
  | `SeamlessM4THifiGan`, `SeamlessM4Tv2HifiGan` | `(self, inputs_embeds) -> torch.FloatTensor` |
  | `SeamlessM4TCodeHifiGan` | `(self, input_ids, spkr_id, lang_id, **kwargs) -> tuple[torch.Tensor]` |
  | `SeamlessM4Tv2CodeHifiGan` | `(self, input_ids, speaker_id, lang_id, **kwargs) -> tuple[torch.Tensor]` |
  | `Qwen2_5OmniToken2WavBigVGANModel` | `(self, mel_spectrogram, **kwargs)`, returning a clamped waveform |

  VITS is the case that settles it, because upstream VITS is GAN trained end to end:
  `VitsModel.forward` declares `labels` and its body opens with
  `raise NotImplementedError("Training of VITS is not supported yet.")`. The one shipped speech
  synthesis model with a real objective, `FastSpeech2ConformerWithHifiGan`, sums L1 mel, duration,
  pitch and energy in `FastSpeech2ConformerLoss` and carries no adversarial term, and its vocoder
  half takes no labels. Against that convention, the two terms this folder inherits from v1 and does
  score, the mel reconstruction loss and the f0 loss, go beyond it rather than falling short of it.
  The consequence to know is that a vocoder trained through them alone would not reproduce a released
  checkpoint.
- **`llm.rl.pt`.** The released v3 directory ships a second language model checkpoint alongside
  `llm.pt`, and nothing here reads it. What it is was established by measurement rather than from
  its name, and the decision is still open; see "The second language model checkpoint" below.
- **Direct preference optimization.** `Qwen2LM.forward_dpo` and `DPOLoss` need a second model
  instance and a preference batch, so they do not fit inside a single `forward`. Still open, with the
  same note as v2 that upstream averages its log probabilities over the positions where the target
  **is** `IGNORE_ID`, which reads like a sign error.
- **`ttsfrd`.** `wetext` is now reachable, as the `frontend` extra, and "The text normalizer" below
  records what it does and does not fix. `ttsfrd`, which `CosyVoiceFrontEnd.__init__` tries first, is
  not: it is a closed source Alibaba wheel whose rules ship as a separate `CosyVoice-ttsfrd` resource
  pack, with no source release, so it cannot be traced under CLAUDE.md section 2.2 and cannot be
  declared under section 9.1. Whether anything is left that only `ttsfrd` would fix cannot be
  measured without running it, and that is **still open**. One concrete candidate is on record: the
  `Dr. Smith works at the U.S. Dept. of Energy.` case below is not fixed by `wetext` either, because
  `Dept.` is not in its grammar.
- **A producer for the inline markup.** Unchanged by any of this. Nothing in the open upstream source
  emits the ARPAbet and pinyin markup the 278 added embedding rows are for; the only producer is
  `ttsfrd`. A caller supplies it, and `normalize_text` now carries it through both branches
  untouched.

## Repository integration

Three things outside this folder are still needed and were deliberately not touched:

- `voicestudio/models/__init__.py` needs a `from .cosyvoice_v3 import *` line in its alphabetical
  list, after `cosyvoice_v2`.
- `PROJECT.md`'s status table still records this model as not started, and its sibling inheritance
  map still lists the f5_tts entries as gated on this migration rather than resolved by it.
- `pyproject.toml` still declares an `onnx` extra holding `onnxruntime`, `onnxruntime-gpu` and
  `onnx`. Nothing in these three folders imports any of them any more, and no other model folder does
  either, so that extra and its entry in `all` can go. Otherwise this folder needs nothing: it
  imports `voicestudio.models.f5_tts`, which is already in the repository, and nothing new from
  outside it.
