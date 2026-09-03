# CosyVoice v1

CosyVoice v1 synthesizes speech in three stages. An autoregressive language model turns Whisper
text tokens, an utterance level speaker embedding and an optional speech token prompt into a
sequence of 4096 supervised semantic speech tokens at 50 Hz. A conditional flow matching model
turns those tokens into an 80 bin mel spectrogram, conditioned on the same speaker embedding and
on the prompt's mel spectrogram, with a one dimensional UNet estimator and a fixed step Euler
solver under classifier free guidance. A HiFTNet vocoder turns the mel spectrogram into a 22050 Hz
waveform through a neural source filter and an inverse short time Fourier transform head.

The three networks are trained separately upstream and are released as three separate `.pt` files.

Original model and code: https://github.com/FunAudioLLM/CosyVoice

## Usage

```python
from voicestudio.models.cosyvoice_v1 import CosyVoiceV1ForConditionalGeneration, CosyVoiceV1Processor

model = CosyVoiceV1ForConditionalGeneration.from_pretrained("cosyvoice-v1-converted")
processor = CosyVoiceV1Processor.from_pretrained("cosyvoice-v1-converted")

inputs = processor(text="The quick brown fox jumps over the lazy dog.")
prompt = processor.get_speaker("英文女")

waveform = model.generate(
    input_ids=inputs.input_ids,
    speaker_embedding=prompt.speaker_embedding,
    flow_prompt_speech_token_ids=prompt.prompt_speech_token_ids,
    prompt_speech_feat=prompt.speech_feat,
)
```

`get_speaker` reads the `spk2info.pt` at the processor's `speaker_info_path`. `CosyVoice-300M-SFT`
and `CosyVoice-300M-Instruct` ship one holding seven speakers; the base model does not, and the
only table that covers it is the one in the stale `model-scope/CosyVoice-300M` mirror, which
carries a speech token sequence and a prompt mel per speaker rather than an embedding alone.

`prompt_speech_token_ids` conditions the language model and `flow_prompt_speech_token_ids`
conditions the flow matching model. Passing both, together with `prompt_input_ids`, is upstream's
zero shot mode; passing only the flow one is upstream's cross lingual mode; passing
`source_speech_token_ids` bypasses the language model and performs voice conversion.

Converting a released directory:

```python
from voicestudio.models.cosyvoice_v1.weight_conversion import convert

convert("base", "cosyvoice-v1-converted")
```

## Training

`CosyVoiceV1ForConditionalGeneration.forward(labels=...)` returns the language model objective
only, because upstream trains the three networks one at a time with three separate runs of
`cosyvoice/bin/train.py`. The other two objectives are on the submodules:

- **Language model.** `LabelSmoothingLoss` over `speech_vocab_size + 1` classes with
  `smoothing=0.0` and `normalize_length=True`, that is a cross entropy divided by the number of
  unmasked targets. Targets are built by `build_speech_token_labels`: `-1` on the start of
  sequence step, on the speaker embedding step and on every encoded text step, then the utterance's
  speech tokens, then the end of speech token `speech_vocab_size`. The speech tokens are teacher
  forced through `speech_embedding` and come from the speech tokenizer, not from the waveform.
  Upstream also reports a token accuracy, which `CosyVoiceV1Output.accuracy` carries and which is
  not part of the loss.
- **Flow matching.** `CosyVoiceV1FlowModel.forward` returns a single term,
  `CosyVoiceV1ConditionalCFM.compute_loss`: optimal transport conditional flow matching with
  `sigma_min=1e-06`, one timestep per sample drawn from `U(0, 1)`,
  `y = (1 - (1 - sigma_min) t) z + t x1` and target `u = x1 - (1 - sigma_min) z`, scored with
  `mse_loss(pred * mask, u * mask, reduction="sum") / (mask.sum() * n_feats)`. With probability
  `training_cfg_rate` (0.2) per sample the encoder output, the speaker embedding and the
  conditioning mel are all zeroed. The conditioning mel is, with probability one half per sample, a
  prefix of the ground truth mel of a length drawn uniformly from `0` to `0.3 * num_frames`, and
  zeros otherwise. The target mel comes from the waveform.
- **Vocoder.** `CosyVoiceV1HiFTGenerator.forward` returns the waveform and the predicted f0, which
  are the two generator side inputs of upstream's objective. The objective itself is not
  implemented; see "Not carried over from upstream".

Upstream freezes nothing. The only `requires_grad` assignment in upstream's tree is
`Snake.alpha.requires_grad = alpha_trainable` in `cosyvoice/transformer/activation.py`, whose
default is `True` and which the vocoder never overrides, and the only `.eval()` on a training path
is `self.ref_model.eval()` in `Executor`, which belongs to the CosyVoice 2 DPO recipe. Both freeze
bugs the project has hit elsewhere, a frozen module that still runs dropout because nothing called
`.eval()`, and a freeze that does not survive `from_pretrained` because transformers 5 rebuilds
`Parameter` objects, are therefore not reachable here: there is nothing to freeze. What was checked
instead is that each objective reaches only its own network. The language model loss puts gradient
on 401 parameters, all of them under `llm`, and none under `flow` or `hift`; the flow loss puts
gradient on 1185 parameters, all under `flow`, and none under `llm` or `hift`.

## Lineage

CosyVoice v1's flow decoder is `MaskedDiffWithXvec` with a `ConditionalCFM` head over a
`ConditionalDecoder`, which is a one dimensional UNet built from Matcha-TTS blocks. It is not
`CausalMaskedDiffWithDiT`; that class belongs to CosyVoice 3, and it is the reason the DiT lineage
does not apply to this model.

Against the sibling inheritance map:

- **`FeedForward` against f5_tts.** Rejected. `F5TTSFeedForward` is
  `Sequential(Sequential(Linear, GELU(approximate="tanh")), Dropout, Linear)`, while the estimator's
  feed forward is diffusers' `FeedForward(activation_fn="gelu")`, a `ModuleList` of
  `GELU(proj=Linear)` with `approximate="none"`, a dropout and a linear. The activation
  approximation differs, and so do the parameter paths: the checkpoint stores
  `ff.net.0.proj.weight` and `ff.net.2.weight`, which f5_tts's layout cannot express.
- **`TimestepEmbedding` against f5_tts.** Rejected for the same reason.
  `F5TTSTimestepEmbedding` bundles the sinusoidal embedding and a `Sequential(Linear, SiLU, Linear)`
  named `time_mlp` into one module, while the checkpoint stores a separate `time_embeddings` (no
  parameters) and `time_mlp.linear_1` / `time_mlp.linear_2`, which is diffusers' layout. The
  sinusoidal formula itself is identical between `F5TTSSinusPositionEmbedding` and Matcha's
  `SinusoidalPosEmb`, so only that parameter free piece could be inherited, and inheriting a
  parameter free ten line function across model folders buys nothing.
- **`Encoder` and `EncoderLayer` against prompt_tts_pp.** Rejected. prompt_tts_pp inherits
  `FastSpeech2ConformerAttention`, `FastSpeech2ConformerConvolutionModule` and
  `FastSpeech2ConformerMultiLayeredConv1d`, and writes its own `Encoder`/`EncoderLayer` on top.
  CosyVoice's encoders come from WeNet, not ESPnet's FastSpeech2 line, and diverge in three ways
  that reach the weights: the feed forward is two plain `Linear` layers (`w_1`, `w_2`) rather than
  `FastSpeech2ConformerMultiLayeredConv1d`'s two `Conv1d` layers, the checkpoint has neither a
  convolution module nor a macaron branch (`use_cnn_module: False`, `macaron_style: False`), and
  the input is a linear projection with no subsampling rather than a conformer input layer.
  What is left after removing the parts that differ is a plain pre norm block, so there is no
  shared surface worth inheriting.
- **`SourceModule` against prompt_tts_pp.** Rejected. `PromptTTSPPSourceModule` holds only a
  `Linear`, and its sine generation integrates phase with the interpolation trick BigVGAN uses.
  CosyVoice's `SineGen` at 22050 Hz is upstream's `sinegen_type='1'`, which builds harmonics
  directly and wraps with `2 * pi * (cumsum(f0 * i / sr) % 1)`, draws one uniform phase per
  harmonic in `[-pi, pi]` with the fundamental pinned to zero, and applies a `Tanh` module the
  checkpoint does not store but the forward does use. The two produce different waveforms from the
  same f0.
- **`Snake` against bigvgan.** Still open. `voicestudio/models/bigvgan/` does not exist yet, so
  `CosyVoiceV1Snake` is local. It is upstream's `Snake(channels, alpha_logscale=False)`, that is
  `x + (1 / (alpha + 1e-9)) * sin(x * alpha) ** 2` with a per channel `alpha` initialised to ones.
  When bigvgan lands, `CosyVoiceV1Snake` and `CosyVoiceV1ResBlock` are the two classes that should
  inherit from it.

Nothing in transformers-tts covers the missing pieces either: it ships no CosyVoice of any version,
no `SinusoidalPosEmb` / `Block1D` / `ResnetBlock1D` / `BasicTransformerBlock`, no conditional flow
matching wrapper, and no neural source filter.

## Dependencies

Upstream pins more than forty packages. What each turned into:

| Upstream dependency | What replaced it |
|---|---|
| `diffusers==0.29.0` | Removed. Its `GELU`, `Attention` with `AttnProcessor2_0`, `FeedForward` and `TimestepEmbedding` are the only pieces the estimator used, and each is a few dozen lines that now live in `modeling_cosyvoice_v1.py`. |
| `matcha-tts` | Removed. `SinusoidalPosEmb`, `Block1D`, `ResnetBlock1D`, `Downsample1D`, `Upsample1D` and `BASECFM` are inlined the same way. |
| `einops` | Removed. Three call sites, `pack([x, mu], "b * t")`, `rearrange(x, "b c t -> b t c")` and `repeat(spks, "b c -> b c t")`, are a concatenation, a transpose and an expand. |
| `omegaconf`, `hyperpyyaml` | Removed. The released `cosyvoice.yaml` is now `CosyVoiceV1Config`. |
| `torchdiffeq` | Not used by v1; the Euler solver is upstream's own `solve_euler`, kept inline. |
| `deepspeed`, `lightning`, `tensorboard`, `wetext`, `inflect`, `ttsfrd` | Training and text frontend tooling, out of the model's scope. See "Not carried over from upstream" for the text frontend. |
| `openai-whisper` | Replaced by `WhisperTokenizer` for text and `WhisperFeatureExtractor` for the speech tokenizer's mel. |
| `onnxruntime` | **Not removed.** See below. |
| `tensorrt`, `vllm`, `modelscope`, `gradio`, `fastapi`, `grpcio` | Serving paths, not part of the model. |
| `transformers==4.51.3` | Replaced by the `transformers-tts` 5.x fork this repository targets. |

`onnxruntime` is the one that could not be removed. CosyVoice v1's speech tokenizer
(`speech_tokenizer_v1.onnx`, 522 MB) and speaker encoder (`campplus.onnx`, 28 MB) are published as
ONNX graphs only. The section 2.3 search found no PyTorch release of the v1 speech tokenizer
anywhere: `xingchensong/S3Tokenizer` is a reimplementation that downloads the same ONNX file and
converts it at load time, and the PyTorch tokenizer repositories that do exist
(`ResembleAI/s3tokenizer-v2`, `mlx-community/S3TokenizerV2`, `mlx-community/S3TokenizerV3`) are
v2 and v3 only. The speaker encoder does have a PyTorch release, `campplus_cn_common.bin` on
`funasr/campplus` and on ModelScope `iic/speech_campplus_sv_zh-cn_16k-common`. Nothing was added to
`pyproject.toml`; `CosyVoiceV1Processor` imports `onnxruntime` lazily and raises with an
explanation if it is missing, so every other path works without it. Converting both graphs to
PyTorch is the work this leaves open.

## Verification

Everything below ran on CPU in float32 in the project venv, against the real
`FunAudioLLM/CosyVoice-300M` weights. Nothing ran on a GPU.

**Checkpoint coverage.** 1813 tensors (401 from `llm.pt`, 1185 from `flow.pt`, 227 from `hift.pt`)
convert onto 436,053,808 parameters with zero MISSING and zero UNEXPECTED keys, so no source weight
goes unused. `f0_predictor.condnet` holds its five convolutions at indices 0, 2, 4, 6 and 8, as the
checkpoint confirms.

**Numeric parity against upstream's own classes.** The upstream tree was still in this folder when
these ran, so every migrated module was run side by side with the class it replaces, loaded from
the same weights. Reported as `max|diff|`:

| Component | Upstream class | max abs difference |
|---|---|---|
| Text encoder | `ConformerEncoder` | 0.0 |
| Language model backbone, full sequence | `TransformerEncoder` | 0.0 |
| Language model prefill | `TransformerEncoder.forward_chunk` | 0.0 |
| Language model one cached decode step | `TransformerEncoder.forward_chunk` | 0.0 |
| Flow encoder | `ConformerEncoder` | 0.0 |
| Length regulator, forward and inference | `InterpolateRegulator` | 0.0 |
| Label smoothing loss | `LabelSmoothingLoss` | 0.0 |
| Flow matching estimator | `ConditionalDecoder` | 0.0 |
| Euler solver, 10 steps with guidance 0.7 | `ConditionalCFM.forward` | 0.0 |
| Euler solver streaming cache, and a second chunk from it | `ConditionalCFM.forward` | 0.0 |
| Flow matching loss | `ConditionalCFM.compute_loss` | 0.0 |
| Flow model inference | `MaskedDiffWithXvec.inference` | 0.0 |
| f0 predictor | `ConvRNNF0Predictor` | 0.0 |
| Neural source filter | `SourceModuleHnNSF` | 7.5e-09 |
| Vocoder, from a shared excitation | `HiFTGenerator.decode` | 3.8e-06 |
| Vocoder, inference end to end | `HiFTGenerator.inference` | 3.2e-06 |
| Vocoder, training entry point | `HiFTGenerator.forward` | 1.1e-05 |
| Mel front end | `matcha.utils.audio.mel_spectrogram` | 0.0 |

The vocoder is the only component that is not bit exact. The cause is the analysis window:
upstream builds it with `scipy.signal.get_window("hann", 16, fftbins=True)` and this model with
`torch.hann_window(16, periodic=True)`, which are the same function rounded differently in
float32 and differ by 1.2e-07. That difference enters through the excitation's short time Fourier
transform and grows to 3.8e-06 on a signal whose maximum is 0.99.

The autoregressive loop was checked as a loop, not only as a forward pass: from the same seed,
`TransformerLM.inference` and `generate_speech_tokens` produced the same 480 speech tokens, token
for token.

**Training objectives.** With both random number generators pinned, the language model loss is
16.041545867919922 on upstream and 16.041545867919922 here, and the flow matching loss is
8.12406063079834 on both. Gradients land only where they should: the language model loss reaches
401 parameters, all under `llm`; the flow loss reaches 1185, all under `flow`.

**Generated speech, transcribed back.** Two utterances were synthesized from the converted weights
and transcribed with `facebook/wav2vec2-base-960h`. The prompt for each is one of the
speakers in the base model's own `spk2info.pt`, which carries a speech token sequence, a prompt mel
and a speaker embedding, so no ONNX graph was involved. The language model saw only the speaker
embedding, which is upstream's cross lingual conditioning.

| Prompt text | Transcript |
|---|---|
| `The quick brown fox jumps over the lazy dog.` | `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG` |
| `She sells sea shells by the sea shore.` | `S SHE SELLS SEA SHELLS BY THE SEASHORE` |

The first is word for word. The second carries a spurious leading `S` and renders `sea shore` as
one word, both of which are artifacts of the connectionist temporal classification decoder rather
than of the synthesis. The two waveforms are 3.41 s at RMS 0.0612 and 2.14 s at RMS 0.0468.

**Round trip through the hub format.** `weight_conversion.convert` writes a directory that
`from_pretrained` loads back to the same 436,053,808 parameters, `save_pretrained` followed by
`from_pretrained` reproduces every parameter with a largest difference of 0.0 and leaves no meta
parameter or buffer behind. A model reloaded that way computes bit for bit what the model it was
saved from computes: 0.0 on the text encoder, 0.0 on the language model logits and 0.0 on the
vocoder waveform.

Getting there took two fixes, both of the same kind. Transformers 5 builds the model on the meta
device before loading weights, and a non-persistent buffer computed in a constructor is then
materialised as uninitialised memory instead of by rerunning that computation. Two of them were
wrong after a reload: the inverse short time Fourier transform window, which read
`[-1.78e-13, 0.0, 0.0, 0.0, ...]` instead of a Hann window, and the relative positional embedding
table shared by all three encoders. Neither raised. The vocoder ran and the reloaded model
generated fluently, but the words were gone: `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG` came
back from the reloaded model as `HADNIN YOUR DET SSFUL I SA YO TO DEING HEE`. Both tables are now
built on first use instead of in the constructor. This is the failure PROJECT.md recorded from the
abandoned branch as `torch.istft`'s NOLA check failing with `window overlap add min: 1`; on
transformers 5.16 it does not raise at all, which is worse.

**Not verified.** The speech tokenizer and speaker encoder paths of the processor were not run,
because `onnxruntime` is not installed and must not be installed to make a migration work. That
leaves two things unchecked: whether `WhisperFeatureExtractor(feature_size=128)` with
`padding=False` reproduces `whisper.log_mel_spectrogram(speech, n_mels=128)` frame for frame, and
whether `WhisperTokenizer.from_pretrained("openai/whisper-large-v3")` produces the same ids as
openai-whisper's `get_tokenizer(multilingual=True, num_languages=100).encode(text,
allowed_special="all")`. The generation above exercises the second one indirectly.

## Not carried over from upstream

Recorded per CLAUDE.md section 2.6. None of these is resolved here.

- **The vocoder's adversarial objective.** Upstream trains HiFT as a GAN through
  `cosyvoice/hifigan/hifigan.py`. The generator loss is five terms: `generator_loss` over the
  discriminators, `feat_match_loss_weight` (2.0) times the feature matching loss,
  `multi_mel_spectral_recon_loss_weight` (45) times a mel reconstruction loss, `tpr_loss_weight`
  (1.0) times `tpr_loss` at `tpr_loss_tau` 0.04, and an L1 loss between the predicted f0 and the
  pitch feature. The discriminator loss is `discriminator_loss` plus the same weighted `tpr_loss`,
  optimized in an alternating turn with its own optimizer. `MultipleDiscriminator` in
  `cosyvoice/hifigan/discriminator.py` and the three loss functions upstream imports from
  `matcha.hifigan.models` are not implemented here, so the vocoder cannot be trained the way
  upstream trained it. `CosyVoiceV1HiFTGenerator.forward` returns the waveform and the f0 that two
  of those terms consume, and nothing else. The target of the f0 term is `pitch_feat`, which
  upstream builds in `cosyvoice/dataset/processor.py` with `pyworld.harvest`, falling back to
  `pyworld.dio` when fewer than five frames come back voiced, refining with `pyworld.stonemask` and
  interpolating onto the mel frame count. That file and the three named above are the four upstream
  files this objective lives in and all four are still in this folder, but the `matcha` losses are
  not and never were. This is the same shape as the open Vocos item and it needs a human.
- **The text frontend.** Upstream's `CosyVoiceFrontEnd.text_normalize` runs the input through
  `ttsfrd` if the resource pack is installed, otherwise `wetext`, otherwise nothing, and then
  splits long text into sentences with `split_paragraph`, using `inflect` for English number
  expansion. `CosyVoiceV1Processor` tokenizes the text as given. Synthesizing text with digits,
  abbreviations or more than one sentence therefore does not behave the way upstream does.
- **`ttsfrd`.** Upstream ships it as a wheel in `FunAudioLLM/CosyVoice-ttsfrd` with a 339 MB
  resource pack. It is not on PyPI and has no source release.
- **The speech tokenizer and the speaker encoder.** Both are still ONNX graphs, so the processor
  still depends on `onnxruntime` for the paths that derive a prompt from a waveform. See
  "Dependencies".
- **Streaming input text.** Upstream's `inference_bistream` accepts a text generator and
  interleaves text and speech tokens. It exists only on `Qwen2LM`, that is CosyVoice 2 and 3, and
  is not part of v1.
- **`n_feats` on the flow matching head.** Upstream's `ConditionalCFM` passes `in_channels` (240)
  to `BASECFM` as `n_feats`, then ignores it and hardcodes 80 everywhere it matters.
  `CosyVoiceV1ConditionalCFM.n_feats` is `estimator_out_channels` (80), which is the value upstream
  actually uses. Identical behaviour for v1; a subclass with a different mel width would diverge.
- **The `CosyVoice-300M-25Hz` text tokenizer.** `cosyvoice/tokenizer/tokenizer.py` holds a second
  text tokenizer, reached through `get_tokenizer` and the
  `multilingual_zh_ja_yue_char_del.tiktoken` asset beside it, which upstream's recipe selects with
  `get_tokenizer: !name:cosyvoice.tokenizer.tokenizer.get_tokenizer` when training the
  `CosyVoice-300M-25Hz` release rather than the 50 Hz one this migration converts. It is not the
  same vocabulary: it carries 58836 mergeable ranks where `openai/whisper-large-v3` carries 51866,
  9292 of its tokens are absent from that vocabulary and 48003 of the rest sit at a different rank,
  and its special tokens add audio event, emotion and TTS vocal markers whisper has none of.
  `CosyVoiceV1Processor` has no counterpart for it and `PUBLISHED_CHECKPOINTS` does not list the
  25 Hz release, so that checkpoint cannot be loaded or tokenized here. The file and its asset stay
  in this folder for that reason, and whether to cover the 25 Hz release at all is the decision
  this leaves open.

## File map

One counterpart per upstream file, per CLAUDE.md section 2.4. The renames are recorded in
`7e93ecff` and the removals in `86a9fa18`, so `git log --follow` walks back into the original
authors' commits and `git show` recovers any file named below.

### Transformed into this folder

| Upstream file | Where it went |
|---|---|
| `cosyvoice/llm/llm.py` | `modeling_cosyvoice_v1.py`. `TransformerLM` became `CosyVoiceV1SpeechTokenLM` plus `CosyVoiceV1ForConditionalGeneration`. |
| `cosyvoice/cli/model.py` | `generation_cosyvoice_v1.py`. `CosyVoiceModel.token2wav` and `CosyVoiceModel.tts` became `CosyVoiceV1GenerationMixin`. |
| `cosyvoice/cli/frontend.py` | `processing_cosyvoice_v1.py`. |
| `cosyvoice/cli/cosyvoice.py` | `weight_conversion.py`. Its `load` method became the conversion. |
| `cosyvoice.yaml` | `configuration_cosyvoice_v1.py`. |
| `cosyvoice/__init__.py` | `__init__.py`. |

### Removed, with a counterpart in one of the three folders

| Upstream file | Counterpart |
|---|---|
| `cosyvoice/flow/flow.py` | `MaskedDiffWithXvec` is `CosyVoiceV1FlowModel`, `CausalMaskedDiffWithXvec` is `CosyVoiceV2FlowModel`, `CausalMaskedDiffWithDiT` is `CosyVoiceV3FlowModel`. |
| `cosyvoice/flow/flow_matching.py` | `ConditionalCFM` is `CosyVoiceV1ConditionalCFM`, `CausalConditionalCFM` is `CosyVoiceV2ConditionalCFM` and `CosyVoiceV3ConditionalCFM`. |
| `cosyvoice/flow/decoder.py` | `ConditionalDecoder` is `CosyVoiceV1ConditionalDecoder` and `CausalConditionalDecoder` is `CosyVoiceV2ConditionalDecoder`. `CausalBlock1D`, `CausalResnetBlock1D`, `CausalConv1d` and `Transpose` are the `CosyVoiceV2` classes of the same names. |
| `cosyvoice/flow/length_regulator.py` | `InterpolateRegulator` is `CosyVoiceV1InterpolateRegulator`. |
| `cosyvoice/flow/DiT/dit.py` | `DiT` is `CosyVoiceV3ConditionalDecoder` and `InputEmbedding` is `CosyVoiceV3InputEmbedding`. `TextEmbedding` is F5-TTS carryover that `DiT` never instantiates. |
| `cosyvoice/flow/DiT/modules.py` | `TimestepEmbedding` is `CosyVoiceV3TimestepEmbedding`, `DiTBlock` is `CosyVoiceV3DecoderLayer`, `AdaLayerNormZero_Final` is `CosyVoiceV3AdaLayerNormFinal` and `CausalConvPositionEmbedding` is `CosyVoiceV3CausalConvPositionEmbedding`. `Attention`, `AttnProcessor`, `AdaLayerNormZero`, `FeedForward` and `SinusPositionEmbedding` are inherited from `voicestudio/models/f5_tts/`. `MelSpec`, `GRN`, `ConvNeXtV2Block`, `MMDiTBlock`, `JointAttnProcessor`, `ConvPositionEmbedding`, `precompute_freqs_cis` and `get_pos_embed_indices` are F5-TTS carryover that no CosyVoice configuration reaches. |
| `cosyvoice/hifigan/generator.py` | `HiFTGenerator`, `SineGen`, `SourceModuleHnNSF` and `ResBlock` are the `CosyVoiceV1HiFTGenerator` family, `SineGen2` is `CosyVoiceV2SineGen` and `CausalHiFTGenerator` is `CosyVoiceV3HiFTGenerator`. |
| `cosyvoice/hifigan/f0_predictor.py` | `ConvRNNF0Predictor` is `CosyVoiceV1F0Predictor` and `CausalConvRNNF0Predictor` is `CosyVoiceV3F0Predictor`. |
| `cosyvoice/transformer/encoder.py`, `encoder_layer.py`, `attention.py`, `embedding.py`, `subsampling.py`, `positionwise_feed_forward.py` | Together they are `CosyVoiceV1Encoder` and its parts. The released configurations select one class per file, `ConformerEncoder`, `ConformerEncoderLayer`, `RelPositionMultiHeadedAttention`, `EspnetRelPositionalEncoding`, `LinearNoSubsampling` and `PositionwiseFeedForward`. The others are WeNet variants no CosyVoice configuration selects and no released checkpoint carries weights for. |
| `cosyvoice/transformer/upsample_encoder.py` | `UpsampleConformerEncoder` is `CosyVoiceV2UpsampleEncoder`, `PreLookaheadLayer` is `CosyVoiceV2PreLookaheadLayer` and `Upsample1D` is `CosyVoiceV2Upsample1D`. |
| `cosyvoice/transformer/convolution.py` | `CausalConv1d`, `CausalConv1dUpsample` and `CausalConv1dDownSample` are `CosyVoiceV3CausalConv1d`, `CosyVoiceV3CausalConv1dUpsample` and `CosyVoiceV3CausalConv1dDownsample`. `ConvolutionModule` is never built: every CosyVoice encoder sets `use_cnn_module: False`. |
| `cosyvoice/transformer/activation.py` | `Snake` is `CosyVoiceV1Snake`. `Swish` is a fallback `COSYVOICE_ACTIVATION_CLASSES` reaches only on a torch without `nn.SiLU`. |
| `cosyvoice/transformer/label_smoothing_loss.py` | `LabelSmoothingLoss` is `CosyVoiceV1LabelSmoothingLoss`. |
| `cosyvoice/transformer/decoder.py`, `decoder_layer.py` | None, and none is needed. `TransformerDecoder`, `BiTransformerDecoder` and `DecoderLayer` are WeNet's attention rescoring decoder for speech recognition. Nothing in the upstream tree imported them outside each other, no CosyVoice configuration names them, and the three conversions load `llm.pt`, `flow.pt` and `hift.pt` with zero UNEXPECTED keys, so no released checkpoint carries their weights. |
| `cosyvoice/utils/mask.py` | `make_pad_mask` is `make_pad_mask`, and `add_optional_chunk_mask` is `build_attention_bias` in `modeling_cosyvoice_v1.py` and `build_chunk_mask` in `modeling_cosyvoice_v3.py`. |
| `cosyvoice/utils/common.py` | `ras_sampling`, `nucleus_sampling`, `random_sampling` and `fade_in_out` are in `generation_cosyvoice_v1.py`, `mask_to_bias` is folded into `build_attention_bias`, and `th_accuracy` is `CosyVoiceV1Output.accuracy`. `TrtContextWrapper` is a TensorRT wrapper. |
| `cosyvoice/utils/onnx.py` | `SpeechTokenExtractor` is `CosyVoiceV1Processor.encode_speech_tokens` and `EmbeddingExtractor` is `CosyVoiceV1Processor.encode_speaker`. |
| `cosyvoice/utils/class_utils.py` | `configuration_cosyvoice_v1.py`. It is the string to class dispatch table hyperpyyaml used to build a model from `cosyvoice.yaml`. It had already stopped importing, since it reads `cosyvoice.llm.llm` and `cosyvoice.cli.model`, both moved in `7e93ecff`. |
| `cosyvoice/utils/file_utils.py` | `load_wav` is the processor's load and resample. `read_lists` and `read_json_lists` are manifest readers, and `convert_onnx_to_trt` and `export_cosyvoice2_vllm` are serving exports. |
| `cosyvoice/cli/__init__.py`, `cosyvoice/dataset/__init__.py`, `cosyvoice/transformer/__init__.py`, `cosyvoice/utils/__init__.py` | `__init__.py`. All four are empty. |

### Removed as out of scope

| Upstream path | Category |
|---|---|
| `runtime/`, 55 files | Serving stacks: a FastAPI server, a gRPC server and a Triton model repository backed by TensorRT-LLM, with their Dockerfiles, clients and conversion scripts. `runtime/triton_trtllm/token2wav*.py` are that stack's own token to waveform paths, which `generation_cosyvoice_v1.py`, `generation_cosyvoice_v2.py` and `generation_cosyvoice_v3.py` cover, and `scripts/convert_cosyvoice3_to_hf.py` merges the speech embedding into the Qwen2 table so `trtllm-build` can read it, which is not the layout `weight_conversion.py` produces. |
| `examples/`, 37 files | Training recipes: the LibriTTS and MagicData shell pipelines, DeepSpeed stage configurations, data preparation scripts, and the GRPO recipe under `examples/grpo/cosyvoice2/`, whose reward in `reward_tts.py` is an HTTP call to the Triton server above. `local/prepare_reject_sample.py` generates the rejected half of a preference batch by running `inference_zero_shot`, which `CosyVoiceV2ForConditionalGeneration.generate` covers. |
| `tools/`, 3 files | Data preparation: `extract_embedding.py` and `extract_speech_token.py` run `campplus.onnx` and the speech tokenizer graph over a manifest, which `CosyVoiceV1Processor.encode_speaker` and `encode_speech_tokens` do per utterance, and `make_parquet_list.py` writes the parquet shards. |
| `cosyvoice/vllm/cosyvoice2.py` | Serving: a vLLM plugin registering `CosyVoice2ForCausalLM`. |
| `cosyvoice/bin/`, 4 files | Training and export tooling. `train.py` is upstream's training entry point, read for the objectives above; `average_model.py` averages checkpoints; `export_jit.py` and `export_onnx.py` are export scripts. |
| `cosyvoice/dataset/dataset.py`, `cosyvoice/utils/executor.py`, `train_utils.py`, `scheduler.py` | Training harness: an `IterableDataset` with its distributed sampler, the train and validation loop, distributed and optimizer setup, and the learning rate schedules. `transformers` supplies all four. |
| `webui.py`, `example.py`, `vllm_example.py` | Demo scripts. Their counterpart is the "Usage" section of this README and of the v2 and v3 ones. All three had already stopped importing, since they read `cosyvoice.cli.cosyvoice`, moved in `7e93ecff`. |
| `docker/Dockerfile` | Packaging. |
| `.github/`, `.gitignore`, `.gitmodules` | Continuous integration and repository configuration. |
| `INFO.md`, `FAQ.md`, `CODE_OF_CONDUCT.md` | Upstream documentation. `INFO.md` is upstream's own README. |
| `asset/`, 3 files | Demo assets: the two prompt waveforms `example.py` and `webui.py` read, and a group chat QR code. |
| `third_party/Matcha-TTS` | A gitlink to `shivammehta25/Matcha-TTS`, never checked out here. What CosyVoice imports from it, `SinusoidalPosEmb`, `Block1D`, `ResnetBlock1D`, `Downsample1D`, `Upsample1D`, `BASECFM` and `BasicTransformerBlock`, is inlined in `modeling_cosyvoice_v1.py`. Its `matcha.hifigan.models` losses are not; see "Not carried over from upstream". |

### Still here, because nothing migrated them

| Upstream file | Why it stays |
|---|---|
| `cosyvoice/hifigan/hifigan.py`, `cosyvoice/hifigan/discriminator.py` | The vocoder's adversarial objective and its discriminators. Open. |
| `cosyvoice/utils/losses.py` | `tpr_loss` and `mel_loss` belong to that objective, and `DPOLoss` to the direct preference optimization recipe. Both are open. |
| `cosyvoice/dataset/processor.py` | `compute_f0` builds `pitch_feat`, the target of that objective's f0 term. Everything else in the file is accounted for: `compute_fbank` is `CosyVoiceV1FeatureExtractor`, `compute_whisper_fbank` is the processor's `WhisperFeatureExtractor`, `tokenize` is the processor's tokenizer, `parse_embedding` is the `speaker_embedding` input, and `parquet_opener` through `padding` is a data pipeline and a collator. |
| `cosyvoice/utils/frontend_utils.py` | The text frontend. Open. |
| `cosyvoice/tokenizer/tokenizer.py`, `cosyvoice/tokenizer/assets/multilingual_zh_ja_yue_char_del.tiktoken` | `get_encoding` and `get_tokenizer`, together with that asset, are the `CosyVoice-300M-25Hz` text tokenizer, which has no counterpart. Open. The rest of the file does have one: `get_qwen_tokenizer`, `CosyVoice2Tokenizer` and `CosyVoice3Tokenizer` are `SPECIAL_TOKENS` and `CosyVoiceV3Processor.add_special_tokens` in `processing_cosyvoice_v3.py`, which v3's `weight_conversion.py` applies. |

`LICENSE` and `requirements.txt.bak` also stay, which is what every finished model folder in this
repository does with the upstream licence and dependency manifest.

## Repository integration

Three things outside this folder are still needed and were deliberately not touched:

- `voicestudio/models/__init__.py` does not import this folder. It needs a
  `from .cosyvoice_v1 import *` line in its alphabetical list, between `chroma` and `dia`.
- `PROJECT.md`'s status table still records this model as not started.
- `pyproject.toml` needs no change. Nothing new is imported; `onnxruntime` is deliberately absent
  and the processor raises rather than depending on it.
