# Spark TTS

Spark-TTS is a text-to-speech model in which synthesis, voice cloning and attribute control are all next-token
prediction over a single flat sequence. A Qwen2.5-0.5B decoder whose vocabulary is extended to 166000 entries emits
BiCodec tokens directly, so no flow-matching or diffusion stage follows it.

BiCodec, the audio tokenizer whose tokens those are, is a model of its own and lives in
[`voicestudio/models/spark_tts_bicodec`](../spark_tts_bicodec). It splits speech into a time-varying semantic stream
at 50 tokens per second and a time-invariant global stream of 32 speaker tokens, and turns either back into a
waveform at 16 kHz.

Original model and code: [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS)


## Usage

```python
import soundfile as sf
import torch

from voicestudio.models.spark_tts import SparkTTSForConditionalGeneration, SparkTTSProcessor

model_id = "SparkAudio/Spark-TTS-0.5B"

processor = SparkTTSProcessor.from_pretrained(model_id)
model = SparkTTSForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32).to("cuda")
processor.audio_tokenizer.to(model.device)
```

The published repo is three independently saved models in three subfolders plus two YAML files, none of which
`from_pretrained` can read on its own, so a first load converts it into a directory under `HF_HOME`, keyed on the
repository and the commit it resolved to. Later loads of the same checkpoint find that conversion and reuse it.
Once it is written, the three models it read are dropped from the `huggingface_hub` cache; the four configuration
files that name the revision stay, and a later load resolves those and nothing else.
`weight_conversion.convert` is the same conversion under a name that takes an explicit output directory.

Passing `reference_audio` clones the voice of that clip by prefixing its global tokens to the prompt:

```python
reference, sampling_rate = sf.read("reference.wav")

inputs = processor(
    text="The quick brown fox jumps over the lazy dog.",
    reference_audio=reference,
    sampling_rate=sampling_rate,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=3000, do_sample=True, top_k=50, top_p=0.95, temperature=0.8)
waveform = processor.decode(generated, input_length=inputs["input_ids"].shape[-1])
sf.write("cloned.wav", waveform.numpy(), processor.feature_extractor.sampling_rate)
```

Passing `gender`, `pitch` and `speed` instead builds a voice from attribute labels, in which case the model emits its
own global tokens and `decode` reads them back out of the continuation:

```python
inputs = processor(
    text="The quick brown fox jumps over the lazy dog.",
    gender="female",
    pitch="moderate",
    speed="moderate",
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=3000, do_sample=True, top_k=50, top_p=0.95, temperature=0.8)
waveform = processor.decode(generated, input_length=inputs["input_ids"].shape[-1])
```

`pitch` and `speed` take `very_low`, `low`, `moderate`, `high` or `very_high`; `gender` takes `female` or `male`.

Passing `prompt_text`, the transcript of the reference clip, turns voice cloning into prompt continuation: the
transcript goes in front of the text to speak inside `<|start_content|>`, the clip's own semantic tokens are appended
after `<|start_global_token|>`, and the model continues the clip instead of starting from silence.

```python
inputs = processor(
    text="Actions speak louder than words.",
    reference_audio=reference,
    sampling_rate=sampling_rate,
    prompt_text="Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.",
).to(model.device)
```

The processor separates the two texts, and this is a deliberate divergence from upstream: `cli/SparkTTS.py`
concatenates them verbatim, so a transcript ending in `gospel.` and a text starting with `Actions` meet inside one
BPE token, `.Actions` (id 72044), where the same word after a space is `ĠActions` (id 26722). The model never saw
the fused form: SparkVox's `egs/speech_synthesis/spark-tts/scripts/prepare_train.py` puts a single utterance's
transcript between the content markers and never concatenates a reference transcript, so the sentence-boundary
tokens fused this way are out of distribution. Measured on `librispeech_1272-128104-0000.wav` with that transcript,
`"Actions speak louder than words."` is 0 of 8 seeds verbatim under `openai/whisper-large-v3-turbo` with the two
joined directly, five seeds dropping the leading word and three collapsing into a filler syllable, one of them
running 3000 tokens into the `max_new_tokens` ceiling. The same sentence with one space between them is 8 of 8. How
badly the fused form fails depends on which token comes out: `.The` is common enough that the same clip and the same
reference transcript give 7 of 8 verbatim without a separator. A reader diffing this repository's audio against
upstream's should expect the two to part company here.

The separator is a space only where the script uses one. Chinese never fuses across the join, because the tokenizer
carries no merge spanning a CJK boundary: `今天天气真不错。` followed by `行动胜于雄辩。` tokenizes as
`。`(1773), `行动`(100675) whether the two are joined directly or not. A space in front of a Chinese character is
what does damage there, since `Ġ行动` does not exist and the tokenizer falls back to the partial-UTF-8 pair
`Ġè¡`(77407), `Į`(234). So the join looks at the text being appended: a space goes in unless that text opens on a
Chinese character, which also gets the mixed cases right, a space before an English sentence following `。` and none
before a Chinese one following `classes.`.

BiCodec is usable on its own through [`SparkTTSBiCodecModel`](../spark_tts_bicodec), which is what
`processor.audio_tokenizer` holds:

```python
inputs = processor.feature_extractor(reference, sampling_rate=sampling_rate).to(model.device)

codes = processor.audio_tokenizer.encode(**inputs)
audio = processor.audio_tokenizer.decode(codes.audio_codes, codes.global_codes).audio_values
```


## Training

### Language model

`SparkTTSForConditionalGeneration` is `Qwen2ForCausalLM` over the extended vocabulary, and its objective is that
class's cross entropy on `labels`, which is exactly what upstream optimizes: `sparkvox/models/speech_synthesis/
sparktts/models/qwen.py` forwards `input_ids`, `attention_mask` and `labels` into `AutoModelForCausalLM` and returns
its `loss` unchanged. Nothing is frozen; the whole 0.5B decoder is fine-tuned after
`resize_token_embeddings(166000)`.

`SparkTTSProcessor` builds the supervised pair when `output_labels=True`:

```python
inputs = processor(
    text="The quick brown fox jumps over the lazy dog.",
    reference_audio=reference,
    sampling_rate=sampling_rate,
    output_labels=True,
)

loss = model(**inputs).loss
```

The prompt and its continuation are concatenated into one sequence, padded on the side the tokenizer is configured
for, and `labels` is `-100` over the padding and over the whole prompt, which is what upstream's
`DataCollatorForCausalLM` produces. Two details of that collator are worth naming. It pins `padding_side="left"`
rather than reading it off the tokenizer, so reproducing it exactly means loading the tokenizer with
`padding_side="left"`; the converted checkpoint's own default is `"right"`. And it overwrites the last label of
every row with the end-of-sequence id instead of appending a token, which for this tokenizer is a no-op, since the
continuation already ends in `<|im_end|>` and that is the end-of-sequence token.

The two layouts upstream trains jointly are both available: the voice-cloning layout supervises
`<|start_semantic_token|>` + semantic tokens + `<|im_end|>`, and the attribute layout, selected by also passing
`gender`/`pitch`/`speed` together with `pitch_value` and `speed_value`, supervises the acoustic value tokens, the
global tokens and the semantic tokens.

### BiCodec

`SparkTTSBiCodecModel.forward` returns a `loss` that is the weighted sum

```
1.0 * vq_loss + 1.0 * feature_loss + 15.0 * mel_loss + 1.0 * speaker_loss
```

with each term also reported on its own in `SparkTTSBiCodecOutput`. The weights come from the `loss_lambdas` block of
`egs/codec/bicodec/config/bicodec.16k.yaml` in [SparkAudio/SparkVox](https://github.com/SparkAudio/SparkVox), which
is where the codec is actually trained; the inference-only `SparkAudio/Spark-TTS` repo ships no trainer. Term by
term, against `sparkvox/models/codec/BiCodec/lightning_models/bicodec.py`:

- `vq_loss` (upstream `vq_loss`, weight 1.0) is the semantic quantizer's own
  `commitment * MSE(latent, quantized.detach()) + codebook_loss_weight * MSE(quantized, latent.detach())`, averaged
  over the batch. Both operands live in the factorized `codebook_dim` space, after `in_proj` and before `out_proj`,
  and are the un-normalized vectors; the L2 normalization applies only to the nearest-neighbour search.
- `feature_loss` (upstream `mse_loss`, weight 1.0) is `MSELoss` between the postnet's prediction and the
  self-supervised features the semantic encoder consumed. The prediction is read before the speaker embedding is
  added back to the prenet output, so that residual is on the wave generator's branch only.
- `mel_loss` (upstream `mel_loss`, weight 15.0) is `MultiResolutionMelSpectrogramLoss`, the sum over seven
  resolutions of the mean absolute error between base-ten log mel magnitudes clamped at `1e-5`. Each resolution
  takes its window length as the transform size and a quarter of it as the hop. It covers the whole spectrum:
  upstream stores `mel_fmin` and `mel_fmax` on the loss and never passes them to the transform, so the filters run
  from 0 Hz to the Nyquist frequency rather than from the `mel_fmin` of 10 Hz the codec's own mel spectrogram uses.
  The term is computed only when `labels`, the target waveform, is passed;
  `SparkTTSFeatureExtractor(..., return_labels=True)` returns it.
- `speaker_loss` (upstream `speaker_loss`, weight 1.0) is `MSELoss(speaker_embedding.detach(), conditioning_embedding)`,
  added only on training steps past `config.d_vector_train_start`.

That last one is a schedule, not a switch: up to and including step `d_vector_train_start` (1000) the prenet and the
wave generator are conditioned on the unquantized ECAPA embedding and no speaker term is added, and past it they are
conditioned on the quantized one and the term turns on. `forward` reproduces this from its `step` argument.
Evaluation always takes the quantized branch with no speaker term. Because the speaker term detaches the ECAPA
embedding, the pooling and output head that produce it stop receiving gradient once the schedule flips, which is the
upstream behaviour.

Upstream freezes exactly one thing: the `wav2vec2-large-xlsr-53` feature source, which
`sparkvox/models/codec/BiCodec/modules/wav2vec.py` puts in eval mode, sets `requires_grad = False` on, and calls
under `torch.no_grad()`. It is also structurally unoptimizable there: it is assigned outside the `nn.ModuleDict`
that `WavCodec.configure_optimizers` builds its parameter groups from, so no optimizer ever sees it. Nothing else is
frozen; the encoder, quantizer, speaker encoder, prenet, postnet and wave generator all train from step 0.
`SparkTTSBiCodecModel.freeze_semantic_model` reproduces all three of eval mode, `requires_grad = False` and the
`torch.no_grad()` call, and runs on construction and after `from_pretrained`.

Eval mode is the one part of that freeze which does not survive a `train()`, upstream or here, because
`nn.Module.train()` recurses into every child. Upstream that happens on the first validation pass, where
`BaseModel.on_validation_end` calls `self.train()` and the self-supervised model's own dropout, layer drop and
spectrogram masking come back on for the rest of training with `requires_grad = False` still holding, so the frozen
features become stochastic without any gradient reaching them. `freeze_semantic_model` is public so that a training
loop that wants the deterministic features can call it again after each `train()`; calling it is a divergence from
upstream and not calling it is not, so the choice is left to the caller.

The convolutions upstream trains under weight normalization are plain in the checkpoint, as in `DacModel`;
`apply_weight_norm()` restores the parametrization and `remove_weight_norm()` folds it back.


## Not carried over from upstream

Recorded per CLAUDE.md section 2.6. None of these is resolved here.

- **The adversarial and feature-matching terms of the codec objective.** Upstream's full generator loss is
  `1.0 * mse + 15.0 * mel + 1.0 * adv + 2.0 * feature_map + 1.0 * vq` plus `1.0 * speaker`, and a discriminator is
  stepped before the generator on every batch under manual optimization, with an LSGAN objective
  (`sum(mean(d_fake ** 2) + mean((1 - d_real) ** 2))` for the discriminator, `sum(MSE(d_fake, 1))` for the
  generator) and an L1 feature-matching term over every intermediate feature map, with the real side detached. That
  discriminator (`sparkvox/models/codec/base/modules/wave_discriminator_dac.py`) is eight sub-discriminators: five
  multi-period at periods 2, 3, 5, 7 and 11, and three multi-resolution STFT over five bands. Its `rates` list is
  left unset, so the multi-scale branch is never constructed. Both loss terms sum over the eight rather than
  averaging. No published checkpoint carries any of it, it pulls in `descript-audiotools` and `julius`, and it is
  not implemented here. `SparkTTSBiCodecOutput.loss` therefore omits `adv_loss` and `feature_map_loss`. Because
  upstream uses manual optimization, `accumulate_grad_batches` must be 1 there.
- **`ssim_loss`.** `loss_lambdas` gives it a weight of 1, but `compute_generator_loss` never puts an `ssim_loss` key
  into its loss dict and the sum filters on `if k in loss_dict`, so the term never contributes. It is dead
  configuration upstream and has no counterpart here.
- **`codebook_loss_weight` disagrees between the two sources.** SparkVox's `bicodec.16k.yaml` sets 4.0, the released
  `BiCodec/config.yaml` sets 2.0. `weight_conversion` reads the checkpoint's own value, so a converted model carries
  2.0. `commitment` is 0.25 in both.
- **The quantizer term is a real number at evaluation here and `NaN` upstream.** Upstream's
  `FactorizedVectorQuantize.forward` computes the commitment and codebook terms only under `self.training` and
  returns `torch.zeros(0)` otherwise, so `(commit_loss + codebook_loss).mean()` is the mean of an empty tensor and
  the validation generator loss upstream is `NaN`. `SparkTTSFactorizedVectorQuantizer` computes both terms in either
  mode, which changes the reported evaluation loss but not a single training step.
- **`use_l2_normlize` and `momentum` are dead configuration.** The released `BiCodec/config.yaml` sets
  `use_l2_normlize: True`, but `FactorizedVectorQuantize.__init__` has no such parameter and swallows it through
  `**kwargs`; the L2 normalization in `decode_latents` is unconditional. `momentum` is stored and never read.
  `SparkTTSBiCodecConfig` carries neither, and `weight_conversion` does not read them.
- **The codec's training-time data pipeline.** `sparkvox/models/codec/BiCodec/dataloaders/wav_XLSR_dataset.py` feeds
  the self-supervised model a window one second wider on each side than the reconstruction target and crops 50
  frames off each end of the features, and it draws the reference clip as a random 6 second crop of the utterance
  with the target segment zeroed out to stop content leaking. It also high-pass filters at 40 Hz and rounds the
  segment to a multiple of four hop lengths. `SparkTTSFeatureExtractor` does none of this: it takes the leading
  `ref_segment_duration` seconds as the reference clip, exactly as upstream inference does, and passes the same clip
  to both branches.
- **The released `BiCodec/config.yaml` records none of the training configuration.** It has no
  `d_vector_train_start`, no discriminator block and no `loss_lambdas`, so every value in the section above comes
  from SparkVox rather than from the checkpoint. `SparkTTSBiCodecConfig` carries SparkVox's values as defaults.
- **Dropping over-length training examples.** Upstream's collator skips any example whose prompt and continuation
  exceed `max_length` (1500 in the released config) rather than truncating it. `SparkTTSProcessor` neither drops nor
  truncates.
- **Task tokens other than TTS and controllable TTS.** The tokenizer carries `<|task_vc|>`, `<|task_asr|>`,
  `<|task_s2s|>`, `<|task_t2s|>`, `<|task_understand|>`, `<|task_cap|>`, `<|task_prompt_tts|>` and `<|task_edit|>`,
  and `TASK_TOKENS` still names them, but no upstream code path builds a prompt for any of them.
- **Age, emotion and pitch-variance attributes.** `sparkvox/utils/attribute_parser.py` and `prepare_train.py` derive
  `<|age_*|>`, `<|emotion_*|>` and `<|pitch_var_*|>` tokens, then keep only gender, pitch and speed when building the
  prompt. `SparkTTSProcessor` exposes the three that survive.
- **The unused halves of two upstream modules.** `VocosResNetBackbone` and `ResBlock1` in
  `sparktts/modules/blocks/vocos.py`, `GroupedResidualFSQ` in `sparktts/modules/fsq/residual_fsq.py`, and every
  pooling layer in `sparktts/modules/speaker/pooling_layers.py` except `ASTP` are never instantiated by any
  configuration and have no counterpart here. The same goes for the quantize-dropout path of `ResidualFSQ`, which
  the speaker encoder disables.
- **Per-module weight initialization.** Upstream applies a truncated normal with standard deviation 0.02 and a zero
  bias only inside the Vocos backbones (`VocosBackbone._init_weights`, over their `Conv1d` and `Linear` layers) and
  to the `Conv1d` layers of the wave generator (`layers.init_weights`), leaving the ECAPA-TDNN encoder, the
  perceiver resampler, the quantizers and every `ConvTranspose1d` on PyTorch's own defaults.
  `SparkTTSBiCodecPreTrainedModel._init_weights` is a single rule over the whole tree instead, as `transformers`
  models are written. The largest single consequence is the semantic codebook: upstream leaves the `nn.Embedding`
  on PyTorch's `N(0, 1)` and the migration draws it from `N(0, 0.02)`. This only affects training from a random
  initialization; loading a checkpoint overwrites all of it.
- **Independent prenet and postnet channel widths.** Upstream's `Decoder` takes `input_channels` and `out_channels`
  per instance, so in principle the postnet's output width, which is the self-supervised feature width, is
  unrelated to the codec latent width. `SparkTTSSemanticDecoder` ties all four to `config.hidden_size`, which also
  makes `feature_loss` require `semantic_model_config.hidden_size == hidden_size`. Every one of those is 1024 in the
  released checkpoint, so nothing about it changes, but a checkpoint that set them apart could not be expressed.
- **The upstream CLI, Gradio app and Triton/TensorRT-LLM serving tree.** `cli/inference.py`, `webui.py` and
  `runtime/triton_trtllm/` are dropped with no counterpart, along with `gradio` and the Triton client dependencies.


## Every removed upstream file, and where it went

Sixty-five files were removed from this folder over the migration, plus the folder's own `LICENSE`. Each one is
accounted for below: the first list is the code that has a named counterpart, the second is everything removed with
no counterpart at all.

Everything BiCodec is now under [`voicestudio/models/spark_tts_bicodec`](../spark_tts_bicodec), so the paths below
that end in `modeling_spark_tts_bicodec.py` name a file in that folder.

- `sparktts/models/bicodec.py` was renamed to `modeling_spark_tts_bicodec.py`; `BiCodec` is `SparkTTSBiCodecModel`,
  and its mel transform moved into `SparkTTSFeatureExtractor`.
- `sparktts/models/audio_tokenizer.py` was renamed to `feature_extraction_spark_tts.py`; `BiCodecTokenizer`'s audio
  loading, reference clip selection and wav2vec2 feature extraction became `SparkTTSFeatureExtractor` and
  `SparkTTSBiCodecModel.extract_semantic_features`.
- `cli/SparkTTS.py` was renamed to `processing_spark_tts.py`; `SparkTTS.process_prompt`,
  `process_prompt_control` and the token parsing half of `inference` became `SparkTTSProcessor.__call__` and
  `SparkTTSProcessor.decode`.
- `sparktts/utils/file.py` was renamed to `weight_conversion.py`; `load_config` became the YAML reading inside
  `convert`, and the JSONL/CSV metadata helpers are dropped.
- `sparktts/modules/encoder_decoder/feat_encoder.py` `Encoder` is `SparkTTSSemanticEncoder`, and
  `feat_decoder.py` `Decoder` is `SparkTTSSemanticDecoder`, used for both the prenet and the postnet.
- `sparktts/modules/encoder_decoder/wave_generator.py` `WaveGenerator` and `DecoderBlock` are
  `SparkTTSWaveGenerator` and `SparkTTSWaveGeneratorBlock`.
- `sparktts/modules/blocks/vocos.py` `ConvNeXtBlock`, `AdaLayerNorm` and `VocosBackbone` are
  `SparkTTSConvNeXtBlock`, `SparkTTSAdaLayerNorm` and `SparkTTSVocosBackbone`.
- `sparktts/modules/blocks/samper.py` `SamplingBlock` is `SparkTTSSamplingBlock`, wrapped with its trailing backbone
  in `SparkTTSResamplingLayer`.
- `sparktts/modules/blocks/layers.py` `Snake1d` and `ResidualUnit` are `transformers` `Snake1d` and
  `DacResidualUnit`, `WNConv1d`/`WNConvTranspose1d` became `SparkTTSBiCodecModel.apply_weight_norm`, and
  `init_weights` became `SparkTTSBiCodecPreTrainedModel._init_weights`.
- `sparktts/modules/vq/factorized_vector_quantize.py` `FactorizedVectorQuantize` is
  `SparkTTSFactorizedVectorQuantizer`.
- `sparktts/modules/fsq/finite_scalar_quantization.py` `FSQ` is `SparkTTSFiniteScalarQuantizer`, and
  `residual_fsq.py` `ResidualFSQ` is `SparkTTSResidualFiniteScalarQuantizer`.
- `sparktts/modules/speaker/speaker_encoder.py` `SpeakerEncoder` is `SparkTTSSpeakerEncoder`.
- `sparktts/modules/speaker/ecapa_tdnn.py` `ECAPA_TDNN`, `SE_Res2Block`, `Res2Conv1dReluBn`, `Conv1dReluBn` and
  `SE_Connect` are `SparkTTSEcapaTdnn`, `SparkTTSSERes2Block`, `SparkTTSRes2Conv1dReluBn`, `SparkTTSConv1dReluBn`
  and `SparkTTSSqueezeExcite`.
- `sparktts/modules/speaker/pooling_layers.py` `ASTP` is `SparkTTSAttentiveStatisticsPooling`.
- `sparktts/modules/speaker/perceiver_encoder.py` `PerceiverResampler`, `Attention`, `FeedForward` and `RMSNorm` are
  `SparkTTSPerceiverResampler`, `SparkTTSPerceiverAttention`, `SparkTTSPerceiverFeedForward` and
  `SparkTTSPerceiverRMSNorm`, with `Attend` replaced by `torch.nn.functional.scaled_dot_product_attention`.
- `sparktts/utils/audio.py` `audio_volume_normalize` is `SparkTTSFeatureExtractor.volume_normalized` and the
  reference clip tiling is `SparkTTSFeatureExtractor.reference_clip`; `load_audio` is the caller's job.
- `sparktts/utils/token_parser.py` `TASK_TOKEN_MAP`, `LEVELS_MAP` and `GENDER_MAP` are `TASK_TOKENS`, `LEVELS` and
  `GENDERS` in `processing_spark_tts.py`.

Removed with no counterpart, none of which carries model behaviour:

- `cli/inference.py`, `webui.py` and the sixteen files of `runtime/triton_trtllm/` are the command line entry
  point, the Gradio app and the Triton plus TensorRT-LLM serving tree. They are listed as an open gap above, since
  dropping them drops functionality rather than only packaging.
- `sparktts/utils/__init__.py` was an empty file marking `sparktts.utils` as a package. There is no `sparktts`
  package any more.
- `sparktts/utils/parse_options.sh` is the Kaldi command line option parser that `example/infer.sh` sourced. It
  parses shell arguments and has nothing to do with the model.
- `example/infer.sh`, `example/prompt_audio.wav` and `example/results/20250225113521.wav` are the demo invocation,
  its Chinese reference clip and its recorded output. The `## Usage` section above replaces the script; the two clips
  are sample data, and the migration ships none.
- `src/demos/` (ten reference clips), `src/figures/` (four screenshots) and `src/logo/` (ten institution and project
  logos) are the assets `INFO.md` embedded.
- `INFO.md` was the upstream repository's own README, kept under a second name once this folder took `README.md`
  over. Its content is the upstream project page, not documentation of this code; the link at the top of this file
  points at it.
- `.gitignore` was the upstream repository's Python `.gitignore`, redundant with the one at this repository's root.
- `requirements.txt.bak` pinned the upstream dependency set. The `## Dependencies` section below is the accounting
  of what happened to each of its entries.
- `LICENSE` was a copy of the Apache 2.0 text. `transformers` carries a model's license as a header in
  `modeling_<model>.py`, which both `modeling_spark_tts.py` and `modeling_spark_tts_bicodec.py` have, and keeps
  one `LICENSE` at the repository root.


## Dependencies

The migration removes `einops`, `einx`, `omegaconf`, `soundfile`, `soxr`, `gradio` and `tqdm` from what this model
needs, and adds nothing. `einops` and `einx` are not installed in this environment and were not installed to do the
work; every `rearrange`, `pack`, `unpack`, `reduce` and `get_at` call in the FSQ, the perceiver resampler and the
quantizer is now a plain `view`/`reshape`/`transpose`/`stack`. `omegaconf` is replaced by `yaml` inside
`weight_conversion`, and audio file I/O is left to the caller. What remains is `torch`, `torchaudio`,
`transformers`, `numpy`, `safetensors` and `huggingface_hub`, every one of them already a `pyproject.toml`
dependency, plus `pyyaml`, which `transformers` and `huggingface_hub` both require and which only
`weight_conversion` imports. `pyproject.toml` and `uv.lock` need no change.

Nothing outside `weight_conversion` reads YAML: the published checkpoint's two YAML files are parsed once, offline,
into `SparkTTSConfig` and `SparkTTSBiCodecConfig`, and the model, the config and the processor only ever see JSON.


## Repository integration

Two things outside this folder are still needed and were deliberately not touched:

- `voicestudio/models/__init__.py` needs a `from .spark_tts import *` line and a
  `from .spark_tts_bicodec import *` line. Importing `spark_tts` alone already registers the codec, since it
  imports it, so the second line only matters for `from voicestudio.models import SparkTTSBiCodecModel`.
- `PROJECT.md` needs a Spark-TTS status entry carrying the gaps listed above.
