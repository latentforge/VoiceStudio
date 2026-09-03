# BigVGAN

BigVGAN is a neural vocoder that upsamples a log mel spectrogram to a waveform in the time domain. A stack of
transposed convolutions raises the frame rate by the product of `upsample_rates`, and after each one a set of
parallel residual blocks with different kernel sizes is averaged. What separates it from the HiFi-GAN it descends
from is the activation: instead of a leaky ReLU, every residual layer applies a periodic snake nonlinearity,
`x + sin(alpha * x) ** 2 / beta`, with a learned `alpha` and `beta` per channel, and it applies it inside an
anti aliasing sandwich. The input is upsampled two times by a Kaiser windowed sinc filter, the nonlinearity runs
at that higher rate, and the result is lowpass filtered back down, so the harmonics the nonlinearity creates stay
below the Nyquist frequency rather than folding back into the signal.

Original model and code: [NVIDIA/BigVGAN](https://github.com/NVIDIA/BigVGAN)


## Usage

`from_pretrained` takes any published repository id as it stands:

```python
import soundfile as sf

from voicestudio.models.bigvgan import BigVGANFeatureExtractor, BigVGANModel

model_id = "nvidia/bigvgan_v2_24khz_100band_256x"

model = BigVGANModel.from_pretrained(model_id).eval()
extractor = BigVGANFeatureExtractor.from_pretrained(model_id)

waveform, sampling_rate = sf.read("reference.wav", dtype="float32")
inputs = extractor(waveform, sampling_rate=sampling_rate)
audio = model(**inputs).audio_values

sf.write("output.wav", audio[0].numpy(), model.config.sampling_rate)
```

A published repository is a `config.json` of training hyperparameters and a `bigvgan_generator.pt` holding the
generator's weight normalized state dict, neither of which `PreTrainedModel.from_pretrained` can read on its own,
so `BigVGANModel.from_pretrained` converts them on the way in. Any key of `PUBLISHED_CHECKPOINTS`, a repository
id, or a local directory holding those two files works as `model_id`, as does a directory
`weight_conversion.convert` wrote.

Three things about loading and the front end are load-bearing:

- `weights_name` selects the generator file inside a published repository. It defaults to `bigvgan_generator.pt`,
  which is what upstream's own loader reads; pass `weights_name="bigvgan_generator_3msteps.pt"` for the longer
  trained weights the v2 repositories also publish.
- `BigVGANFeatureExtractor` scales each waveform so that its largest absolute sample is 0.95 before framing it,
  because `normalize_volume` is what upstream's dataset and its demo both do to every clip they take a mel
  spectrogram of. Pass `normalize_volume=False` to turn it off, but then the mel is on a scale the checkpoint was
  not trained on.
- `attention_mask`, which the extractor returns for a padded batch, is not a model argument. The stack is fully
  convolutional, so padded frames become padded samples that the caller has to trim.

A first load converts the published layout into a directory under `HF_HOME`, keyed on the repository and the commit it
resolved to, and later loads read that directory through the ordinary loading path instead of converting again.

`weight_conversion.convert` writes that same conversion to a directory of the caller's choosing, for a checkpoint
that is shipped elsewhere or kept outside the cache:

```python
from voicestudio.models.bigvgan.weight_conversion import convert

convert("v2_24khz_100band_256x", "bigvgan-v2-24khz-converted")
```


## Training

Pass the ground truth waveform as `labels` and the standard `forward` returns a loss:

```python
outputs = model(input_features=inputs["input_features"], labels=waveform)
outputs.loss.backward()
```

**What that loss is.** With `use_multiscale_mel_loss`, which every BigVGAN-v2 configuration sets, it is
`loss.py:MultiScaleMelSpectrogramLoss` as `train.py` constructs it, from the sampling rate alone and otherwise at
its defaults. Seven resolutions with window lengths 32, 64, 128, 256, 512, 1024 and 2048, each hopping by a
quarter of its own window and each with its own mel filterbank of 5, 10, 20, 40, 80, 160 and 320 channels over
the full band. Each resolution contributes the L1 distance between the base ten logarithms of the target's and
the generated waveform's mel spectrograms, clamped at 1e-5 first, and the seven are summed and multiplied by
`lambda_melloss`, which is 15 in every v2 configuration. The `mag_weight` term upstream adds alongside is
weighted 0.0 and adds nothing. Without `use_multiscale_mel_loss`, which is the BigVGAN-v1 case, it is instead a
single `F.l1_loss` between the natural log mel spectrograms of `n_fft`, `hop_length` and `win_length` at
`fmax_for_loss`, weighted by 45.

**What the full upstream objective is, term by term.** `train.py` is a generative adversarial network with two
optimizers alternating per step, and the reconstruction term above is one of five in the generator half.
`loss_gen_all` sums

- `loss_gen_f`, the least squares generator loss `mean((1 - D(y_hat)) ** 2)` summed over the five
  sub-discriminators of a `MultiPeriodDiscriminator` at periods 2, 3, 5, 7 and 11,
- `loss_gen_s`, the same over the second discriminator, which is a `MultiScaleSubbandCQTDiscriminator` at hop
  lengths 512, 256 and 256 over 9 octaves at 24, 36 and 48 bins per octave when `use_cqtd_instead_of_mrd` is set,
  a `MultiBandDiscriminator` when `use_mbd_instead_of_mrd` is set, and otherwise UnivNet's
  `MultiResolutionDiscriminator` at the `resolutions` the configuration lists, `[1024, 120, 600]`,
  `[2048, 240, 1200]` and `[512, 50, 240]` in every v1 configuration,
- `loss_fm_f`, the feature matching loss `mean(abs(f_real - f_generated))` summed over every layer of every
  sub-discriminator of the multi period discriminator and multiplied by 2,
- `loss_fm_s`, the same over the second discriminator,
- `loss_mel`, the reconstruction term above.

The discriminator half optimizes `loss_disc_s + loss_disc_f`, the least squares discriminator loss
`mean((1 - D(y)) ** 2) + mean(D(y_hat) ** 2)` summed over the sub-discriminators of each. Both halves are AdamW at
`learning_rate` 1e-4 and `betas` (0.8, 0.99) under an `ExponentialLR` with `lr_decay` 0.9999996, and both are
clipped to `clip_grad_norm`, which is 500 in the v2 configurations and defaults to 1000. `--freeze_step`, which
would skip the discriminator update and run the generator against `loss_mel` alone, defaults to 0, so the
adversarial terms are live from the first step.

**What upstream freezes.** Nothing in the generator. `optim_g` is given `generator.parameters()` whole, with no
`requires_grad` flag set anywhere in `bigvgan.py`, `train.py` or `discriminators.py`, and no submodule put into
evaluation mode during training. The two discriminators are separate models under their own optimizer, and the
only thing `--freeze_step` freezes is those, for its first N steps.


## Lineage

Nothing in `transformers` is a general BigVGAN. `univnet` and the `speecht5` HiFi-GAN are the two standalone
vocoders it ships, and both are leaky ReLU generators with no periodic activation and no anti aliasing
resampling. `vits` decodes with a HiFi-GAN of its own.

Two derived copies do exist, and neither is a base to inherit.
`qwen2_5_omni.Qwen2_5OmniToken2WavBigVGANModel` and
`qwen3_tts_tokenizer_single_codebook.Qwen3TTSTokenizerSingleCodebookDecoderBigVGANModel` are Qwen's own
specializations: their residual block hard codes exactly three dilations, the activation is `snakebeta` in log
scale with no `snake` alternative, the output is always clamped and never passed through a hyperbolic tangent,
`conv_post` is always built without a bias, `forward` begins with a decibel renormalization of the mel
spectrogram that only Qwen's own token to waveform path wants and ends by moving the result to the CPU, and the
Qwen3-TTS one additionally builds causal convolutions selected per upsampling layer. Inheriting either would make
the general vocoder depend on one model that uses it. Their `SnakeBeta`, `AMPBlock`, `Activation1d`, `UpSample1d`
and `DownSample1d` names are deprecated aliases that log a warning on construction.

Reuse at layer level was checked and rejected for the same reason: the pieces that would be worth reusing,
`Qwen2_5OmniSnakeBeta` and `Qwen2_5OmniAMPBlock`, are the two that carry those hard coded specializations, so
they are not line for line identical to NVIDIA's `SnakeBeta` and `AMPBlock1`. `kaiser_sinc_filter1d` is identical
in all three, and is the one piece that could have been imported, but it is a module private function of a model
file rather than an exported layer, and thirty lines of window arithmetic did not justify the dependency.

The folder layout follows `voicestudio/models/vocos`, which is the repository's other standalone vocoder, and
`models/univnet` in `transformers-tts`, which is the one standalone vocoder there that is a first class model
with a `configuration_`, a `modeling_` and a `feature_extraction_` file.

`voicestudio/models/prompt_tts_pp` now inherits this model. Upstream PromptTTS++'s `F0AwareBigVGAN` is BigVGAN
plus a neural source filter excitation path, so `PromptTTSPPBigVGan` inherits `BigVGANModel`,
`PromptTTSPPAmpBlock` inherits `BigVGANAmpBlock`, `PromptTTSPPAmpLayer` inherits `BigVGANAmpLayer` and
`PromptTTSPPSnakeActivation` inherits `BigVGANSnakeActivation`. Only the harmonic source module, the noise
convolutions that inject it into each upsampling stage, and the transposed convolution padding it needs for its
odd upsampling rates stay local.


## Dependencies

Upstream `requirements.txt` lists `torch`, `numpy`, `librosa>=0.8.1`, `scipy`, `tensorboard`, `soundfile`,
`matplotlib`, `pesq`, `auraloss`, `tqdm`, `nnAudio`, `ninja` and `huggingface_hub>=0.23.4`. This migration needs
`torch`, `torchaudio`, `transformers`, `safetensors`, `numpy` and `huggingface_hub`, all of which the repository
already carries. `pyproject.toml` and `uv.lock` need no change, and nothing was added.

- `librosa` supplied `librosa.filters.mel` for every mel filterbank and `librosa.util.normalize` for the volume
  normalization. `torchaudio.functional.melscale_fbanks` with `norm="slaney"` and `mel_scale="slaney"` is the
  same filterbank, and the peak normalization is one division. The Verification section measures both.
- `scipy` supplied `scipy.signal.get_window("hann", n)` for the multi scale loss windows, which is
  `torch.hann_window(n)`, and `scipy.io.wavfile.write` for saving audio.
- `nnAudio` supplied the constant Q transform of `DiscriminatorCQT`, and `auraloss` and `pesq` the validation
  metrics. They go with the discriminators and the validation loop.
- `ninja` built the optional fused CUDA kernel of the anti aliased activation, `matplotlib` and `tensorboard`
  were the training logs, and `tqdm` the progress bars.
- `soundfile` is a caller's concern, not the model's.


## Verification

`from_pretrained` straight onto the published `nvidia/bigvgan_v2_24khz_100band_256x` repository id, with no
conversion call before it, reports no missing, unexpected or mismatched keys across all 449 tensors and 112414512
parameters. The published checkpoint holds 783 tensors: 449 of them are these, 116 are the `weight_v` halves that
the conversion folds into the weight they share with a `weight_g`, and 218 are the resampling filters of the anti
aliased activations, which the model rebuilds from its configuration into the 109 `filter` buffers its 109 anti
aliased activations hold, two upstream tensors each. Every source tensor is therefore accounted for, and
`convert_state_dict` raises on any key it has no destination for rather than dropping it. Those 109 buffers were
read back after the load and are all finite and non-zero, which is the check that catches a constructor-computed
buffer coming back as uninitialised memory under meta-device initialisation.

Copy synthesis through that direct load, on a LibriSpeech validation clip resampled to 24 kHz and normalised the
way the extractor normalises it, gives a log mel L1 of 0.0913 against the 0.0887 calibration point, at a
reconstruction to reference RMS ratio of 0.98. Re-randomising `conv_post` at the model's own initializer range
takes it to 1.59 and re-randomising `resblocks[0]` to 0.83.

The migrated model was checked against NVIDIA's own `bigvgan.BigVGAN`, run from the same weights on the same mel
spectrogram, in float32 on the CPU. On a 5.3 second clip the two waveforms agree to a maximum absolute difference
of 1.1e-05 and a mean absolute difference of 1.8e-07, on a waveform whose magnitude is 0.95. That residual is
float32 accumulation order and not a structural difference, and it was pinned down rather than assumed:

- Folded against folded, every convolution weight agrees with the one upstream's own `remove_weight_norm`
  produces to at most 6.0e-08, and every `alpha` and `beta` is bit identical.
- Stage by stage, `conv_pre` and the first transposed convolution agree bit for bit, at exactly 0.0. The
  disagreement first appears after the first residual block, at 1.7e-06 on activations whose magnitude is 6.7,
  and grows smoothly through the six stages to 2.6e-05 on activations whose magnitude is 3.0.
- With upstream's own folded weights copied in, so that the two models are bit identical, the waveforms still
  differ by 4.6e-06, which is the floor set by the operation ordering alone.
- The rebuilt resampling filters differ from the ones the checkpoint stores as buffers by 2.98e-08, the float32
  rounding of the same Kaiser windowed sinc formula, and feeding the checkpoint's own filters in instead leaves
  the difference at the same order.

`BigVGANFeatureExtractor` agrees with upstream's `meldataset.get_mel_spectrogram`, which is `librosa`'s mel
filterbank, to 2.3e-05 on log mel values whose magnitude is 10.4.

The training objective was checked the same way. `forward(labels=...)` returns 9.619857788085938 where upstream's
own `MultiScaleMelSpectrogramLoss` times `lambda_melloss` returns 9.619857966899872, a relative difference of
1.9e-08; the residual is that upstream's filterbank arrives from `librosa` as float64 and promotes the whole
computation, while this one stays in float32. Against the same clip the loss is 9.62 on that clip's own waveform,
107.09 on the same clip shifted half a second and 152.02 on noise of the same standard deviation, so the
objective is reading the waveform it is handed.

Copy synthesis of the F5-TTS demo reference clip through the real weights transcribes back under wav2vec2 as SOME
CALL ME NATURE OTHERS CALL ME MOTHER NATURE, word for word with the transcription of the original recording.

The two configuration settings no published checkpoint exercises, `resblock_type="2"` and `activation="snake"`,
were only checked for construction and shape. `activation="snake"` is what PromptTTS++'s vocoder uses, and that
path is verified against real weights in that folder.

Checkpoint search, per CLAUDE.md section 2.3. The Hugging Face hub holds nine NVIDIA repositories:
`bigvgan_v2_24khz_100band_256x`, `bigvgan_v2_22khz_80band_256x`, `bigvgan_v2_22khz_80band_fmax8k_256x`,
`bigvgan_v2_44khz_128band_256x`, `bigvgan_v2_44khz_128band_512x`, `bigvgan_base_22khz_80band`,
`bigvgan_base_24khz_100band`, `bigvgan_22khz_80band` and `bigvgan_24khz_100band`, which are the nine the upstream
README's table links. All nine set `resblock` to `"1"` and `activation` to `"snakebeta"` with `snake_logscale`
true; the five v2 ones set `use_tanh_at_final` and `use_bias_at_final` false and the four older ones predate both
keys, which default to true. The v2 repositories also publish `bigvgan_generator_3msteps.pt` beside
`bigvgan_generator.pt`, and every repository publishes its discriminator optimizer state as
`bigvgan_discriminator_optimizer.pt`. The hub search also returns community forks and finetunes of the same
architectures, of which `PolyAI/BigVGAN-L`, `amphion/BigVGAN_singing_bigdata` and `SPRINGLab/bigvgan_16khz` are
examples. Hugging Face Spaces holds `nvidia/BigVGAN`, which bundles the source but downloads the weights from
those model repositories, and `Arrcttacsrks/BigVGAN-main`. No Zenodo record or paper appendix was needed.

**Independently re-verified**, on a remote GPU session per CLAUDE.md section 2.5 rather than the local machine.
`convert` then `from_pretrained` on the real `nvidia/bigvgan_v2_24khz_100band_256x` weights again reports zero
missing, unexpected or mismatched keys across all 449 tensors and 112414512 parameters. Enumerating the 783
source tensors directly, rather than trusting that a clean load implies full coverage, shows 565 consumed (449
destination tensors plus the 116 `weight_v` halves each paired `weight_g` reads) and 218 discarded by design,
the resampling filters the model rebuilds from its configuration, with zero left over.

Copy synthesis of the F5-TTS demo reference clip (`basic_ref_en.wav`, 5.33 seconds) reproduces its own log mel
spectrogram to an L1 of 0.0887 and an SNR of 5.70 dB against the loudness normalized reference waveform, against
the 0.0886 the f5_tts path measured in `d35f867f`. As a negative control, re-randomizing `conv_post` alone
collapses that to an L1 of 3.13 and an SNR of -15.5 dB, and re-randomizing every convolution of `resblocks[0]`
collapses it to 4.35 and -16.9 dB, confirming the metric actually responds to a broken conversion rather than
passing regardless of what the weights are. wav2vec2 transcribes the copy synthesized clip as SOME CALL ME
NATURE OTHERS CALL ME MOTHER NATURE, word for word against the source recording.

Both consumers were re-checked against the current state of this folder rather than assumed from the commits
that wired them up. `f5_tts`'s `F5TTS_Base_bigvgan` converts and loads clean at 812 tensors and 449511316
parameters with the composed `BigVGANModel` frozen, and generating the upstream demo text in the upstream demo
reference voice transcribes verbatim under wav2vec2. `prompt_tts_pp` converts and loads clean at 669 tensors
(181696350 parameters) for the acoustic model and 239 tensors (13269867 parameters) for its `PromptTTSPPBigVGan`
vocoder, and both CMUdict test prompts transcribe verbatim.


## Not carried over from upstream

Recorded per CLAUDE.md section 2.6. None of these is resolved here.

- **The adversarial half of the training objective.** `forward(labels=...)` returns the mel reconstruction term
  only. `MultiPeriodDiscriminator`, `MultiResolutionDiscriminator`, `MultiBandDiscriminator`,
  `MultiScaleSubbandCQTDiscriminator`, `CombinedDiscriminator`, `feature_loss`, `discriminator_loss` and
  `generator_loss` have no counterpart, so four of the five terms of the generator loss and the whole
  discriminator loss are missing, and with `--freeze_step` defaulting to 0 there is no phase of upstream training
  that optimizes the reconstruction term alone. This is the same gap `voicestudio/models/vocos` has and it is
  still open there, but one of the reasons recorded for it does not hold here: BigVGAN's discriminators are not
  absent from the published checkpoints. Every `nvidia/bigvgan*` repository ships a
  `bigvgan_discriminator_optimizer.pt`, so weights to verify them against do exist. A BigVGAN trained through
  `forward` alone would not reproduce a released checkpoint. This is a scope decision and it needs a human.
- **The fused CUDA kernel of the anti aliased activation.** `alias_free_activation/cuda/` is a hand written
  `anti_alias_activation_cuda.cu` that fuses the upsample, the snake nonlinearity and the downsample into one
  kernel, loaded through `torch.utils.cpp_extension.load` when the configuration sets `use_cuda_kernel`. It is
  inference only upstream, it needs `nvcc` and `ninja` at run time, and the repository's rule is that kernels go
  through the `kernels` package. Only the PyTorch path is carried over, which is the path upstream's own
  `use_cuda_kernel=False` default takes.
- **`h.get("use_cuda_kernel")` and the `AttrDict` hyperparameter object.** `BigVGANConfig` is a
  `PreTrainedConfig`, so the training-only keys of the published `config.json` (`num_gpus`, `batch_size`,
  `learning_rate`, `adam_b1`, `adam_b2`, `lr_decay`, `seed`, `segment_size`, `num_workers`, `dist_config`,
  `clip_grad_norm`, `num_freq`, `normalize_volume`, and every `cqtd_*`, `mpd_*` and discriminator key) are read
  by `build_config` only where they describe the generator or its reconstruction loss, and are otherwise dropped
  along with the training loop they belong to.
- **The optimizers, the schedule and the two-optimizer loop.** `train.py` alternates a discriminator step and a
  generator step per batch under two AdamW optimizers with their own exponential schedules and their own gradient
  clipping. Neither the alternation nor the split parameter groups is what `transformers.Trainer` sets up, and
  reproducing them means a custom training loop.
- **`meldataset.MelDataset` and the training entry point.** The dataset read a filelist, cropped a random
  `segment_size` window, resampled it, peak normalized it and returned the `(mel, audio, filename, mel_loss)`
  tuple the trainer wants, retrying a random other sample on a read failure. `BigVGANFeatureExtractor` carries
  the mel spectrogram and the peak normalization; the cropping, the resampling and the collation are `Trainer`
  and data collator territory.
- **The validation loop and its metrics.** `train.py:validate` computes a mel L1 error, PESQ through the `pesq`
  package on a 16 kHz resample, and a multi resolution STFT loss through `auraloss`, and logs spectrograms to
  tensorboard. They are dropped along with those two packages.
- **`inference.py` and `inference_e2e.py`.** Two command line entry points that vocode a directory of wav files
  or of precomputed mel `.npy` files. The Usage section above is their replacement.
- **The exact from scratch initialization.** `utils.py:init_weights` draws `normal_(0.0, 0.01)` for the weight of
  anything whose class name contains "Conv", and upstream applies it to the upsampling stack, the residual block
  convolutions and `conv_post` only, leaving `conv_pre` and every bias at PyTorch's own defaults.
  `BigVGANPreTrainedModel._init_weights` applies the same distribution to `conv_pre` as well and zeroes every
  bias, which is the `transformers` convention and what `voicestudio/models/vocos` does. It affects a from
  scratch training run and nothing about loading a published checkpoint.


## File map

One counterpart per upstream file, per CLAUDE.md section 2.4. No BigVGAN tree was ever vendored under
`voicestudio/models/`, so there was nothing to `git mv`; the files below were traced from NVIDIA's own source and
written into the layout this repository uses.

| Upstream file | Where it went |
|---|---|
| `bigvgan.py` | `modeling_bigvgan.py`: `BigVGAN` became `BigVGANModel`, `AMPBlock1` and `AMPBlock2` became `BigVGANAmpBlock` and `BigVGANAmpLayer`, whose second convolution and second activation exist only under `resblock_type` `"1"`. `_save_pretrained` and `_from_pretrained`, its `PyTorchModelHubMixin` hooks, became `weight_conversion.py` |
| `activations.py` | `modeling_bigvgan.py`: `Snake` and `SnakeBeta` are the two branches of `BigVGANSnakeActivation`, selected by `config.activation` |
| `alias_free_activation/torch/act.py`, `resample.py`, `filter.py` | `modeling_bigvgan.py`: `Activation1d`, `UpSample1d`, `DownSample1d` and `LowPassFilter1d` are inlined into `BigVGANSnakeActivation.forward`, which shares one filter buffer because both directions build it from the same cutoff, half width and kernel size. `kaiser_sinc_filter1d` became `build_anti_alias_filter` |
| `alias_free_activation/cuda/*` | Not carried over, see above |
| `meldataset.py` | `feature_extraction_bigvgan.py` for `mel_spectrogram` and the peak normalization, `modeling_bigvgan.py`'s `mel_spectrogram` for the same computation inside the loss, and `dynamic_range_compression` for `spectral_normalize_torch`. `MelDataset` and `get_dataset_filelist` are not carried over, see above |
| `loss.py` | `modeling_bigvgan.py`: `MultiScaleMelSpectrogramLoss` is `BigVGANModel.mel_loss` plus `mel_loss_resolutions`. `feature_loss`, `discriminator_loss` and `generator_loss` are not carried over, see above |
| `discriminators.py` | Not carried over, see above |
| `train.py` | The objective is `forward(labels=...)`. The two-optimizer loop, the schedules, the checkpointing and the validation have no counterpart |
| `utils.py` | `modeling_bigvgan.py`: `init_weights` is the `nn.Conv1d` and `nn.ConvTranspose1d` branch of `BigVGANPreTrainedModel._init_weights` and `get_padding` is inlined into `BigVGANAmpLayer`. `apply_weight_norm` is `BigVGANModel.apply_weight_norm`. The plotting, checkpoint scanning and audio saving helpers are dropped |
| `env.py` | `configuration_bigvgan.py`. `AttrDict` and `build_env` are what `PreTrainedConfig` replaces |
| `inference.py`, `inference_e2e.py` | Not carried over, see above |
| `config.json` | `BigVGANConfig`, and `build_config` in `weight_conversion.py`, which reads the same keys out of a published repository's `config.json` |
| `configs/*.json` | The nine published configurations, which `PUBLISHED_CHECKPOINTS` in `weight_conversion.py` names |
| `tests/test_cuda_vs_torch_model.py` | Dropped, no counterpart. It checks the CUDA kernel against the PyTorch path, and the kernel is not carried over |
| `demo/*`, `scripts/*`, `nv-modelcard++/*` | Dropped, no counterpart. The Gradio demo, the inference helper scripts and NVIDIA's model card supplements |
| `LICENSE`, `incl_licenses/*` | The header of `modeling_bigvgan.py`, per CLAUDE.md section 6. The included licenses are HiFi-GAN's, `alias-free-torch`'s, `julius`', `snake`'s and `descript-audio-codec`'s, which are the projects the files above are adapted from |
| `README.md` | This file's Usage, Training and Lineage sections |


## Repository integration

Three things outside this folder are still needed and were deliberately not touched:

- `voicestudio/models/__init__.py` needs a `from .bigvgan import *` line, before the `f5_tts` and
  `prompt_tts_pp` lines are reached, though `from ..bigvgan import ...` inside those two packages already imports
  it either way.
- `PROJECT.md` needs a BigVGAN status entry carrying the gaps listed above, in particular the adversarial half of
  the training objective, and its "Sibling inheritance map" entry for `PromptTTSPPBigVGan` can move from
  "actionable once `voicestudio/models/bigvgan/` exists" to done.
- `PROJECT.md`'s Vocos open item says the discriminators "are training-only modules absent from every published
  checkpoint". That reason does not hold for BigVGAN, whose repositories all publish
  `bigvgan_discriminator_optimizer.pt`.

Nothing in `pyproject.toml` or `uv.lock` changes.
