# PromptTTS++

PromptTTS++ speaks a phoneme sequence in a voice described in natural language. A four block conformer encodes the
phonemes; a BERT encoder turns the style prompt into a 256 dimensional embedding, which a mixture density network
maps to the style embedding a global style token encoder would have read off a real recording of that speaker;
that style embedding is added onto every encoded phoneme. A variance adaptor then predicts phoneme durations with
a second mixture density network, regulates the encoder output onto the frame grid, runs it through a frame prior
network, and predicts log continuous f0 and voicing, embedding the f0 back into the frame features. A denoising
diffusion decoder generates the mel spectrogram from those features over 100 steps, and an f0 aware BigVGAN
vocoder turns the spectrogram and the predicted f0 into a waveform.

Original model and code: [line/promptttspp](https://github.com/line/promptttspp)


## Usage

The only public weights are the ones bundled inside the model's Hugging Face Space, in the upstream trainer's
`.ckpt` format, so they need a one time conversion. Called without paths, `convert` downloads them from the Space:

```python
from voicestudio.models.prompt_tts_pp.weight_conversion import convert

convert(output_dir="prompt-tts-pp-converted")
```

The acoustic model, the tokenizers and the feature extractor land in the output directory, the vocoder in its
`vocoder` subdirectory:

```python
import torch
from voicestudio.models.prompt_tts_pp import (
    PromptTTSPPBigVGan,
    PromptTTSPPForConditionalGeneration,
    PromptTTSPPProcessor,
)

model_id = "prompt-tts-pp-converted"

processor = PromptTTSPPProcessor.from_pretrained(model_id)
model = PromptTTSPPForConditionalGeneration.from_pretrained(model_id).eval()
vocoder = PromptTTSPPBigVGan.from_pretrained(f"{model_id}/vocoder").eval()
```

```python
import soundfile as sf

inputs = processor(
    text="The quick brown fox jumps over the lazy dog.",
    style_prompt="A man speaks slowly in a low tone.",
)

with torch.no_grad():
    outputs = model(**inputs, style_noise_scale=0.5)
    spectrogram, f0 = processor.postprocess(outputs)
    waveform = vocoder(spectrogram, f0)

sf.write("output.wav", waveform.squeeze(0).numpy(), processor.feature_extractor.sampling_rate)
```

`processor.postprocess` is what turns a model output into vocoder inputs: it lowpass filters the predicted log f0
contour at 20 Hz, exponentiates it, zeroes the frames the model called unvoiced, and undoes the mel
standardization. The vocoder needs both the spectrogram and that f0, not the spectrogram alone.

Text goes through `g2p_en`, which is an optional backend: install it, or build the tokenizer with
`phonemize=False` and pass a whitespace separated sequence of the Montreal Forced Aligner phoneme symbols
directly.

The speaker can also come from a recording instead of a description, in which case the global style token encoder
reads the style embedding off the reference mel spectrogram:

```python
inputs = processor(text="The quick brown fox jumps over the lazy dog.", audio=reference_waveform, sampling_rate=24000)
```

`style_noise_scale` scales the noise added around the style mixture density network's mean, and
`use_max_style=False` samples a mixture component instead of taking the most probable one. Both only apply to the
style prompt path.


## Training

Training uses the standard `forward`: pass the normalized target mel spectrogram as `labels`, the frame mask as
`spectrogram_attention_mask`, and the aligned `duration_labels`, `pitch_labels` and `vuv_labels`. The returned
`loss` is the sum of the five terms below, each also reported on its own in `PromptTTSPPOutput`.

Term by term, as upstream `PromptTTSMDNDurCFG.forward` computes it:

- `spectrogram_loss`: the diffusion decoder draws one timestep per batch item uniformly from `[0, 100)`,
  normalizes the target spectrogram by `diffusion_norm_scale` (6.0), noises it, and asks the denoiser for the
  noise. The term is the L1 distance between the drawn and the predicted noise, both zeroed on padded frames,
  summed over the 80 mel channels and all frames, divided by the number of valid frames and then by
  `spectrogram_loss_scale` (8.0).
- `duration_loss`: the negative log likelihood of the log durations under the duration predictor's four component
  dimension-wise mixture, computed per phoneme, masked to the valid phonemes and averaged over them.
- `pitch_loss` and `vuv_loss`: L1 between the pitch predictor's two channels and `pitch_labels` / `vuv_labels`,
  each summed over all frames and divided by the number of valid frames.
- `style_loss`: the negative log likelihood of the reference style embedding, detached, under the style mixture
  density network driven by the prompt embedding, averaged over the batch. Both embeddings are L2 normalized
  first, since `normalize_style_embedding` is set. With `use_style_mdn=False` this becomes the mean squared error
  between the detached reference embedding and the prompt embedding, which is what upstream falls back on.

Both mixture density networks and their losses run in full precision under autocast, which is what
`disable_mdn_autocast` (upstream `mdn_disable_amp`) means, because they destabilize otherwise.

Three teacher forcing details decide what the model actually optimizes. The alignment comes from
`duration_labels`, not from the duration predictor, so the duration term never feeds the decoder. The pitch
embedding added to the frame features is built from `pitch_labels`, not from the pitch prediction. And the style
embedding the encoder output is conditioned on comes from the global style token encoder reading the target
spectrogram, not from the prompt: the prompt only ever supervises the style mixture density network against that
embedding. `reference_spectrogram` therefore defaults to `labels` when training.

Upstream freezes every parameter of the BERT prompt encoder except the attention of its last layer
(`BertWrapper.__init__`), which `freeze_prompt_encoder` reproduces and `PromptTTSPPPromptEncoder.freeze_bert`
applies. Nothing else is frozen. The upstream trainer optimizes the rest with AdamW under a Noam schedule (4000
warmup steps) and clips gradients at a global norm of 1.0.


## Not carried over from upstream

Recorded per CLAUDE.md section 2.6. None of these is resolved here.

- **The energy branch.** `VarianceAdaptor` supports an energy predictor and an energy embedding, and the loss
  gains an `energy` term when they are present. The released configuration sets both to `null` (the config is
  named `wo_erg`), so `use_energy_predictor` defaults to `False`. The branch is implemented and reachable, but no
  checkpoint exercises it, so it has never been run against real weights.
- **Classifier-free guidance.** The upstream class is named `PromptTTSMDNDurCFG` and its docstring advertises
  "(Optional) classifier-free guidance (CFG) for diffusion-based decoder", but no code path in the released
  version implements it: neither `forward`, `infer` nor `GaussianDiffusion` ever drops the conditioning or mixes
  a conditional and an unconditional branch. Nothing was migrated for it.
- **The PLMS sampler.** `GaussianDiffusion.p_sample_plms` implements the pseudo linear multistep sampler that
  `pndm_speedup` would select, but `GaussianDiffusion.__init__` raises `NotImplementedError` whenever
  `pndm_speedup` is set, so the released model can only run the full 100 step ancestral sampler. Only the latter
  was migrated.
- **The conformer decoder variant.** Upstream's top level model accepts a `ConformerEncoder` in place of the
  diffusion decoder, in which case an `out_conv` projects to the mel channels and the decoder term becomes a
  plain L1 on the spectrogram. The released configuration uses the diffusion decoder, and only that path was
  migrated.
- **Feature extraction for training.** Durations come from Montreal Forced Aligner TextGrids
  (`promptttspp/preprocess/duration.py`, `promptttspp/utils/textgrid.py`) and the f0 contour from WORLD's DIO and
  StoneMask followed by interpolation over the unvoiced frames (`promptttspp/preprocess/pitch.py`). Both need
  `pyworld` and `nnmnkwii`, which this migration does not add, so `duration_labels`, `pitch_labels` and
  `vuv_labels` have to be produced by the caller.
- **The training data pipeline.** The dataset classes, the collator, the dynamic batch sampler, the trainer, the
  Noam scheduler, the loss tracker and the Hydra experiment configuration under `promptttspp/datasets/`,
  `promptttspp/trainers/`, `promptttspp/utils/` and `egs/` are dropped, as is `data_prep/`, which builds the
  LibriTTS-R derived corpus. The style prompt and speaker prompt candidate tables under `metadata/` go with them:
  they are training data, and the prompt is a free text argument at inference.
- **The Gradio demo and the batch synthesis script.** `app.py` and `egs/proposed/bin/synthesize.py` become
  `processing_prompt_tts_pp.py` and `weight_conversion.py` respectively, but their user interface, plotting and
  corpus iteration are gone along with `gradio` and `matplotlib`.
- **The unused module library.** `promptttspp/modules/` also ships a continuous normalizing flow, a ConvNeXt
  stack, a Glow, a multi receptive field net, a score based SDE, a plain transformer, a U-Net, a second conformer
  implementation and a copy of nnsvs' diffusion and denoiser. None of them is referenced by the released model
  configuration, and none was migrated. The same holds for the part of the vendored espnet subset under
  `promptttspp/modules/esp/transformer/` that the conformer encoder never reaches: the transformer encoder and
  decoder, the light and dynamic convolutions, the convolutional subsampling, the plain position wise feed
  forward and the weight initializers.
- **The f0 free BigVGAN.** `promptttspp/vocoders/bigvgan.py` also defines the vocoder variant that takes only a
  spectrogram, selected by `egs/proposed/bin/conf/vocoder/bigvgan.yaml`. Its residual blocks are migrated, since
  the f0 aware variant is built out of them, but the variant itself is not: the released vocoder checkpoint is
  the f0 aware one.


## Repository integration

Three things outside this folder are still needed and were deliberately not touched:

- `voicestudio/models/__init__.py` needs a `from .prompt_tts_pp import *` line.
- `PROJECT.md`'s PromptTTS++ row still describes the discarded `FastSpeech2Conformer` migration and records "no
  public checkpoint" as the reason it was never verified. Both statements are wrong: the architecture is the one
  described above, and the weights are bundled in the Space. The row needs replacing, along with the gaps listed
  above.
- `pyproject.toml`'s `eval` extra can drop `pysptk`, which nothing in the repository imports now that the
  vendored tree is gone. `pyworld` has to stay for `cosyvoice_v1`, `matplotlib` for
  `voicestudio/utils/audio_utils.py`, and `torchvision` for the `omni` extra, none of which this folder ever
  reached.

No new dependency is required. The migration drops `hydra-core`, `omegaconf`, `scipy`, `gradio`, `matplotlib`,
`pandas`, `pyworld`, `nnmnkwii`, `pysptk`, `tensorboard`, `joblib` and `faster_whisper` from what this model
needs, leaving `torch`, `torchaudio`, `numpy`, `transformers`, `safetensors`, `huggingface_hub` and `pyyaml`.
`g2p_en` stays an optional backend of the tokenizer, exactly as it is for `FastSpeech2ConformerTokenizer` in
`transformers` itself.
