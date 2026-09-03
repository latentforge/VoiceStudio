# VoiceStudio

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

A unified toolkit for voice cloning, designing and editing.

---

## Overview

VoiceStudio collects nineteen speech synthesis models, vocoders and codecs into one repository behind
one API. Each is implemented directly against the `transformers` model API as a `PreTrainedConfig`, a
`PreTrainedModel` and a `Processor`: loaded with `from_pretrained`, driven by `generate`, trained
through `forward(labels=...)`. There is no wrapper layer, no per-model runtime, and no vendored copy
of an upstream inference stack sitting beside the library.

**One API, and models that compose.** Every model takes its inputs from its own `Processor` and
returns audio or audio codes the same processor decodes, so switching models is switching a class
name. Because the pieces are ordinary `transformers` submodels, they also compose across folders:
Parler-TTS owns a `DacModel`, Chroma owns a `MimiModel` and a Qwen2.5-Omni thinker, VoxInstruct owns
a `VocosModel`, and F5-TTS owns whichever of `VocosModel` or `BigVGANModel` its checkpoint was
trained against.

**Inheritance instead of reimplementation.** A model is rebased onto the closest existing lineage
rather than carrying a parallel copy of it. What the folders here actually inherit from spans
`llama`, `qwen2`, `qwen3`, `csm`, `mimi`, `encodec`, `dac`, `xcodec2`, `whisper`, `hubert`,
`wav2vec2`, `mt5`, `t5gemma2`, `speecht5`, `musicgen`, `bert`, `fastspeech2_conformer` and
`qwen2_5_omni`. They also inherit from each other: CosyVoice v2 subclasses v1 and v3 subclasses v2,
CosyVoice v3's diffusion transformer layers come from F5-TTS, and PromptTTS++'s vocoder is BigVGAN's
with a source filter path added.

**Trainable, not inference only.** Every top-level model accepts `labels` and returns a loss through
the standard `ModelOutput` pattern. The objective is read out of the upstream project's own trainer,
loss module, collator or evaluation script rather than guessed from the model's shape, down to the
term weights, the masking, and which modules upstream freezes. Where a term could not be carried
across, the gap is written down instead of being papered over; the Status column below says which
models that applies to.

**Verified against published weights.** A dummy tensor passing through a `forward` proves nothing, so
it is not what "verified" means here. Each model is loaded from its real published checkpoint, made
to speak, and the audio is transcribed back and compared against the text it was given. Where an
upstream class can be run side by side, the migration is also checked numerically against it: F5-TTS's
backbone diverges by 1.2e-05 to 3.0e-04 on activations of magnitude 13, CosyVoice v2's flow matching
stack is bit exact, Vocos agrees to 1.5e-08 on mel features, and OmniVoice's forward agrees to
4.8e-07 against an independent reimplementation of it.

**Fewer dependencies, not more.** A migration ends an upstream import rather than adding one. The
runtime requirement is `torch`, `torchaudio`, `torchcodec`, `numpy`, `librosa`, `transformers`,
`hf-xet` and `tqdm`. Along the way `descript-audio-codec`, `vocos`, `encodec`, `fairseq`,
`torchdiffeq`, `whisper-timestamped`, `diffusers`, `matcha-tts`, `einops`, `omegaconf`,
`hyperpyyaml`, `openai-whisper`, `peft`, `pydub`, `gradio`, `speechbrain` and `audiotools`, among
others, were replaced by a native `transformers` class or by inlining the few dozen lines actually
used. The handful that survived are optional rather than required: `onnxruntime`, because CosyVoice
v1's speech tokenizer and speaker encoder are published as ONNX graphs and nothing else, plus
`pillow` and `torchvision`, which reach Chroma through `Qwen2_5OmniProcessor`. Each lives in an
extra, or is imported lazily at the point of use and named in the error if it is absent.

---

## Installation

Python 3.11 or newer, and PyTorch 2.8 or newer.

```bash
git clone https://github.com/LatentForge/VoiceStudio.git
cd VoiceStudio
uv sync
```

`uv sync` is the supported path. The `transformers` requirement resolves through `[tool.uv.sources]`
to [latentforge/transformers-tts](https://github.com/latentforge/transformers-tts), the branch that
carries the speech models this repository relays and inherits from. Installation routes that ignore
`[tool.uv.sources]`, including `pip install` and `uv pip install`, take `transformers` from PyPI
instead, and the relayed models are not in it. The `voicestudio` distribution on PyPI is likewise
older than this work and pins `transformers==4.57.6`, so it does not carry the models below.

Optional extras, selected with `uv sync --extra <name>` or all at once with `uv sync --all-extras`:

| Extra | Pulls in | Needed for |
|---|---|---|
| `train` | `accelerate`, `wandb`, `matplotlib`, `notebook`, `ipywidgets` | training runs and notebooks |
| `eval` | `pyworld`, `jiwer` | f0 extraction for CosyVoice's vocoder objective, and the word error rate check used to verify a model |
| `kernels` | `transformers[kernels]` | flash attention and other fused kernels |
| `omni` | `pillow`, `torchvision` | Chroma, whose processor subclasses `Qwen2_5OmniProcessor` |
| `onnx` | `onnxruntime`, `onnx` | CosyVoice v1's speech tokenizer and speaker encoder, and Qwen3-TTS |
| `native` | `torchnative` | on-device inference |
| `web` | `fastapi` | the web front end |
| `all` | `train`, `eval`, `kernels`, `omni`, `onnx`, `web` | everything except `native` |

---

## Usage

Models that ship in `transformers-tts` load straight from their published repository:

```python
import soundfile as sf
from transformers import AutoModelForTextToWaveform, AutoProcessor

model_id = "bosonai/higgs-tts-2-3b-base"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTextToWaveform.from_pretrained(model_id).to("cuda")
processor.audio_tokenizer.to(model.device)

conversation = [
    {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
    {"role": "user", "content": [{"type": "text", "text": "The sun rises in the east."}]},
]
inputs = processor.apply_chat_template(
    conversation,
    return_dict=True,
    tokenize=True,
    add_generation_prompt=True,
    sampling_rate=24000,
    return_tensors="pt",
).to(model.device)

audio_codes = model.generate(**inputs, max_new_tokens=1024)
waveform = processor.decode(audio_codes)
sf.write("output.wav", waveform.numpy(), processor.audio_tokenizer.config.sample_rate)
```

Models whose upstream release ships a bespoke weight layout are converted once through their folder's
`weight_conversion.convert`, after which they load the same way. Each model's own README carries its
conversion call, the generation arguments that are load bearing for it, its training objective, and
what was not carried over from upstream.

---

## Models

Every model below loads real published weights and has been run against them. Follow the model name
for its folder README, which documents its usage, its objective and its open items.

### Voice Cloning

Reproduce the voice of a reference recording.

| Model | Year | Paper | Hugging Face | Status |
|---|---|---|---|---|
| [Breeze TTS 2](voicestudio/models/breeze_tts) | 2026 | | [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) | Verified |
| [Chroma](voicestudio/models/chroma) | 2026 | [arXiv:2601.11141](https://arxiv.org/abs/2601.11141) | [FlashLabs/Chroma-4B](https://huggingface.co/FlashLabs/Chroma-4B) | Verified |
| [CosyVoice v1](voicestudio/models/cosyvoice_v1) | 2024 | [arXiv:2407.05407](https://arxiv.org/abs/2407.05407) | [FunAudioLLM/CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M) | Verified, objective gap |
| [CosyVoice v2](voicestudio/models/cosyvoice_v2) | 2024 | [arXiv:2412.10117](https://arxiv.org/abs/2412.10117) | [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | Verified, objective gap |
| [CosyVoice v3](voicestudio/models/cosyvoice_v3) | 2025 | [arXiv:2505.17589](https://arxiv.org/abs/2505.17589) | [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | Verified, objective gap |
| [Dia](voicestudio/models/dia) | 2025 | | [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626) | Verified, relay |
| [Dia2](voicestudio/models/dia2) | 2025 | | [nari-labs/Dia2-2B](https://huggingface.co/nari-labs/Dia2-2B) | Verified, objective gap |
| [F5-TTS](voicestudio/models/f5_tts) | 2024 | [arXiv:2410.06885](https://arxiv.org/abs/2410.06885) | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) | Verified |
| [Higgs TTS 2](voicestudio/models/higgs_tts2) | 2025 | | [bosonai/higgs-tts-2-3b-base](https://huggingface.co/bosonai/higgs-tts-2-3b-base) | Verified, relay |
| [Higgs TTS 3](voicestudio/models/higgs_tts3) | 2026 | | [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b) | Verified |
| [OmniVoice](voicestudio/models/ommivoice) | 2026 | [arXiv:2604.00688](https://arxiv.org/abs/2604.00688) | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | Verified |
| [Qwen3-TTS](voicestudio/models/qwen3_tts) | 2026 | [arXiv:2601.15621](https://arxiv.org/abs/2601.15621) | [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Verified, relay |
| [Spark-TTS](voicestudio/models/spark_tts) | 2025 | [arXiv:2503.01710](https://arxiv.org/abs/2503.01710) | [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | Verified |
| [VoxInstruct](voicestudio/models/vox_instruct) | 2024 | [arXiv:2408.15676](https://arxiv.org/abs/2408.15676) | [niobures/VoxInstruct](https://huggingface.co/niobures/VoxInstruct) | Verified |

### Voice Design

Build a voice from a natural language description, with no reference recording.

| Model | Year | Paper | Hugging Face | Status |
|---|---|---|---|---|
| [Breeze TTS 2](voicestudio/models/breeze_tts) | 2026 | | [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) | Verified |
| [CosyVoice v1](voicestudio/models/cosyvoice_v1) | 2024 | [arXiv:2407.05407](https://arxiv.org/abs/2407.05407) | [FunAudioLLM/CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M) | Verified, objective gap |
| [CosyVoice v2](voicestudio/models/cosyvoice_v2) | 2024 | [arXiv:2412.10117](https://arxiv.org/abs/2412.10117) | [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | Verified, objective gap |
| [CosyVoice v3](voicestudio/models/cosyvoice_v3) | 2025 | [arXiv:2505.17589](https://arxiv.org/abs/2505.17589) | [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | Verified, objective gap |
| [OmniVoice](voicestudio/models/ommivoice) | 2026 | [arXiv:2604.00688](https://arxiv.org/abs/2604.00688) | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | Verified |
| [Parler-TTS](voicestudio/models/parler_tts) | 2024 | [arXiv:2402.01912](https://arxiv.org/abs/2402.01912) | [parler-tts/parler-tts-mini-v1](https://huggingface.co/parler-tts/parler-tts-mini-v1) | Verified |
| [PromptTTS++](voicestudio/models/prompt_tts_pp) | 2023 | [arXiv:2309.08140](https://arxiv.org/abs/2309.08140) | [line-corporation/promptttspp](https://huggingface.co/spaces/line-corporation/promptttspp) | Verified, objective gap |
| [Qwen3-TTS](voicestudio/models/qwen3_tts) | 2026 | [arXiv:2601.15621](https://arxiv.org/abs/2601.15621) | [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Verified, relay |
| [Spark-TTS](voicestudio/models/spark_tts) | 2025 | [arXiv:2503.01710](https://arxiv.org/abs/2503.01710) | [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | Verified |
| [VoxInstruct](voicestudio/models/vox_instruct) | 2024 | [arXiv:2408.15676](https://arxiv.org/abs/2408.15676) | [niobures/VoxInstruct](https://huggingface.co/niobures/VoxInstruct) | Verified |

PromptTTS++ publishes no model repository. Its only public weights are bundled inside the Space linked
above, which is what its `weight_conversion.convert` downloads.

### Voice Editing

Change the voice of a recording, or rewrite part of it, while keeping the rest.

| Model | Year | Paper | Hugging Face | Status |
|---|---|---|---|---|
| [CosyVoice v1](voicestudio/models/cosyvoice_v1) | 2024 | [arXiv:2407.05407](https://arxiv.org/abs/2407.05407) | [FunAudioLLM/CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M) | Verified, objective gap |
| [CosyVoice v2](voicestudio/models/cosyvoice_v2) | 2024 | [arXiv:2412.10117](https://arxiv.org/abs/2412.10117) | [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | Verified, objective gap |
| [CosyVoice v3](voicestudio/models/cosyvoice_v3) | 2025 | [arXiv:2505.17589](https://arxiv.org/abs/2505.17589) | [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | Verified, objective gap |
| [F5-TTS](voicestudio/models/f5_tts) | 2024 | [arXiv:2410.06885](https://arxiv.org/abs/2410.06885) | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) | Verified |

F5-TTS infills a masked span of an existing recording through its `edit_mask` argument. All three
CosyVoice versions convert the voice of a recording while keeping its content, through
`source_speech_token_ids`.

### Vocoders and Codecs

Not text-to-speech models. These turn features or codes into a waveform, or a waveform into tokens,
and the models above hold them as submodels.

| Model | Year | Paper | Hugging Face | Status |
|---|---|---|---|---|
| [BigVGAN](voicestudio/models/bigvgan) | 2022 | [arXiv:2206.04658](https://arxiv.org/abs/2206.04658) | [nvidia/bigvgan_v2_24khz_100band_256x](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x) | Verified, objective gap |
| [Spark-TTS BiCodec](voicestudio/models/spark_tts_bicodec) | 2025 | [arXiv:2503.01710](https://arxiv.org/abs/2503.01710) | [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | Verified, objective gap |
| [Vocos](voicestudio/models/vocos) | 2023 | [arXiv:2306.00814](https://arxiv.org/abs/2306.00814) | [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) | Verified, objective gap |

### Status legend

| Value | Meaning |
|---|---|
| Verified | Loads its real published checkpoint, generates audio that transcribes back to the text it was given, and its `forward(labels=...)` implements upstream's own training objective term for term. |
| Verified, objective gap | Verified in the same way, but part of upstream's training objective is not implemented and the omission is still open. In every case but one the missing part is a GAN discriminator, which no `transformers` model class carries. Dia2 is the exception: upstream publishes no training code at all, so its three loss terms are summed unweighted. |
| Verified, relay | The model itself ships in `transformers-tts`; the folder re-exports it, adding only a processor where one was missing. Verified against real weights in the same way. |

Year is the year the model was first published. An empty Paper cell means the release has no arXiv
paper, only code and a model card. `PROJECT.md` carries the per-model verification evidence and the
full list of open items, including two the Status column does not cover: Higgs TTS 3 reports 528
unexpected keys on load, all of them the codec copy bundled in its checkpoint, and Qwen3-TTS's audio
tokenizer reports one missing key for a module that is never called.

---

## Contributing

Issues and pull requests are welcome at
[github.com/LatentForge/VoiceStudio](https://github.com/LatentForge/VoiceStudio/issues).

Two files in the repository root are the working documentation, and both are worth reading before
opening a pull request. [CLAUDE.md](CLAUDE.md) is the conventions document: how a model is migrated,
what counts as verification, how files and comments are named, and the rules on dependencies and
licence headers. [PROJECT.md](PROJECT.md) is the running status of the work, including every open
item recorded against a model.

Areas where help is most useful:

- The open objective gaps in the table above, all of which need a decision before they need code.
- Inference performance. Nothing here has been tuned for it, and the route is the `transformers` one,
  a static cache and a compiled graph selected through `GenerationConfig`, rather than a per-model
  capture. `PROJECT.md` has the detail.
- More models, migrated the way the existing nineteen were.

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

Each `modeling_<model>.py` also carries the licence header of the project its code came from, which is
not always Apache 2.0.

The checkpoints are under their own licences, which are not this repository's, and several are more
restrictive than the code that loads them. `BreezeBlue/Breeze-TTS-2` ships a research and
non-commercial licence, `bosonai/higgs-tts-3-4b` likewise, and `FlashLabs/Chroma-4B` is gated behind
an access request. Review a checkpoint's licence before using it.

---

## Acknowledgments

This repository is other people's research, brought under one API. The models come from:

- [NVIDIA](https://github.com/NVIDIA/BigVGAN) for BigVGAN
- [BreezeBlue](https://github.com/breezeblue-ai/breeze-tts) for Breeze TTS 2
- [FlashLabs](https://github.com/FlashLabs-AI-Corp/FlashLabs-Chroma) for Chroma
- [FunAudioLLM](https://github.com/FunAudioLLM/CosyVoice) for CosyVoice v1, v2 and v3
- [Nari Labs](https://github.com/nari-labs) for [Dia](https://github.com/nari-labs/dia) and [Dia2](https://github.com/nari-labs/dia2)
- [SWivid](https://github.com/SWivid/F5-TTS) for F5-TTS and E2-TTS
- [Boson AI](https://github.com/boson-ai/higgs-audio) for Higgs TTS 2 and Higgs TTS 3
- [k2-fsa](https://github.com/k2-fsa/OmniVoice) for OmniVoice
- [Hugging Face](https://github.com/huggingface/parler-tts) for Parler-TTS
- [LINE](https://github.com/line/promptttspp) for PromptTTS++
- [Qwen](https://github.com/QwenLM/Qwen3-TTS) for Qwen3-TTS
- [SparkAudio](https://github.com/SparkAudio/Spark-TTS) for Spark-TTS and BiCodec
- [gemelo.ai](https://github.com/gemelo-ai/vocos) for Vocos
- [THU-HCSI](https://github.com/thuhcsi/VoxInstruct) for VoxInstruct

And the libraries the code is built out of:

- [Hugging Face `transformers`](https://github.com/huggingface/transformers), whose model classes
  almost every file here inherits from, through
  [latentforge/transformers-tts](https://github.com/latentforge/transformers-tts).
- [PyTorch](https://pytorch.org/), with `torchaudio` and `torchcodec`.
- [librosa](https://github.com/librosa/librosa) and [NumPy](https://numpy.org/).

---

## Links

- Repository: [github.com/LatentForge/VoiceStudio](https://github.com/LatentForge/VoiceStudio)
- Group homepage: [latentforge.github.io](https://latentforge.github.io/)
