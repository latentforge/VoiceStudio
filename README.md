# VoiceStudio

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

A unified toolkit for voice cloning, designing and editing.

---

## 🎯 Overview

Speech synthesis research is held back by its own tooling. Every model arrives as its own repository,
with its own runtime, its own checkpoint format, its own inference script and its own pinned
dependency set, and most of them ship weights you can run but no path to training them. Comparing two
models means learning two codebases; building on one means adopting it wholesale.

VoiceStudio removes that tax. Every model here is an ordinary `transformers` model: a
`PreTrainedConfig`, a `PreTrainedModel` and a `Processor`, loaded with `from_pretrained`, run with
`generate`, trained with `forward(labels=...)`. Swapping one for another is changing a class name.
Comparing them is a loop. Fine tuning one is the training code you already have.

**Key Features:**
- **One API**: every model takes its inputs from its own `Processor` and returns audio the same processor decodes, so switching models is switching a class name
- **Composable**: models hold each other as ordinary submodels, so Parler-TTS owns a `DacModel`, Chroma a `MimiModel`, and F5-TTS whichever of `VocosModel` or `BigVGANModel` its checkpoint was trained against
- **Inheritance over reimplementation**: rebased onto `llama`, `qwen3`, `csm`, `mimi`, `dac`, `speecht5` and a dozen more, and onto each other, rather than carrying parallel copies
- **Trainable, not inference only**: every model returns a loss, with the objective read out of the upstream project's own trainer rather than guessed from its shape
- **Verified against published weights**: loaded from the real checkpoint, made to speak, and the audio transcribed back and compared to the text it was given
- **Direct loading**: `from_pretrained` on the official repository id, with no conversion step for the caller to run
- **Fewer dependencies**: a migration ends an upstream import rather than adding one, leaving
  `transformers` as the only required dependency and everything else behind an extra

---

## 🛠️ Installation

Python 3.13 or newer, and PyTorch 2.8 or newer.

The base install carries only `transformers[kernels]`. The runtime and the audio stack are extras,
so pick the ones for the machine you are on.

### From source

```bash
git clone https://github.com/LatentForge/VoiceStudio.git
cd VoiceStudio
uv sync --extra cloud --extra audio
```

### For research

```bash
git clone https://github.com/LatentForge/VoiceStudio.git
cd VoiceStudio
uv sync --extra research
```

Use `uv sync` rather than `pip install`. `torch` is pinned to a specific index for Windows in
`[tool.uv.sources]`, and `pip` ignores that file. The `voicestudio` distribution on PyPI predates
this work and does not carry the models below.

Extras, selected with `uv sync --extra <name>` or all at once with `uv sync --all-extras`:

| Extra | Pulls in | Needed for |
|---|---|---|
| `research` | `cloud`, `audio`, `omni`, `train`, `eval` | the full research setup, everything but `native` and `web` |
| `cloud` | `torch`, `numpy`, `hf-xet` | running on NVIDIA hardware, the usual runtime |
| `native` | `torchnative` | on-device inference, in place of `cloud` |
| `audio` | `torchaudio`, `torchcodec`, `librosa`, `soundfile` | reading and writing waveforms, which every processor here does |
| `omni` | `pillow`, `torchvision` | Chroma, whose processor subclasses `Qwen2_5OmniProcessor` |
| `train` | `accelerate`, `wandb`, `matplotlib`, `notebook`, `ipywidgets`, `tqdm` | training runs and notebooks |
| `eval` | `jiwer` | The word error rate check used to verify a model |
| `web` | `fastapi` | the web front end |

Flash attention and the other fused kernels come through `transformers[kernels]`, which the base
install already carries, so there is no extra to select for them.

---

## 🚀 Usage

Models that `transformers` already ships load straight from their published repository:

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

## 📊 Models

Every model below loads real published weights and has been run against them. Follow the model name
for its folder README, which documents its usage, its objective and its open items.

### Voice Cloning

Reproduce the voice of a reference recording.

| Model | Year | Paper | Hugging Face | Status |
|---|---|---|---|---|
| [Breeze TTS 2](voicestudio/models/breeze_tts) | 2026 | | [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) | Verified |
| [Chroma](voicestudio/models/chroma) | 2026 | [arXiv:2601.11141](https://arxiv.org/abs/2601.11141) | [FlashLabs/Chroma-4B](https://huggingface.co/FlashLabs/Chroma-4B) | Verified |
| [Higgs TTS 3](voicestudio/models/higgs_tts3) | 2026 | | [bosonai/higgs-tts-3-4b](https://huggingface.co/bosonai/higgs-tts-3-4b) | Verified |
| [OmniVoice](voicestudio/models/ommivoice) | 2026 | [arXiv:2604.00688](https://arxiv.org/abs/2604.00688) | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | Verified |
| [Qwen3-TTS](voicestudio/models/qwen3_tts) | 2026 | [arXiv:2601.15621](https://arxiv.org/abs/2601.15621) | [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Verified, relay |
| [CosyVoice v3](voicestudio/models/cosyvoice_v3) | 2025 | [arXiv:2505.17589](https://arxiv.org/abs/2505.17589) | [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | Verified, no discriminator |
| [Dia](voicestudio/models/dia) | 2025 | | [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626) | Verified, relay |
| [Dia2](voicestudio/models/dia2) | 2025 | | [nari-labs/Dia2-2B](https://huggingface.co/nari-labs/Dia2-2B) | Verified, loss weights inferred |
| [Higgs TTS 2](voicestudio/models/higgs_tts2) | 2025 | | [bosonai/higgs-tts-2-3b-base](https://huggingface.co/bosonai/higgs-tts-2-3b-base) | Verified, relay |
| [Spark-TTS](voicestudio/models/spark_tts) | 2025 | [arXiv:2503.01710](https://arxiv.org/abs/2503.01710) | [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | Verified |
| [CosyVoice v1](voicestudio/models/cosyvoice_v1) | 2024 | [arXiv:2407.05407](https://arxiv.org/abs/2407.05407) | [FunAudioLLM/CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M) | Verified, no discriminator |
| [CosyVoice v2](voicestudio/models/cosyvoice_v2) | 2024 | [arXiv:2412.10117](https://arxiv.org/abs/2412.10117) | [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | Verified, no discriminator |
| [F5-TTS](voicestudio/models/f5_tts) | 2024 | [arXiv:2410.06885](https://arxiv.org/abs/2410.06885) | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) | Verified |
| [VoxInstruct](voicestudio/models/vox_instruct) | 2024 | [arXiv:2408.15676](https://arxiv.org/abs/2408.15676) | [niobures/VoxInstruct](https://huggingface.co/niobures/VoxInstruct) | Verified |

### Voice Design

Build a voice from a natural language description, with no reference recording.

| Model | Year | Paper | Hugging Face | Status |
|---|---|---|---|---|
| [Breeze TTS 2](voicestudio/models/breeze_tts) | 2026 | | [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) | Verified |
| [OmniVoice](voicestudio/models/ommivoice) | 2026 | [arXiv:2604.00688](https://arxiv.org/abs/2604.00688) | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | Verified |
| [Qwen3-TTS](voicestudio/models/qwen3_tts) | 2026 | [arXiv:2601.15621](https://arxiv.org/abs/2601.15621) | [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Verified, relay |
| [CosyVoice v3](voicestudio/models/cosyvoice_v3) | 2025 | [arXiv:2505.17589](https://arxiv.org/abs/2505.17589) | [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | Verified, no discriminator |
| [Spark-TTS](voicestudio/models/spark_tts) | 2025 | [arXiv:2503.01710](https://arxiv.org/abs/2503.01710) | [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | Verified |
| [CosyVoice v1](voicestudio/models/cosyvoice_v1) | 2024 | [arXiv:2407.05407](https://arxiv.org/abs/2407.05407) | [FunAudioLLM/CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M) | Verified, no discriminator |
| [CosyVoice v2](voicestudio/models/cosyvoice_v2) | 2024 | [arXiv:2412.10117](https://arxiv.org/abs/2412.10117) | [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | Verified, no discriminator |
| [Parler-TTS](voicestudio/models/parler_tts) | 2024 | [arXiv:2402.01912](https://arxiv.org/abs/2402.01912) | [parler-tts/parler-tts-mini-v1](https://huggingface.co/parler-tts/parler-tts-mini-v1) | Verified |
| [VoxInstruct](voicestudio/models/vox_instruct) | 2024 | [arXiv:2408.15676](https://arxiv.org/abs/2408.15676) | [niobures/VoxInstruct](https://huggingface.co/niobures/VoxInstruct) | Verified |
| [PromptTTS++](voicestudio/models/prompt_tts_pp) | 2023 | [arXiv:2309.08140](https://arxiv.org/abs/2309.08140) | [line-corporation/promptttspp](https://huggingface.co/spaces/line-corporation/promptttspp) | Verified, no discriminator |

PromptTTS++ publishes no model repository. Its only public weights are bundled inside the Space linked
above, which is what its `weight_conversion.convert` downloads.

### Voice Editing

Change the voice of a recording, or rewrite part of it, while keeping the rest.

| Model | Year | Paper | Hugging Face | Status |
|---|---|---|---|---|
| [CosyVoice v3](voicestudio/models/cosyvoice_v3) | 2025 | [arXiv:2505.17589](https://arxiv.org/abs/2505.17589) | [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | Verified, no discriminator |
| [CosyVoice v1](voicestudio/models/cosyvoice_v1) | 2024 | [arXiv:2407.05407](https://arxiv.org/abs/2407.05407) | [FunAudioLLM/CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M) | Verified, no discriminator |
| [CosyVoice v2](voicestudio/models/cosyvoice_v2) | 2024 | [arXiv:2412.10117](https://arxiv.org/abs/2412.10117) | [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | Verified, no discriminator |
| [F5-TTS](voicestudio/models/f5_tts) | 2024 | [arXiv:2410.06885](https://arxiv.org/abs/2410.06885) | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) | Verified |

F5-TTS infills a masked span of an existing recording through its `edit_mask` argument. All three
CosyVoice versions convert the voice of a recording while keeping its content, through
`source_speech_token_ids`.

### Vocoders and Codecs

Not text-to-speech models. These turn features or codes into a waveform, or a waveform into tokens,
and the models above hold them as submodels.

| Model | Year | Paper | Hugging Face | Status |
|---|---|---|---|---|
| [Spark-TTS BiCodec](voicestudio/models/spark_tts_bicodec) | 2025 | [arXiv:2503.01710](https://arxiv.org/abs/2503.01710) | [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | Verified, no discriminator |
| [Vocos](voicestudio/models/vocos) | 2023 | [arXiv:2306.00814](https://arxiv.org/abs/2306.00814) | [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) | Verified, no discriminator |
| [BigVGAN](voicestudio/models/bigvgan) | 2022 | [arXiv:2206.04658](https://arxiv.org/abs/2206.04658) | [nvidia/bigvgan_v2_24khz_100band_256x](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x) | Verified, no discriminator |

### Status legend

| Value | Meaning |
|---|---|
| Verified | Loads its real published checkpoint, generates audio that transcribes back to the text it was given, and its `forward(labels=...)` implements upstream's own training objective term for term. |
| Verified, no discriminator | Verified in the same way, and `forward(labels=...)` returns every term of upstream's objective that does not need a discriminator. The adversarial terms are deliberately absent, which is the `transformers` convention rather than a shortfall: across its 510 model folders no model class carries a GAN discriminator, every shipped vocoder takes no `labels` at all, and DAC, which is adversarially trained upstream, returns only its commitment and codebook terms. The consequence is worth knowing: training one of these vocoders from scratch through `forward` alone would not reproduce the released weights. |
| Verified, loss weights inferred | Verified in the same way, and every term of upstream's objective is implemented. What is not knowable is how loudly each term counts, because upstream publishes no training code, optimizer state or paper. Dia2 pools its 31 acoustic codebooks into one term, following its closest sibling CSM; summing them per codebook the way Higgs Audio V2 does would make that term roughly 31 times heavier. That is a defensible lineage choice, not a fact about Dia2. |
| Verified, relay | The model itself ships in `transformers`; the folder re-exports it, adding only a processor where one was missing. Verified against real weights in the same way. |

Year is the year the model was first published. An empty Paper cell means the release has no arXiv
paper, only code and a model card. `PROJECT.md` carries the per-model verification evidence and the
full list of open items, including one the Status column does not cover: Higgs TTS 3 reports 528
unexpected keys on load, all of them the codec copy bundled in its checkpoint.

---

## 🤝 Contributing

Issues and pull requests are welcome at
[github.com/LatentForge/VoiceStudio](https://github.com/LatentForge/VoiceStudio/issues).

Two files in the repository root are the working documentation, and both are worth reading before
opening a pull request. [CLAUDE.md](CLAUDE.md) is the conventions document: how a model is migrated,
what counts as verification, how files and comments are named, and the rules on dependencies and
licence headers. [PROJECT.md](PROJECT.md) is the running status of the work, including every open
item recorded against a model.

Areas where help is most useful:

- The open items `PROJECT.md` records against each model, which name what is missing and what has
  already been measured about it.
- Inference performance. Nothing here has been tuned for it, and the route is the `transformers` one,
  a static cache and a compiled graph selected through `GenerationConfig`, rather than a per-model
  capture. `PROJECT.md` has the detail.
- More models, migrated the way the existing nineteen were.

---

## 📝 License

Apache License 2.0. See [LICENSE](LICENSE).

Each `modeling_<model>.py` also carries the licence header of the project its code came from, which is
not always Apache 2.0.

The checkpoints are under their own licences, which are not this repository's, and several are more
restrictive than the code that loads them. `BreezeBlue/Breeze-TTS-2` ships a research and
non-commercial licence, `bosonai/higgs-tts-3-4b` likewise, and `FlashLabs/Chroma-4B` is gated behind
an access request. Review a checkpoint's licence before using it.

---

## 🙏 Acknowledgments

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
  almost every file here inherits from.
- [PyTorch](https://pytorch.org/), with `torchaudio` and `torchcodec` behind the `audio` extra.
- [librosa](https://github.com/librosa/librosa) and [NumPy](https://numpy.org/).

---

## 🔗 Links

- Repository: [github.com/LatentForge/VoiceStudio](https://github.com/LatentForge/VoiceStudio)
- Group homepage: [latentforge.github.io](https://latentforge.github.io/)
