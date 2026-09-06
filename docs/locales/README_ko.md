# VoiceStudio

이 문서의 영문 버전은 [README.md](../../README.md)에서 확인할 수 있습니다.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

음성 복제, 디자인 및 편집을 위한 통합 툴킷입니다.

---

## 🎯 개요

음성 합성 연구는 도구 생태계의 파편화로 인해 속도를 내지 못하고 있습니다. 새로운 모델이 나올 때마다 각자의 저장소, 런타임, 체크포인트 포맷, 추론 스크립트, 특정 버전으로 고정된 의존성 패키지를 따로 두고 있으며, 대부분 실행 가능한 가중치만 제공할 뿐 학습할 수 있는 방법은 제공하지 않습니다. 두 모델을 비교하려면 두 개의 코드베이스를 새로 익혀야 하고, 특정 모델을 기반으로 연구를 발전시키려면 해당 코드베이스 전체를 그대로 떠안아야 합니다.

VoiceStudio는 이러한 부담을 없앱니다. 이곳의 모든 모델은 일반적인 `transformers` 모델입니다. 즉, `PreTrainedConfig`, `PreTrainedModel`, `Processor`로 구성되어 `from_pretrained`로 로드하고, `generate`로 실행하며, `forward(labels=...)`로 학습합니다. 모델 교체는 클래스 이름 하나를 바꾸는 것만으로 충분합니다. 여러 모델의 비교는 루프 하나로 가능하며, 파인튜닝은 이미 보유하고 계신 학습 코드를 그대로 활용하면 됩니다.

**주요 특징:**
- **단일 API**: 모든 모델이 자체 `Processor`로부터 입력을 받고 동일한 프로세서가 디코딩하는 오디오를 반환하므로, 클래스 이름 변경만으로 모델을 전환할 수 있습니다.
- **조합 가능한 구조**: 모델들이 서로를 일반 서브모델로 포함합니다. Parler-TTS는 `DacModel`을, Chroma는 `MimiModel`을 포함하며, F5-TTS는 체크포인트 학습에 사용된 `VocosModel` 또는 `BigVGANModel`을 서브모델로 포함합니다.
- **재구현 대신 상속**: 병렬 사본을 유지하는 대신 `llama`, `qwen3`, `csm`, `mimi`, `dac`, `speecht5` 및 십여 개 이상의 기본 모델과 상호 간의 상속을 기반으로 재구성했습니다.
- **추론 전용이 아닌 학습 지원**: 모든 모델이 손실(loss)을 반환하며, 모델 구조를 추측하는 대신 업스트림 프로젝트 자체의 트레이너에서 읽어온 학습 목적함수를 정확히 반영합니다.
- **공개 가중치 기반 검증**: 실제 체크포인트로부터 로드하여 음성을 생성한 뒤, 해당 오디오를 다시 텍스트로 전사(transcribe)하여 입력 텍스트와 비교 검증했습니다.
- **직접 로드 지원**: 사용자가 별도의 변환 단계를 거칠 필요 없이 공식 저장소 ID에 대해 `from_pretrained`를 바로 호출할 수 있습니다.
- **최소화된 의존성**: 마이그레이션 과정에서 업스트림의 import를 추가하는 대신 제거하여, `torch`, `transformers` 외에 6개의 패키지만 남겼습니다.

---

## 🛠️ 설치 방법

Python 3.11 이상 및 PyTorch 2.8 이상이 필요합니다.

### 소스 코드에서 설치

```bash
git clone https://github.com/LatentForge/VoiceStudio.git
cd VoiceStudio
uv sync
```

### 개발용 설치

```bash
git clone https://github.com/LatentForge/VoiceStudio.git
cd VoiceStudio
uv sync --all-extras
```

`pip install` 대신 `uv sync`를 사용해야 합니다. 여러 의존성 패키지가 `[tool.uv.sources]`의 특정 소스로 고정되어 있으며, `pip`은 해당 설정을 무시합니다. PyPI에 등록되어 있는 기존 `voicestudio` 배포판은 본 프로젝트 이전의 버전이므로 아래에 소개된 모델들을 포함하고 있지 않습니다.

`uv sync --extra <name>`으로 개별 선택하거나 `uv sync --all-extras`로 한 번에 설치할 수 있는 선택적 extras 목록입니다:

| Extra | 설치 패키지 | 용도 |
|---|---|---|
| `train` | `accelerate`, `wandb`, `matplotlib`, `notebook`, `ipywidgets` | 학습 실행 및 노트북 환경 |
| `eval` | `pyworld`, `jiwer` | CosyVoice의 보코더 목적함수를 위한 f0 추출, 그리고 모델 검증에 사용되는 단어 오류율(WER) 측정 |
| `kernels` | `transformers[kernels]` | flash attention 및 기타 fused 커널 |
| `omni` | `pillow`, `torchvision` | 프로세서가 `Qwen2_5OmniProcessor`를 상속하는 Chroma 모델 |
| `native` | `torchnative` | 온디바이스 추론 |
| `web` | `fastapi` | 웹 프론트엔드 |
| `all` | `train`, `eval`, `kernels`, `omni`, `web` | `native`를 제외한 모든 항목 |

---

## 🚀 사용법

`transformers`에 이미 포함되어 있는 모델은 공식 공개 저장소로부터 직접 로드할 수 있습니다:

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

업스트림 릴리스에서 독자적인 가중치 레이아웃을 사용하는 모델은 해당 폴더의 `weight_conversion.convert`를 통해 한 번 변환한 후 동일한 방식으로 로드할 수 있습니다. 각 모델 폴더의 README에는 변환 호출 방법, 동작에 필수적인 generation 인자, 학습 목적함수, 그리고 업스트림에서 가져오지 않은 항목들이 문서화되어 있습니다.

---

## 📊 모델

아래의 모든 모델은 실제 공개된 가중치를 로드하여 정상 동작을 확인했습니다. 모델 이름을 클릭하면 사용법, 목적함수 및 미해결 항목이 정리된 각 모델 폴더의 README로 이동할 수 있습니다.

### 음성 복제

참조 녹음의 음성을 재현합니다.

| 모델 | 연도 | 논문 | Hugging Face | 상태 |
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

### 음성 디자인

참조 녹음 없이 자연어 설명으로부터 음성을 생성합니다.

| 모델 | 연도 | 논문 | Hugging Face | 상태 |
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

PromptTTS++는 별도의 모델 저장소를 공개하지 않았습니다. 공개된 유일한 가중치는 위 링크의 Space 내에 포함되어 있으며, 해당 모델의 `weight_conversion.convert`가 이를 다운로드합니다.

### 음성 편집

녹음의 나머지 부분을 유지하면서 목소리를 변경하거나 일부 구간을 수정합니다.

| 모델 | 연도 | 논문 | Hugging Face | 상태 |
|---|---|---|---|---|
| [CosyVoice v3](voicestudio/models/cosyvoice_v3) | 2025 | [arXiv:2505.17589](https://arxiv.org/abs/2505.17589) | [FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | Verified, no discriminator |
| [CosyVoice v1](voicestudio/models/cosyvoice_v1) | 2024 | [arXiv:2407.05407](https://arxiv.org/abs/2407.05407) | [FunAudioLLM/CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M) | Verified, no discriminator |
| [CosyVoice v2](voicestudio/models/cosyvoice_v2) | 2024 | [arXiv:2412.10117](https://arxiv.org/abs/2412.10117) | [FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | Verified, no discriminator |
| [F5-TTS](voicestudio/models/f5_tts) | 2024 | [arXiv:2410.06885](https://arxiv.org/abs/2410.06885) | [SWivid/F5-TTS](https://huggingface.co/SWivid/F5-TTS) | Verified |

F5-TTS는 `edit_mask` 인자를 통해 기존 녹음의 마스킹된 구간을 채워 넣습니다(infill). 세 가지 버전의 CosyVoice는 모두 `source_speech_token_ids`를 통해 발화 내용을 유지하면서 녹음의 음색을 변환합니다.

### 보코더 및 코덱

텍스트 음성 변환(TTS) 모델이 아닙니다. 피처나 코드를 파형으로 변환하거나 파형을 토큰으로 변환하는 모델이며, 위의 모델들이 이들을 서브모델로 포함하고 있습니다.

| 모델 | 연도 | 논문 | Hugging Face | 상태 |
|---|---|---|---|---|
| [Spark-TTS BiCodec](voicestudio/models/spark_tts_bicodec) | 2025 | [arXiv:2503.01710](https://arxiv.org/abs/2503.01710) | [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) | Verified, no discriminator |
| [Vocos](voicestudio/models/vocos) | 2023 | [arXiv:2306.00814](https://arxiv.org/abs/2306.00814) | [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) | Verified, no discriminator |
| [BigVGAN](voicestudio/models/bigvgan) | 2022 | [arXiv:2206.04658](https://arxiv.org/abs/2206.04658) | [nvidia/bigvgan_v2_24khz_100band_256x](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x) | Verified, no discriminator |

### 상태 범례

| 값 | 의미 |
|---|---|
| Verified | 실제 공개된 체크포인트를 로드하여 주어진 텍스트대로 전사되는 오디오를 생성하며, `forward(labels=...)`가 업스트림 자체의 학습 목적함수를 항별로 그대로 구현함을 확인했습니다. |
| Verified, no discriminator | 동일한 방식으로 검증되었으며, `forward(labels=...)`가 discriminator를 필요로 하지 않는 업스트림 목적함수의 모든 항을 반환합니다. 적대적 항이 없는 것은 미달이 아니라 `transformers`의 관례입니다. 510개 모델 폴더 전체에서 GAN discriminator를 모델 클래스에 담은 사례가 없고, 배포되는 보코더는 모두 `labels`를 아예 받지 않으며, 업스트림에서 적대적으로 학습되는 DAC조차 commitment와 codebook 항만 반환합니다. 다만 결과 하나는 알아둘 필요가 있습니다. 이 보코더들을 `forward`만으로 처음부터 학습하면 공개된 가중치를 재현하지 못합니다. |
| Verified, loss weights inferred | 동일한 방식으로 검증되었으며 업스트림 목적함수의 모든 항이 구현되어 있습니다. 알 수 없는 것은 각 항이 서로 얼마나 무겁게 작용하는가입니다. 업스트림이 학습 코드도, 옵티마이저 상태도, 논문도 공개하지 않았기 때문입니다. Dia2는 31개 음향 코드북을 하나의 항으로 묶는데, 이는 가장 가까운 형제인 CSM을 따른 것입니다. Higgs Audio V2처럼 코드북별로 합산하면 그 항이 약 31배 무거워집니다. 이는 Dia2에 대한 사실이 아니라 방어 가능한 계보 선택입니다. |
| Verified, relay | 모델 자체가 이미 `transformers`에 포함되어 있어 폴더에서는 이를 re-export하며, 프로세서가 없는 경우에만 추가했습니다. 실제 가중치를 대상으로 동일하게 검증되었습니다. |

연도(Year)는 모델이 처음 공개된 연도입니다. 논문(Paper) 칸이 비어 있는 것은 arXiv 논문 없이 코드와 모델 카드만 공개되었음을 의미합니다. `PROJECT.md`에는 모델별 검증 근거와 미해결 항목 전체 목록이 정리되어 있으며, 여기에는 상태 열에서 다루지 않는 항목 하나도 포함됩니다. 즉, Higgs TTS 3 로드 시 보고되는 528개의 unexpected key이며, 모두 체크포인트에 포함된 코덱 사본입니다.

---

## 🤝 기여하기

이슈 및 풀 리퀘스트는
[github.com/LatentForge/VoiceStudio](https://github.com/LatentForge/VoiceStudio/issues)를 통해 언제든 환영합니다.

저장소 루트에 있는 두 파일은 개발 작업 문서이며, 풀 리퀘스트를 열기 전에 모두 읽어보실 것을 권장합니다. [CLAUDE.md](CLAUDE.md)는 모델 마이그레이션 방법, 검증 기준, 파일 및 주석 명명 규칙, 의존성 및 라이선스 헤더에 관한 규칙이 담긴 컨벤션 문서입니다. [PROJECT.md](PROJECT.md)는 모델별로 기록된 모든 미해결 항목을 포함하여 작업의 진행 상황을 관리하는 문서입니다.

도움이 가장 유용한 분야는 다음과 같습니다:

- `PROJECT.md`가 미해결로 기록한 항목들. 코드보다 결정이 먼저 필요한 것은 두 가지입니다. CosyVoice v3의 텍스트 정규화기는 업스트림의 `ttsfrd`와 `wetext`가 비공개 wheel과 컴파일된 문법이라 인라인할 수 있는 로직이 아니고, 두 번째로 공개된 `llm.rl.pt` 체크포인트는 문자 오류율과 화자 유사도를 맞바꾸는데 아직 선택할 방법이 없습니다.
- 추론 성능: 현재는 별도의 최적화가 되어 있지 않으며, 모델별 캡처 대신 `GenerationConfig`를 통해 선택되는 정적 캐시(static cache) 및 컴파일된 그래프(compiled graph)라는 `transformers` 표준 경로를 따르고 있습니다. 자세한 내용은 `PROJECT.md`에 설명되어 있습니다.
- 추가 모델 마이그레이션: 기존 열아홉 개 모델이 마이그레이션된 방식을 따라 새로운 모델을 추가하는 작업입니다.

---

## 📝 라이선스

Apache License 2.0을 따릅니다. [LICENSE](LICENSE)를 참고하시기 바랍니다.

각 `modeling_<model>.py` 파일에는 해당 코드의 원본 프로젝트 라이선스 헤더가 포함되어 있으며, 이는 항상 Apache 2.0인 것은 아닙니다.

체크포인트는 본 저장소의 라이선스가 아닌 자체 라이선스를 따르며, 일부는 이를 로드하는 코드보다 더 제한적인 라이선스를 적용받습니다. `BreezeBlue/Breeze-TTS-2`는 연구 및 비상업용 라이선스로 제공되며, `bosonai/higgs-tts-3-4b` 역시 마찬가지입니다. `FlashLabs/Chroma-4B`는 접근 요청 승인이 필요합니다. 체크포인트를 사용하기 전에 해당 라이선스를 반드시 확인하시기 바랍니다.

---

## 🙏 감사의 글

본 저장소는 다양한 연구자들의 연구 성과를 하나의 API 아래로 통합한 것입니다. 포함된 모델들의 출처는 다음과 같습니다:

- [NVIDIA](https://github.com/NVIDIA/BigVGAN): BigVGAN
- [BreezeBlue](https://github.com/breezeblue-ai/breeze-tts): Breeze TTS 2
- [FlashLabs](https://github.com/FlashLabs-AI-Corp/FlashLabs-Chroma): Chroma
- [FunAudioLLM](https://github.com/FunAudioLLM/CosyVoice): CosyVoice v1, v2 및 v3
- [Nari Labs](https://github.com/nari-labs): [Dia](https://github.com/nari-labs/dia) 및 [Dia2](https://github.com/nari-labs/dia2)
- [SWivid](https://github.com/SWivid/F5-TTS): F5-TTS 및 E2-TTS
- [Boson AI](https://github.com/boson-ai/higgs-audio): Higgs TTS 2 및 Higgs TTS 3
- [k2-fsa](https://github.com/k2-fsa/OmniVoice): OmniVoice
- [Hugging Face](https://github.com/huggingface/parler-tts): Parler-TTS
- [LINE](https://github.com/line/promptttspp): PromptTTS++
- [Qwen](https://github.com/QwenLM/Qwen3-TTS): Qwen3-TTS
- [SparkAudio](https://github.com/SparkAudio/Spark-TTS): Spark-TTS 및 BiCodec
- [gemelo.ai](https://github.com/gemelo-ai/vocos): Vocos
- [THU-HCSI](https://github.com/thuhcsi/VoxInstruct): VoxInstruct

그리고 코드의 기반이 된 라이브러리들:

- [Hugging Face `transformers`](https://github.com/huggingface/transformers): 이 저장소의 거의 모든 파일이 상속하는 모델 클래스의 기반입니다.
- [PyTorch](https://pytorch.org/): `torchaudio` 및 `torchcodec`과 함께 사용됩니다.
- [librosa](https://github.com/librosa/librosa) 및 [NumPy](https://numpy.org/).

---

## 🔗 링크

- 저장소: [github.com/LatentForge/VoiceStudio](https://github.com/LatentForge/VoiceStudio)
- 그룹 홈페이지: [latentforge.github.io](https://latentforge.github.io/)
