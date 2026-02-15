# VoiceStudio

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Your Complete Voice Adaptation Research Workspace

---

## 🎯 Overview

VoiceStudio is a unified toolkit for **text-style prompted speech synthesis**, enabling instant voice adaptation and editing through natural language descriptions. Built on cutting-edge research in voice style prompting, LoRA adaptation, and language-audio models.

**Key Features:**
- **Text-Conditional Generation**: Generate voice characteristics using natural language descriptions like "young female voice with warm tone"
- **Multimodal Input**: Support both text descriptions and audio feature vectors
- **Voice Editing**: Modify existing voices with simple instructions (Future Work)
- **Instant Adaptation**: Generate LoRA weights in a single forward pass without fine-tuning
- **Architecture Agnostic**: Works with multiple TTS architectures
- **Zero-shot Generalization**: Adapt to unseen voice characteristics not present in training data
- **Parameter Efficiency**: Minimal computational overhead compared to full model fine-tuning

---

## 🛠️ Installation

### From PyPI (Recommended)
```bash
uv add voicestudio[all]  # Install with all available base TTS models
```

### From Source
```bash
git clone https://github.com/LatentForge/voicestudio.git
cd voicestudio
uv pip install -e ".[all]"
```

### Development Installation
```bash
git clone https://github.com/LatentForge/voicestudio.git
cd voicestudio
uv pip install -e ".[all,web]"
```

### Building and Publishing
```bash
# Build package
uv build

# Upload to PyPI
uv publish
```

---

## 📊 Supported Models

VoiceStudio works with various TTS architectures:

| Model       | Status | Notes                 |
|-------------|--------|-----------------------|
| Parler-TTS  | ✅ Supported | Required further testing |
| Higgs-Audio | ✅ Supported | Required further testing |
| Qwen3-TTS   | ✅ Supported | Required further testing |
| Chroma      | ✅ Supported | Required further testing |
| Spark       | 🔄 Experimental | Coming soon           |
| Dia         | ✅ Supported | Fully tested (by HF)  |
| CozyVoice   | 🔄 Experimental | Coming soon           |
| F5-TTS      | 🔄 Experimental | Coming soon           |

**Add your own model**: See our [Integration Guide](docs/integration.md)

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas we need help with:**
- 🔧 Additional TTS model adapters
- 📚 Documentation improvements
- 🐛 Bug fixes and testing
- 🌍 Multi-language support
- 🎨 New voice editing techniques

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

The base TTS models supported by this project are subject to their own respective licenses. Users are responsible for reviewing and complying with each model’s license before use.

---

## 🙏 Acknowledgments

- **Sakana AI** for the original Text-to-LoRA concept
- **HyperTTS** authors for hypernetwork applications in TTS
- **The open-source community** for tools and datasets
- **CLAP**: Microsoft & LAION-AI for CLAP model
- **LoRA**: Microsoft for LoRA technique
- **HuggingFace**: For transformers library and model hub


## 📚 Citation

If you use VoiceStudio in your research, please cite:

```bibtex
@software{voicestudio2026,
  title={VoiceStudio: A Unified Toolkit for Voice Style Adaptation},
  author={Your Name},
  year={2026},
  url={https://github.com/LatentForge/voicestudio}
}
```
```bibtex
@article{t2a-lora-2025,
  title={T2A-LoRA: Text-to-Audio LoRA Generation via Hypernetworks for Real-time Voice Adaptation},
  author={LatentForge},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```

## 🔗 Links

- **Paper**: [arXiv:2501.XXXXX](https://arxiv.org/abs/2501.XXXXX)
- **Demo**: [https://latentforge.github.io/VoiceStudio](https://latentforge.github.io/VoiceStudio)
- **Documentation**: [https://latentforge.github.io/VoiceStudio](https://latentforge.github.io/VoiceStudio)
- **Models**: [HuggingFace Hub](https://huggingface.co/LatentForge)

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/LatentForge/VoiceStudio/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LatentForge/VoiceStudio/discussions)
- **Email**: contact@latentforge.org

---

<p align="center">
  <img src="https://img.shields.io/github/stars/LatentForge/VoiceStudio?style=social" alt="Stars">
  <img src="https://img.shields.io/github/forks/LatentForge/VoiceStudio?style=social" alt="Forks">
  <img src="https://img.shields.io/github/watchers/LatentForge/VoiceStudio?style=social" alt="Watchers">
</p>

<p align="center">
  <strong>Made with ❤️ by LatentForge Team</strong>
</p>
