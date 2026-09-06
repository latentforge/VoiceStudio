import torch
import torchaudio

_MAX_POINTS = 11025


def show_waveform(audio_path: str | None, waveform: torch.Tensor | None = None, sr: int = 48000):
    try:
        import matplotlib.pyplot as plt
        from IPython.display import Audio
    except ImportError:  # Not in Jupyter notebook
        return None

    if audio_path:
        waveform, sr = torchaudio.load(audio_path)
    elif waveform is not None:
        waveform = waveform.unsqueeze(0) if len(waveform.shape) == 1 else waveform
    else:
        raise ValueError("Either audio_path or waveform must be provided.")

    samples = waveform[0].detach().cpu()
    # Non-overlapping max envelope, so a long clip draws at most _MAX_POINTS bands
    hop = max(1, samples.shape[-1] // _MAX_POINTS)
    envelope = samples[: samples.shape[-1] // hop * hop].abs().reshape(-1, hop).amax(dim=1)
    times = torch.arange(envelope.shape[0], dtype=torch.float32) * hop / sr

    plt.figure(figsize=(10, 4))
    plt.fill_between(times, -envelope, envelope, step="pre", linewidth=0)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

    return Audio(samples, rate=sr)
