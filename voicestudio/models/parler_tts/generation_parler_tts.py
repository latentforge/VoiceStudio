import math
from queue import Queue
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers import LogitsProcessor
from transformers.generation.streamers import BaseStreamer


if TYPE_CHECKING:
    from .modeling_parler_tts import ParlerTTSForConditionalGeneration


class ParlerTTSLogitsProcessor(LogitsProcessor):
    r"""This processor ensures that the delayed pattern mask constraints are respected.

    <Tip warning={true}>

    This logits processor is exclusively compatible with Parler-TTS. 
    See the model documentation for examples.

    </Tip>

    Args:
        eos_token_id (`Union[int, List[int], torch.Tensor]`):
            The id(s) of the *end-of-sequence* token.
        min_eos_p (`float`, *optional*):
            Minimum end of speech threshold.
    """

    def __init__(self, eos_token_id, num_codebooks: int, batch_size: int, device: str = "cpu"):
        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)
        self.eos_token_id = eos_token_id
        self.batch_size = batch_size

        if torch.is_floating_point(eos_token_id) or (eos_token_id < 0).any():
            raise ValueError(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

        self.num_codebooks = num_codebooks
        self.device = device

        self.codebook_idx = torch.arange(self.batch_size*self.num_codebooks, device=self.device)
        self.first_codebooks_unfinished = torch.arange(batch_size, device=device)*num_codebooks
        
        max_codebooks = torch.arange(self.batch_size, device=self.device)*self.num_codebooks + self.num_codebooks -1
        self.max_codebooks = max_codebooks
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        is_eos = torch.isin(input_ids, self.eos_token_id).sum(1)
        
        self.first_codebooks_unfinished = torch.where((is_eos[self.first_codebooks_unfinished]>0) & (self.first_codebooks_unfinished<self.max_codebooks), self.first_codebooks_unfinished+1, self.first_codebooks_unfinished)
                
        # every codebook higher than the first one unfinished will never be eos
        eos_token_mask = self.codebook_idx > self.first_codebooks_unfinished.repeat_interleave(self.num_codebooks)
        scores[eos_token_mask, self.eos_token_id] = -math.inf

        return scores


class ParlerTTSStreamer(BaseStreamer):
    r"""
    Streamer that stores playback-ready audio in a queue, to be used by a downstream application as an iterator. This is
    useful for applications that benefit from accessing the generated audio in a non-blocking way (e.g. in an interactive
    Gradio demo).

    Args:
        model ([`ParlerTTSForConditionalGeneration`]):
            The Parler-TTS model used to generate the audio waveform.
        device (`str`, *optional*):
            The torch device on which to run the computation. If `None`, will default to the device of the model.
        play_steps (`int`, *optional*, defaults to 10):
            The number of generation steps with which to return the generated audio array. Using fewer steps will
            mean the first chunk is ready faster, but will require more codec decoding steps overall. This value
            should be tuned to your device and latency requirements.
        stride (`int`, *optional*):
            The window (stride) between adjacent audio samples. Using a stride between adjacent audio samples reduces
            the hard boundary between them, giving smoother playback. If `None`, will default to a value equivalent to
            play_steps // 6 in the audio space.
        timeout (`float`, *optional*):
            The timeout for the audio queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
            in `.generate()`, when it is called in a separate thread.
    """

    def __init__(
        self,
        model: "ParlerTTSForConditionalGeneration",
        device: Optional[str] = None,
        play_steps: Optional[int] = 10,
        stride: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        self.decoder = model.decoder
        self.audio_encoder = model.audio_encoder
        self.generation_config = model.generation_config
        self.device = device if device is not None else model.device
        self.use_audio_scales = model.use_audio_scales
        self.use_4dim_audio_codes = model.use_4dim_audio_codes
        self.audio_kwargs = {}
        if self.use_audio_scales:
            self.audio_kwargs["audio_scales"] = [None]

        # variables used in the streaming process
        self.play_steps = play_steps
        if stride is not None:
            self.stride = stride
        else:
            hop_length = math.floor(self.audio_encoder.config.sampling_rate / self.audio_encoder.config.frame_rate)
            self.stride = hop_length * (play_steps - self.decoder.num_codebooks) // 6
        self.token_cache = None
        self.to_yield = 0

        # varibles used in the thread process
        self.audio_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def apply_delay_pattern_mask(self, input_ids):
        # build the delay pattern mask for offsetting each codebook prediction by 1 (this behaviour is specific to Parler)
        _, delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids[:, :1],
            bos_token_id=self.generation_config.bos_token_id,
            pad_token_id=self.generation_config.decoder_start_token_id,
            max_length=input_ids.shape[-1],
        )
        # apply the pattern mask to the input ids
        input_ids = self.decoder.apply_delay_pattern_mask(input_ids, delay_pattern_mask)

        # revert the pattern delay mask by filtering the pad token id
        mask = (delay_pattern_mask != self.generation_config.bos_token_id) & (delay_pattern_mask != self.generation_config.pad_token_id)
        input_ids = input_ids[mask].reshape(1, self.decoder.num_codebooks, -1)

        if self.use_4dim_audio_codes:
            # append the frame dimension back to the audio codes
            input_ids = input_ids[None, ...]

        # send the input_ids to the correct device
        input_ids = input_ids.to(self.audio_encoder.device)

        decode_sequentially = (
            self.generation_config.bos_token_id in input_ids
            or self.generation_config.pad_token_id in input_ids
            or self.generation_config.eos_token_id in input_ids
        )
        if not decode_sequentially:
            sample = self.audio_encoder.decode(
                audio_codes=input_ids,
                **self.audio_kwargs,
            ).audio_values
            output_values = sample if sample.ndim == 3 else sample.unsqueeze(0)
        else:
            sample = input_ids[:, 0] if self.use_4dim_audio_codes else input_ids[0]
            sample_mask = ((sample >= self.audio_encoder.config.codebook_size).sum(dim=(0, 1)) == 0) if self.use_4dim_audio_codes else ((sample >= self.audio_encoder.config.codebook_size).sum(dim=0) == 0)
            sample = sample[:, :, sample_mask] if self.use_4dim_audio_codes else sample[:, sample_mask]
            sample = self.audio_encoder.decode(audio_codes=sample[None, ...], **self.audio_kwargs).audio_values
            output_values = sample if sample.ndim == 3 else sample.unsqueeze(0)

        audio_values = output_values[0, 0]
        return audio_values.cpu().float().numpy()

    def put(self, value):
        batch_size = value.shape[0] // self.decoder.num_codebooks
        if batch_size > 1:
            raise ValueError("ParlerTTSStreamer only supports batch size 1")

        if self.token_cache is None:
            self.token_cache = value
        else:
            self.token_cache = torch.concatenate([self.token_cache, value[:, None]], dim=-1)

        if self.token_cache.shape[-1] % self.play_steps == 0:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
            self.on_finalized_audio(audio_values[self.to_yield : -self.stride])
            self.to_yield += len(audio_values) - self.to_yield - self.stride

    def end(self):
        """Flushes any remaining cache and appends the stop symbol."""
        if self.token_cache is not None:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
        else:
            audio_values = np.zeros(self.to_yield)

        self.on_finalized_audio(audio_values[self.to_yield :], stream_end=True)

    def on_finalized_audio(self, audio: np.ndarray, stream_end: bool = False):
        """Put the new audio in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.audio_queue.put(audio, timeout=self.timeout)
        if stream_end:
            self.audio_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.audio_queue.get(timeout=self.timeout)
        if not isinstance(value, np.ndarray) and value == self.stop_signal:
            raise StopIteration()
        else:
            return value
