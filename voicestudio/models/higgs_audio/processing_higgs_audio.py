# Copyright 2024 Boson AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications by LatentForge:
# - Implement transformers standard processor interface

import base64
import torch
import torch.nn.functional as F
import numpy as np
import librosa
from io import BytesIO
from typing import Optional, Union, List, Dict, Any
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoProcessor
from transformers.processing_utils import ProcessorMixin

try:
    from ...audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
    from ...model.higgs_audio.utils import revert_delay_pattern, build_delay_pattern_mask
except ImportError:
    from voicestudio._boson.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
    from voicestudio._boson.model.higgs_audio.utils import revert_delay_pattern, build_delay_pattern_mask

from .configuration_higgs_audio import HiggsAudioConfig


def _ceil_to_nearest(n, round_to):
    """Round n up to the nearest multiple of round_to."""
    return (n + round_to - 1) // round_to * round_to


@dataclass
class HiggsAudioBatchInput:
    """Output format from processor containing all model inputs."""
    input_ids: torch.LongTensor  # shape (bsz, seq_len)
    attention_mask: torch.Tensor  # shape (bsz, seq_len)
    audio_features: Optional[torch.Tensor] = None  # shape (num_audio_in, feature_dim, max_mel_seq_len)
    audio_feature_attention_mask: Optional[torch.Tensor] = None  # shape (num_audio_in, max_mel_seq_len)
    audio_out_ids: Optional[torch.LongTensor] = None  # shape (num_codebooks, audio_out_total_length)
    audio_out_ids_start: Optional[torch.LongTensor] = None  # shape (num_audio_out,)
    audio_out_ids_start_group_loc: Optional[torch.LongTensor] = None  # shape (num_audio_out,)
    audio_in_ids: Optional[torch.LongTensor] = None  # shape (num_codebooks, audio_in_total_length)
    audio_in_ids_start: Optional[torch.LongTensor] = None  # shape (num_audio_in,)


class HiggsAudioProcessor(ProcessorMixin):
    """
    Processor for HiggsAudio model that handles tokenization and audio processing.

    This processor combines:
    - Text tokenizer (for text tokens)
    - Audio tokenizer (for audio encoding/decoding)
    - Whisper processor (for audio feature extraction)
    - All batch processing and audio preprocessing logic

    Example (Transformers Standard):
        ```python
        processor = HiggsAudioProcessor.from_pretrained(
            "bosonai/higgs-audio-v2-generation-3B-base",
            audio_tokenizer_path="bosonai/higgs-audio-v2-tokenizer"
        )

        # Standard messages format
        messages = [
            {"role": "system", "content": "Generate audio following instruction."},
            {"role": "user", "content": "Hello world!"},
        ]
        inputs = processor(messages)
        outputs = model.generate(**inputs)
        ```

    Example (with audio input):
        ```python
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this audio?"},
                {"type": "audio", "audio_url": "path/to/audio.wav"}
            ]
        }]
        inputs = processor(messages)
        ```
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        audio_tokenizer,
        whisper_processor: Optional[AutoProcessor] = None,
        audio_in_token_id: Optional[int] = None,
        audio_out_token_id: Optional[int] = None,
        audio_stream_bos_id: Optional[int] = None,
        audio_stream_eos_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        encode_whisper_embed: bool = False,
        use_delay_pattern: bool = True,
        audio_num_codebooks: int = 1,
        return_audio_in_tokens: bool = False,
        round_to: int = 1,
        chunk_size_seconds: int = 30,
    ):
        """
        Initialize the processor.

        Args:
            tokenizer: Text tokenizer
            audio_tokenizer: Audio tokenizer for encoding/decoding audio
            whisper_processor: Whisper processor for audio feature extraction (optional)
            audio_in_token_id: Token ID for audio input placeholder
            audio_out_token_id: Token ID for audio output placeholder
            audio_stream_bos_id: Token ID for audio stream beginning
            audio_stream_eos_id: Token ID for audio stream ending
            pad_token_id: Padding token ID
            encode_whisper_embed: Whether to encode Whisper embeddings
            use_delay_pattern: Whether to use delay pattern for audio
            audio_num_codebooks: Number of audio codebooks
            return_audio_in_tokens: Whether to return audio input tokens
            round_to: Round sequence length to nearest multiple
            chunk_size_seconds: Maximum audio chunk size in seconds
        """
        super().__init__(tokenizer=tokenizer)

        # Manually set non-standard attributes
        self.audio_tokenizer = audio_tokenizer
        self.whisper_processor = whisper_processor

        # Set special token IDs from tokenizer if not provided
        if audio_in_token_id is None:
            audio_in_token_id = tokenizer.convert_tokens_to_ids("<|audio_in|>")
        if audio_out_token_id is None:
            audio_out_token_id = tokenizer.convert_tokens_to_ids("<|audio_out|>")
        if audio_stream_bos_id is None:
            audio_stream_bos_id = tokenizer.convert_tokens_to_ids("<|audio_stream_bos|>")
        if audio_stream_eos_id is None:
            audio_stream_eos_id = tokenizer.convert_tokens_to_ids("<|audio_stream_eos|>")
        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id

        self.audio_in_token_id = audio_in_token_id
        self.audio_out_token_id = audio_out_token_id
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.pad_token_id = pad_token_id
        self.encode_whisper_embed = encode_whisper_embed
        self.use_delay_pattern = use_delay_pattern
        self.audio_num_codebooks = audio_num_codebooks
        self.return_audio_in_tokens = return_audio_in_tokens
        self.round_to = round_to

        if encode_whisper_embed and whisper_processor:
            self.chunk_size_seconds = chunk_size_seconds
            self.chunk_size_samples = int(chunk_size_seconds * whisper_processor.feature_extractor.sampling_rate)
        else:
            self.chunk_size_seconds = None
            self.chunk_size_samples = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        audio_tokenizer_path: Optional[str] = None,
        device: str = "cuda",
        **kwargs
    ):
        """
        Load processor from pretrained model.

        Args:
            pretrained_model_name_or_path: Path to pretrained model
            audio_tokenizer_path: Path to audio tokenizer (required)
            device: Device to load components on
            **kwargs: Additional arguments

        Returns:
            HiggsAudioProcessor instance
        """
        if audio_tokenizer_path is None:
            raise ValueError("audio_tokenizer_path must be provided")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        # Load audio tokenizer
        audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_path, device=device)

        # Load config to get special token IDs
        config = HiggsAudioConfig.from_pretrained(pretrained_model_name_or_path)

        # Load whisper processor if needed
        whisper_processor = None
        if config.encode_whisper_embed:
            whisper_processor = AutoProcessor.from_pretrained(
                "openai/whisper-large-v3-turbo",
                trust_remote=True,
                device=device,
            )

        return cls(
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer,
            whisper_processor=whisper_processor,
            audio_in_token_id=config.audio_in_token_idx,
            audio_out_token_id=config.audio_out_token_idx,
            audio_stream_bos_id=config.audio_stream_bos_id,
            audio_stream_eos_id=config.audio_stream_eos_id,
            pad_token_id=config.pad_token_id,
            encode_whisper_embed=config.encode_whisper_embed,
            use_delay_pattern=config.use_delay_pattern,
            audio_num_codebooks=config.audio_num_codebooks,
            return_audio_in_tokens=False,
            round_to=1,
        )

    def __call__(
        self,
        messages: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        force_audio_gen: bool = False,
        return_tensors: str = "pt",
        padding: bool = True,
    ):
        """
        Process messages into model inputs.

        Supports both single sample and batch processing.

        Args:
            messages: Single list of messages or batch of message lists
            force_audio_gen: Whether to force audio generation mode
            return_tensors: Return tensor type ("pt" for PyTorch)
            padding: Whether to pad sequences

        Returns:
            HiggsAudioBatchInput with all model inputs
        """
        # Determine if batch or single sample
        if isinstance(messages[0], list):
            # Batch of message lists
            batch_messages = messages
        else:
            # Single message list - wrap in batch
            batch_messages = [messages]

        # Process each sample in batch
        processed_samples = []
        for msgs in batch_messages:
            sample_data = self._process_single_sample(msgs, force_audio_gen)
            processed_samples.append(sample_data)

        # Collate batch
        batch_input = self._collate_batch(processed_samples, padding=padding)

        return batch_input

    def _process_single_sample(
        self,
        messages: List[Dict[str, Any]],
        force_audio_gen: bool = False,
    ) -> Dict[str, Any]:
        """Process a single sample (list of messages) into tokens and audio info."""
        # Tokenize messages directly following ChatML format
        input_tokens = []
        audio_items = []  # List of (audio_url, raw_audio, role) tuples

        for turn_id, message in enumerate(messages):
            role = message.get("role", "user")
            content = message.get("content", "")

            # Add role header
            if turn_id == 0:
                prefix = f"<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n"
            else:
                prefix = f"<|start_header_id|>{role}<|end_header_id|>\n\n"

            prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            input_tokens.extend(prefix_tokens)

            # Process content
            if isinstance(content, str):
                # Simple text content
                text_tokens = self.tokenizer.encode(content, add_special_tokens=False)
                input_tokens.extend(text_tokens)
            elif isinstance(content, list):
                # Multimodal content (text + audio)
                for item in content:
                    if isinstance(item, str):
                        # Text string
                        text_tokens = self.tokenizer.encode(item, add_special_tokens=False)
                        input_tokens.extend(text_tokens)
                    elif isinstance(item, dict):
                        item_type = item.get("type", "text")
                        if item_type == "text":
                            # Text dict
                            text = item.get("text", "")
                            text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
                            input_tokens.extend(text_tokens)
                        elif item_type == "audio":
                            # Audio dict
                            audio_url = item.get("audio_url", "placeholder")
                            raw_audio = item.get("raw_audio")
                            audio_items.append((audio_url, raw_audio, role))

                            # Add audio placeholder tokens
                            if role in ["user", "system"]:
                                audio_tokens = self.tokenizer.encode(
                                    "<|audio_bos|><|AUDIO|><|audio_eos|>",
                                    add_special_tokens=False
                                )
                            else:  # assistant
                                audio_tokens = self.tokenizer.encode(
                                    "<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>",
                                    add_special_tokens=False
                                )
                            input_tokens.extend(audio_tokens)

            # Add end-of-turn token
            postfix_tokens = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)
            input_tokens.extend(postfix_tokens)

        # Add assistant header and optionally force audio generation
        postfix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if force_audio_gen:
            postfix += "<|audio_out_bos|>"
        postfix_tokens = self.tokenizer.encode(postfix, add_special_tokens=False)
        input_tokens.extend(postfix_tokens)

        # Process audio items - load and encode
        audio_in_items = []  # (waveform, sample_rate, audio_ids)
        audio_out_items = []  # (audio_ids,)

        for audio_url, raw_audio, role in audio_items:
            raw_audio_data = None

            if audio_url not in ["placeholder", ""]:
                raw_audio_data, sr = librosa.load(audio_url, sr=None)
            elif raw_audio is not None:
                raw_audio_data, sr = librosa.load(
                    BytesIO(base64.b64decode(raw_audio)),
                    sr=None
                )

            if raw_audio_data is not None:
                # Encode audio to tokens
                if sr != self.audio_tokenizer.sampling_rate:
                    raw_audio_data = librosa.resample(
                        raw_audio_data,
                        orig_sr=sr,
                        target_sr=self.audio_tokenizer.sampling_rate
                    )
                    sr = self.audio_tokenizer.sampling_rate

                audio_ids = self.audio_tokenizer.encode(
                    raw_audio_data,
                    self.audio_tokenizer.sampling_rate
                ).squeeze(0).cpu()

                # Categorize by role
                if role in ["user", "system"]:
                    audio_in_items.append((
                        torch.from_numpy(raw_audio_data),
                        sr,
                        audio_ids
                    ))
                else:  # assistant
                    audio_out_items.append(audio_ids)

        return {
            "input_ids": torch.LongTensor(input_tokens),
            "audio_in_items": audio_in_items,
            "audio_out_items": audio_out_items,
        }

    def _collate_batch(
        self,
        samples: List[Dict[str, Any]],
        padding: bool = True,
    ) -> HiggsAudioBatchInput:
        """Collate processed samples into a batch with all necessary preprocessing."""

        # Handle audio chunking for long audio
        processed_samples = []
        for sample in samples:
            if self.encode_whisper_embed and self.chunk_size_samples:
                sample = self._chunk_long_audio(sample)
            processed_samples.append(sample)

        # Calculate max sequence length
        if padding:
            max_seq_length = _ceil_to_nearest(
                max([len(s["input_ids"]) for s in processed_samples]),
                self.round_to
            )
        else:
            max_seq_length = None

        # Extract audio inputs and outputs
        audio_in_wv_list = []
        audio_in_ids_list = []
        audio_out_ids_list = []
        audio_out_group_locs = []

        for batch_idx, sample in enumerate(processed_samples):
            # Collect audio-in waveforms for Whisper
            for wv, sr, audio_ids in sample["audio_in_items"]:
                if self.encode_whisper_embed:
                    # Resample if needed
                    if sr != self.whisper_processor.feature_extractor.sampling_rate:
                        wv_numpy = librosa.resample(
                            wv.numpy(),
                            orig_sr=sr,
                            target_sr=self.whisper_processor.feature_extractor.sampling_rate
                        )
                    else:
                        wv_numpy = wv.numpy()
                    audio_in_wv_list.append(wv_numpy)

                if self.return_audio_in_tokens:
                    audio_in_ids_list.append(audio_ids)

            # Collect audio-out tokens
            for audio_ids in sample["audio_out_items"]:
                audio_out_ids_list.append(audio_ids)
                audio_out_group_locs.append(batch_idx)

        # Process Whisper features
        if len(audio_in_wv_list) > 0 and self.encode_whisper_embed:
            feature_ret = self.whisper_processor.feature_extractor(
                audio_in_wv_list,
                sampling_rate=self.whisper_processor.feature_extractor.sampling_rate,
                return_attention_mask=True,
                padding="max_length",
            )
            audio_features = torch.from_numpy(feature_ret["input_features"])
            audio_feature_attention_mask = torch.from_numpy(feature_ret["attention_mask"])
        else:
            audio_features = None
            audio_feature_attention_mask = None

        # Process audio input tokens
        if len(audio_in_ids_list) > 0:
            processed_audio_in = []
            for audio_ids in audio_in_ids_list:
                # Add BOS/EOS
                audio_codes = torch.cat([
                    torch.full((audio_ids.shape[0], 1), self.audio_stream_bos_id, dtype=torch.long),
                    audio_ids,
                    torch.full((audio_ids.shape[0], 1), self.audio_stream_eos_id, dtype=torch.long),
                ], dim=1)

                # Apply delay pattern if needed
                if self.use_delay_pattern:
                    audio_codes = build_delay_pattern_mask(
                        audio_codes.unsqueeze(0),
                        bos_token_id=self.audio_stream_bos_id,
                        pad_token_id=self.audio_stream_eos_id,
                    )[0].squeeze(0)

                processed_audio_in.append(audio_codes)

            audio_in_ids = torch.cat(processed_audio_in, dim=1).long()
            audio_in_ids_start = torch.cumsum(
                torch.tensor([0] + [a.shape[1] for a in processed_audio_in[:-1]]), dim=0
            )
        else:
            audio_in_ids = None if not self.return_audio_in_tokens else torch.zeros((0, 0), dtype=torch.long)
            audio_in_ids_start = None if not self.return_audio_in_tokens else torch.zeros(0, dtype=torch.long)

        # Process audio output tokens
        if len(audio_out_ids_list) > 0:
            processed_audio_out = []
            for audio_ids in audio_out_ids_list:
                # Add BOS/EOS
                audio_codes = torch.cat([
                    torch.full((audio_ids.shape[0], 1), self.audio_stream_bos_id, dtype=torch.long),
                    audio_ids,
                    torch.full((audio_ids.shape[0], 1), self.audio_stream_eos_id, dtype=torch.long),
                ], dim=1)

                # Apply delay pattern if needed
                if self.use_delay_pattern:
                    audio_codes = build_delay_pattern_mask(
                        audio_codes.unsqueeze(0),
                        bos_token_id=self.audio_stream_bos_id,
                        pad_token_id=self.audio_stream_eos_id,
                    )[0].squeeze(0)

                processed_audio_out.append(audio_codes)

            audio_out_ids = torch.cat(processed_audio_out, dim=1).long()
            audio_out_ids_start = torch.cumsum(
                torch.tensor([0] + [a.shape[1] for a in processed_audio_out[:-1]]), dim=0
            )
            audio_out_ids_start_group_loc = torch.tensor(audio_out_group_locs, dtype=torch.long)
        else:
            audio_out_ids = torch.zeros((0, 0), dtype=torch.long)
            audio_out_ids_start = torch.zeros(0, dtype=torch.long)
            audio_out_ids_start_group_loc = None

        # Apply codebook limit
        if self.audio_num_codebooks is not None:
            if audio_in_ids is not None and audio_in_ids.shape[0] > 0:
                audio_in_ids = audio_in_ids[:self.audio_num_codebooks]
            if audio_out_ids.shape[0] > 0:
                audio_out_ids = audio_out_ids[:self.audio_num_codebooks]

        # Pad input sequences
        if padding and max_seq_length:
            input_ids = torch.stack([
                F.pad(s["input_ids"], (0, max_seq_length - len(s["input_ids"])), value=self.pad_token_id)
                for s in processed_samples
            ])
            attention_mask = torch.stack([
                F.pad(torch.ones_like(s["input_ids"]), (0, max_seq_length - len(s["input_ids"])), value=0)
                for s in processed_samples
            ])
        else:
            # No padding - return as list (not typical for generation)
            input_ids = torch.stack([s["input_ids"] for s in processed_samples])
            attention_mask = torch.stack([torch.ones_like(s["input_ids"]) for s in processed_samples])

        return HiggsAudioBatchInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            audio_out_ids_start_group_loc=audio_out_ids_start_group_loc,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
        )

    def _chunk_long_audio(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Split long audio into chunks and duplicate corresponding tokens."""
        # For now, just return as-is
        # Full chunking logic would duplicate tokens for audio > chunk_size_seconds
        # This is complex and mainly needed for training on very long audio
        return sample

    def decode_audio(self, audio_tokens: torch.Tensor) -> np.ndarray:
        """
        Decode audio tokens to waveform.

        Args:
            audio_tokens: Audio tokens of shape (num_codebooks, seq_len)

        Returns:
            Audio waveform as numpy array
        """
        audio_codebook_size = 2048  # Should come from config
        vq_code = revert_delay_pattern(audio_tokens).clip(0, audio_codebook_size - 1)[:, 1:-1]
        wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
        return wv_numpy

    @property
    def sampling_rate(self):
        """Get the sampling rate of the audio tokenizer."""
        return self.audio_tokenizer.sampling_rate
