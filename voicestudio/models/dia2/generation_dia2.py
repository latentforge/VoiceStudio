"""Generation utilities for Dia2."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

import torch

from transformers.cache_utils import Cache

from .configuration_dia2 import Dia2Config


# Placeholder written into the codebook grid before a frame has been decoded.
UNGENERATED_CODE = -2


@dataclass
class Dia2ScriptEntry:
    """
    One word of the script, as consumed by [`Dia2TextStateMachine`].

    Args:
        token_ids (`list[int]`):
            Text tokens of the word. Empty for a pure silence entry.
        padding (`int`, *optional*, defaults to 0):
            Number of frames the state machine must stay on this entry before the next word may start.
    """

    token_ids: list[int]
    padding: int = 0


@dataclass
class Dia2TextStreamState:
    """
    Mutable state of a [`Dia2TextStateMachine`] run.

    Args:
        entries (`deque[Dia2ScriptEntry]`):
            Words that have not been reached yet.
        padding_budget (`int`):
            Frames still available before the action head is forced to advance to the next word.
        forced_padding (`int`):
            Frames during which the action head may not advance to the next word.
        pending_tokens (`deque[int]`):
            Tokens of the current word that the main stream has not emitted yet.
        lookahead_tokens (`deque[int]`):
            Tokens the second stream emits ahead of the main one.
        end_step (`int`, *optional*):
            Frame at which the script ran out of words.
        word_start_frames (`list[int]`):
            Frame at which each word was reached, in script order.
    """

    entries: deque
    padding_budget: int
    forced_padding: int
    pending_tokens: deque = field(default_factory=deque)
    lookahead_tokens: deque = field(default_factory=deque)
    end_step: int | None = None
    word_start_frames: list[int] = field(default_factory=list)

    def peek_tokens(self, count: int) -> list[int]:
        for entry in self.entries:
            if entry.token_ids:
                count -= 1
                if count == 0:
                    return entry.token_ids
        return []


class Dia2TextStateMachine:
    r"""
    Turns the binary action predicted by [`Dia2ForConditionalGeneration`]'s action head into the two text stream
    tokens fed back into the model on the next frame.

    The model never predicts text: it only decides, once per frame, whether to stay on the current word or to
    advance to the next one. This class owns the script cursor that decision drives.

    Args:
        config ([`Dia2Config`]):
            Configuration holding the text stream and action token ids.
        max_padding (`int`, *optional*, defaults to 6):
            Frames a word may be held before an advance is forced.
        initial_padding (`int`, *optional*, defaults to 0):
            Frames of silence before the first word may start.
    """

    def __init__(self, config: Dia2Config, max_padding: int = 6, initial_padding: int = 0):
        self.new_word_token_id = config.text_new_word_token_id
        self.pad_token_id = config.text_pad_token_id
        self.zero_token_id = config.text_zero_token_id
        self.action_new_word_token_id = config.action_new_word_token_id
        self.action_pad_token_id = config.action_pad_token_id
        self.second_stream_ahead = config.second_stream_ahead
        self.max_padding = max_padding
        self.initial_padding = initial_padding

    def new_state(self, entries: Iterable[Dia2ScriptEntry]) -> Dia2TextStreamState:
        r"""
        Args:
            entries (`Iterable[Dia2ScriptEntry]`):
                Words to speak, in order.

        Returns:
            [`Dia2TextStreamState`]: A fresh cursor over `entries`.
        """
        return Dia2TextStreamState(
            entries=deque(entries),
            padding_budget=self.initial_padding,
            forced_padding=self.initial_padding,
        )

    def process(
        self, step: int, state: Dia2TextStreamState, action: int, is_forced: bool = False
    ) -> tuple[int, int]:
        r"""
        Args:
            step (`int`):
                Index of the frame being decoded.
            state ([`Dia2TextStreamState`]):
                Cursor to advance in place.
            action (`int`):
                Action sampled from the action head, or a text stream token when `is_forced` is `True`.
            is_forced (`bool`, *optional*, defaults to `False`):
                Whether `action` comes from a known alignment rather than from the model.

        Returns:
            `tuple[int, int]`: The main and second text stream tokens for this frame.
        """
        token = self._sanitize(action)
        token = self._enforce_constraints(state, token, is_forced)
        token = self._handle_new_word(step, state, token)
        token = self._select_output(state, token)
        return self._multiplex_second_stream(state, token)

    def _sanitize(self, token: int) -> int:
        if token == self.action_new_word_token_id:
            token = self.new_word_token_id
        elif token == self.action_pad_token_id:
            token = self.pad_token_id
        if token not in (self.new_word_token_id, self.pad_token_id):
            return self.pad_token_id
        return token

    def _enforce_constraints(self, state: Dia2TextStreamState, token: int, is_forced: bool) -> int:
        if state.pending_tokens:
            return self.pad_token_id
        if is_forced:
            return token
        if state.forced_padding > 0:
            return self.pad_token_id
        if state.padding_budget <= 0 and token != self.new_word_token_id:
            return self.new_word_token_id
        return token

    def _handle_new_word(self, step: int, state: Dia2TextStreamState, token: int) -> int:
        if token != self.new_word_token_id:
            return token
        if state.entries:
            entry = state.entries.popleft()
            if entry.token_ids:
                state.word_start_frames.append(step)
                state.pending_tokens.extend(entry.token_ids)
                if self.second_stream_ahead:
                    state.lookahead_tokens.extend(state.peek_tokens(self.second_stream_ahead))
                state.padding_budget = self.max_padding
            else:
                token = self.pad_token_id
            state.forced_padding = entry.padding
            return token
        token = self.pad_token_id
        if self.second_stream_ahead and state.end_step is None:
            token = self.new_word_token_id
        if state.end_step is None:
            state.end_step = step
        return token

    def _select_output(self, state: Dia2TextStreamState, token: int) -> int:
        if token == self.pad_token_id:
            if state.padding_budget > 0:
                state.padding_budget -= 1
            if state.forced_padding > 0:
                state.forced_padding -= 1
            if state.pending_tokens:
                return state.pending_tokens.popleft()
            return self.pad_token_id
        if token in (self.new_word_token_id, self.zero_token_id):
            return token
        raise ValueError(f"Invalid text stream token {token}")

    def _multiplex_second_stream(self, state: Dia2TextStreamState, token: int) -> tuple[int, int]:
        if not self.second_stream_ahead:
            return token, token
        if token == self.new_word_token_id:
            second = self.new_word_token_id
            token = state.pending_tokens.popleft() if state.pending_tokens else self.pad_token_id
        elif state.lookahead_tokens:
            second = state.lookahead_tokens.popleft()
        else:
            second = self.pad_token_id
        return token, second


def apply_delay_pattern(codes: torch.Tensor, delay_pattern: list[int], pad_token_id: int) -> torch.Tensor:
    r"""
    Args:
        codes (`torch.LongTensor` of shape `(num_codebooks, num_frames)`):
            Time-aligned codebook grid.
        delay_pattern (`list[int]`):
            Per-codebook delay in frames.
        pad_token_id (`int`):
            Id written into the slots the shift leaves empty.

    Returns:
        `torch.LongTensor` of shape `(num_codebooks, num_frames + max(delay_pattern))`: The delayed grid.
    """
    num_codebooks, num_frames = codes.shape
    max_delay = max(delay_pattern) if delay_pattern else 0
    delayed = codes.new_full((num_codebooks, num_frames + max_delay), pad_token_id)
    for codebook, delay in enumerate(delay_pattern):
        delayed[codebook, delay : delay + num_frames] = codes[codebook]
    return delayed


def revert_delay_pattern(codes: torch.Tensor, delay_pattern: list[int], pad_token_id: int) -> torch.Tensor:
    r"""
    Args:
        codes (`torch.LongTensor` of shape `(num_codebooks, num_frames)`):
            Delayed codebook grid.
        delay_pattern (`list[int]`):
            Per-codebook delay in frames.
        pad_token_id (`int`):
            Id written into the slots the shift leaves empty.

    Returns:
        `torch.LongTensor` of shape `(num_codebooks, num_frames - max(delay_pattern))`: The time-aligned grid.
    """
    num_codebooks, num_frames = codes.shape
    max_delay = max(delay_pattern) if delay_pattern else 0
    length = max(0, num_frames - max_delay)
    aligned = codes.new_full((num_codebooks, length), pad_token_id)
    for codebook, delay in enumerate(delay_pattern):
        aligned[codebook] = codes[codebook, delay : delay + length]
    return aligned


def mask_audio_logits(logits: torch.Tensor, *forbidden_ids: int) -> torch.Tensor:
    r"""
    Args:
        logits (`torch.FloatTensor` of shape `(..., codebook_size)`):
            Codebook logits to constrain.
        forbidden_ids (`int`):
            Codebook ids that must never be sampled.

    Returns:
        `torch.FloatTensor` of shape `(..., codebook_size)`: `logits` with `forbidden_ids` driven to the dtype's
        minimum.
    """
    targets = [index for index in forbidden_ids if 0 <= index < logits.shape[-1]]
    if not targets:
        return logits
    masked = logits.clone()
    masked[..., targets] = torch.finfo(masked.dtype).min
    return masked


def apply_classifier_free_guidance(logits: torch.Tensor, guidance_scale: float, top_k: int) -> torch.Tensor:
    r"""
    Args:
        logits (`torch.FloatTensor` of shape `(2, sequence_length, vocab_size)`):
            Conditional logits on row 0 and unconditional logits on row 1.
        guidance_scale (`float`):
            Interpolation weight of the conditional logits.
        top_k (`int`):
            Number of candidates the guided distribution keeps. `0` keeps all of them.

    Returns:
        `torch.FloatTensor` of shape `(1, sequence_length, vocab_size)`: The conditional logits, restricted to the
        candidates the guided distribution ranks highest.
    """
    conditional = logits[0:1].float()
    unconditional = logits[1:2].float()
    guided = torch.lerp(unconditional, conditional, guidance_scale)
    if top_k > 0 and guided.shape[-1] > 0:
        threshold = torch.topk(guided, k=min(top_k, guided.shape[-1]), dim=-1, sorted=False).values[..., -1:]
        conditional = torch.where(guided >= threshold, conditional, torch.full_like(conditional, float("-inf")))
    return conditional.to(logits.dtype)


def sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    r"""
    Args:
        logits (`torch.FloatTensor` of shape `(..., vocab_size)`):
            Logits to sample from.
        temperature (`float`):
            Sampling temperature. Values at or below `0` select the argmax.
        top_k (`int`):
            Number of highest-probability candidates to keep. `0` keeps all of them.

    Returns:
        `torch.LongTensor` of shape `(..., 1)`: One sampled id per row.
    """
    logits = logits.float()
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    flat = probs.reshape(-1, probs.shape[-1])
    total = flat.sum(dim=-1, keepdim=True)
    # A row whose probabilities all underflowed to zero would make `multinomial` raise; give it id 0 instead.
    fallback = torch.zeros_like(flat)
    fallback[..., 0] = 1.0
    flat = torch.where((total <= 0).expand_as(flat), fallback, flat / total.clamp_min(1e-12))

    if 0 < top_k < flat.shape[-1]:
        values, indices = torch.topk(flat, top_k, dim=-1)
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        picks = torch.gather(indices, dim=-1, index=torch.multinomial(values, num_samples=1))
    else:
        picks = torch.multinomial(flat, num_samples=1)
    return picks.reshape(*probs.shape[:-1], 1)


class Dia2GenerationMixin:
    """Streaming dialogue decoding loop of [`Dia2ForConditionalGeneration`]."""

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        word_lengths: torch.LongTensor,
        word_paddings: torch.LongTensor,
        prefix_audio_codes: torch.LongTensor | None = None,
        prefix_word_start_frames: list[int] | None = None,
        max_new_frames: int | None = None,
        guidance_scale: float = 2.0,
        guidance_top_k: int = 50,
        action_temperature: float = 0.6,
        action_top_k: int = 50,
        audio_temperature: float = 0.8,
        audio_top_k: int = 50,
        initial_padding: int = 2,
        max_word_padding: int = 6,
        keep_prefix_audio: bool = False,
    ) -> torch.LongTensor:
        r"""
        Autoregressively decodes the script into a time-aligned Mimi codebook grid.

        Args:
            input_ids (`torch.LongTensor` of shape `(1, sequence_length)`):
                Text tokens of every word of the script, concatenated, as built by [`Dia2Processor`].
            word_lengths (`torch.LongTensor` of shape `(1, num_words)`):
                Number of `input_ids` tokens belonging to each word.
            word_paddings (`torch.LongTensor` of shape `(1, num_words)`):
                Number of frames each word must be held before the next one may start.
            prefix_audio_codes (`torch.LongTensor` of shape `(1, num_codebooks, num_prefix_frames)`, *optional*):
                Time-aligned codes of the conditioning audio, whose transcript must be the first
                `len(prefix_word_start_frames)` words of `input_ids`.
            prefix_word_start_frames (`list[int]`, *optional*):
                Frame at which each conditioning word starts, as returned by [`Dia2Processor`].
            max_new_frames (`int`, *optional*):
                Maximum number of frames to decode. Defaults to `config.max_position_embeddings`.
            guidance_scale (`float`, *optional*, defaults to 2.0):
                Classifier-free guidance scale. `1.0` disables guidance and halves the batch that is run.
            guidance_top_k (`int`, *optional*, defaults to 50):
                Number of candidates the guided distribution keeps.
            action_temperature (`float`, *optional*, defaults to 0.6):
                Sampling temperature of the action head.
            action_top_k (`int`, *optional*, defaults to 50):
                Top-k of the action head.
            audio_temperature (`float`, *optional*, defaults to 0.8):
                Sampling temperature of the codebook heads.
            audio_top_k (`int`, *optional*, defaults to 50):
                Top-k of the codebook heads.
            initial_padding (`int`, *optional*, defaults to 2):
                Frames of silence before the first word may start.
            max_word_padding (`int`, *optional*, defaults to 6):
                Frames a word may be held before an advance to the next word is forced.
            keep_prefix_audio (`bool`, *optional*, defaults to `False`):
                Whether the conditioning audio stays in the returned codes.

        Returns:
            `torch.LongTensor` of shape `(1, num_codebooks, num_frames)`: Time-aligned codes, ready for
            [`~Dia2Processor.decode`].

        Raises:
            ValueError: If more than one script is passed, or if a prefix is passed without its word alignment.
        """
        if input_ids.shape[0] != 1:
            raise ValueError("Dia2ForConditionalGeneration.generate only supports one script at a time.")
        if (prefix_audio_codes is None) != (prefix_word_start_frames is None):
            raise ValueError("`prefix_audio_codes` and `prefix_word_start_frames` must be passed together.")

        config = self.config
        device = input_ids.device
        num_codebooks = config.num_codebooks
        delay_pattern = list(config.delay_pattern)
        max_delay = config.max_delay
        max_new_frames = max_new_frames or config.max_position_embeddings

        entries = self._build_script_entries(input_ids, word_lengths, word_paddings)
        machine = Dia2TextStateMachine(config, max_padding=max_word_padding, initial_padding=initial_padding)
        state = machine.new_state(entries)

        branches = 2 if guidance_scale != 1.0 else 1
        frame_tokens = torch.full(
            (branches, 1, config.num_channels), config.text_pad_token_id, dtype=torch.long, device=device
        )
        frame_tokens[0, 0, 0] = config.text_bos_token_id
        if branches > 1:
            frame_tokens[1, 0, 0] = config.text_zero_token_id

        num_aligned_frames = 0
        prefix_codes = None
        if prefix_audio_codes is not None:
            num_aligned_frames = prefix_audio_codes.shape[-1]
            prefix_codes = apply_delay_pattern(
                prefix_audio_codes[0].to(device), delay_pattern, config.codebook_pad_token_id
            )

        total_frames = max_new_frames + (num_aligned_frames + max_delay if prefix_codes is not None else 0) + 1
        codes = torch.full(
            (branches, num_codebooks, total_frames), UNGENERATED_CODE, dtype=torch.long, device=device
        )
        if prefix_codes is not None:
            codes[:, :, : prefix_codes.shape[-1]] = prefix_codes

        delays = torch.tensor(delay_pattern, dtype=torch.long, device=device)
        past_key_values = None
        start_frame = 0
        if prefix_codes is not None:
            start_frame, past_key_values = self._warm_up_prefix(
                frame_tokens, codes, delays, num_aligned_frames, prefix_word_start_frames, machine, state
            )

        first_word_frame = None
        stop_frame = None
        last_frame = start_frame - 1
        flush_tail = max_delay + max_word_padding

        for offset in range(max_new_frames):
            frame = start_frame + offset
            if stop_frame is not None and frame >= stop_frame:
                break
            if frame + 1 >= total_frames:
                break

            self._fill_codebook_channels(frame_tokens, codes, delays, frame)
            if branches > 1:
                frame_tokens[1:, 0, 0] = config.text_zero_token_id
                frame_tokens[1:, 0, 1] = config.text_pad_token_id

            outputs = self.backbone_model(
                input_ids=frame_tokens,
                position_ids=torch.tensor([[frame]], device=device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            action_logits = self.action_head(hidden_states)
            if branches > 1:
                action_logits = apply_classifier_free_guidance(action_logits, guidance_scale, guidance_top_k)
            action = sample_from_logits(action_logits, action_temperature, action_top_k).item()

            main_token, second_token = machine.process(frame, state, action)
            if first_word_frame is None and main_token == config.text_new_word_token_id:
                first_word_frame = frame - initial_padding
            frame_tokens[:, 0, 0] = main_token
            frame_tokens[:, 0, 1] = second_token

            codebook_logits = self.lm_head(hidden_states)
            if branches > 1:
                codebook_logits = apply_classifier_free_guidance(codebook_logits, guidance_scale, guidance_top_k)
            codebook_logits = mask_audio_logits(
                codebook_logits, config.codebook_pad_token_id, config.codebook_bos_token_id
            )
            code = sample_from_logits(codebook_logits, audio_temperature, audio_top_k).view(1)
            codes[:, 0, frame + 1] = code

            depth_cache = None
            previous_code = code.expand(branches)
            text_tokens = torch.full((branches,), main_token, dtype=torch.long, device=device)
            second_text_tokens = torch.full((branches,), second_token, dtype=torch.long, device=device)
            for codebook in range(num_codebooks - 1):
                depth_outputs = self.depth_decoder(
                    input_ids=previous_code.unsqueeze(-1),
                    backbone_hidden_states=hidden_states[:, -1, :],
                    text_input_ids=text_tokens,
                    second_text_input_ids=second_text_tokens,
                    past_key_values=depth_cache,
                    use_cache=True,
                )
                depth_cache = depth_outputs.past_key_values
                depth_logits = mask_audio_logits(
                    depth_outputs.logits, config.codebook_pad_token_id, config.codebook_bos_token_id
                )
                if branches > 1:
                    depth_logits = apply_classifier_free_guidance(depth_logits, guidance_scale, guidance_top_k)
                next_code = sample_from_logits(depth_logits, audio_temperature, audio_top_k).view(1)
                codes[:, codebook + 1, frame + 1] = next_code
                previous_code = next_code.expand(branches)

            last_frame = frame
            if stop_frame is None and state.end_step is not None:
                stop_frame = state.end_step + flush_tail

        limit = min(max(last_frame + 2, start_frame + 1), total_frames)
        generated = codes[0, :, :limit]
        generated = torch.where(
            generated == UNGENERATED_CODE, torch.full_like(generated, config.codebook_pad_token_id), generated
        )
        aligned = revert_delay_pattern(generated, delay_pattern, config.codebook_pad_token_id)

        crop = 0 if keep_prefix_audio else max(first_word_frame if first_word_frame is not None else start_frame, 0)
        if 0 < crop < aligned.shape[-1]:
            aligned = aligned[:, crop:]
        return aligned.unsqueeze(0)

    def _build_script_entries(
        self, input_ids: torch.LongTensor, word_lengths: torch.LongTensor, word_paddings: torch.LongTensor
    ) -> list[Dia2ScriptEntry]:
        token_ids = input_ids[0].tolist()
        lengths = word_lengths[0].tolist()
        paddings = word_paddings[0].tolist()
        entries = []
        offset = 0
        for length, padding in zip(lengths, paddings):
            entries.append(Dia2ScriptEntry(token_ids[offset : offset + length], padding))
            offset += length
        return entries

    def _fill_codebook_channels(
        self, frame_tokens: torch.Tensor, codes: torch.Tensor, delays: torch.Tensor, frame: int
    ) -> None:
        target = frame_tokens[:, 0, 2:]
        if frame < codes.shape[-1]:
            target.copy_(codes[:, :, frame])
        else:
            target.fill_(self.config.codebook_bos_token_id)
        target.masked_fill_(delays.unsqueeze(0) > frame, self.config.codebook_bos_token_id)
        target.masked_fill_(target == UNGENERATED_CODE, self.config.codebook_bos_token_id)

    def _warm_up_prefix(
        self,
        frame_tokens: torch.Tensor,
        codes: torch.Tensor,
        delays: torch.Tensor,
        num_aligned_frames: int,
        prefix_word_start_frames: list[int],
        machine: Dia2TextStateMachine,
        state: Dia2TextStreamState,
    ) -> tuple[int, Cache]:
        config = self.config
        device = frame_tokens.device
        branches = frame_tokens.shape[0]
        new_word_frames = {int(frame) for frame in prefix_word_start_frames}
        past_key_values = None

        for frame in range(num_aligned_frames):
            self._fill_codebook_channels(frame_tokens, codes, delays, frame)
            outputs = self.backbone_model(
                input_ids=frame_tokens,
                position_ids=torch.tensor([[frame]], device=device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            forced = config.text_new_word_token_id if frame in new_word_frames else config.text_pad_token_id
            main_token, second_token = machine.process(frame, state, forced, is_forced=True)
            frame_tokens[0, 0, 0] = main_token
            frame_tokens[0, 0, 1] = second_token
            if branches > 1:
                frame_tokens[1:, 0, 0] = config.text_zero_token_id
                frame_tokens[1:, 0, 1] = config.text_pad_token_id

        return max(num_aligned_frames - 1, 0), past_key_values


__all__ = [
    "Dia2GenerationMixin",
    "Dia2ScriptEntry",
    "Dia2TextStateMachine",
    "Dia2TextStreamState",
    "apply_delay_pattern",
    "revert_delay_pattern",
]
