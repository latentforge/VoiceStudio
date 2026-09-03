import copy
import math

import torch
import torch.nn.functional as F

from transformers.generation import GenerationConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class OmniVoiceGenerationConfig(GenerationConfig):
    r"""
    A [`GenerationConfig`] parameterized for [`OmniVoiceGenerationMixin.generate`]. OmniVoice fills a fixed-length
    canvas of masked audio frames by iterative unmasking, so none of the autoregressive sampling, cache or
    stopping-criteria parameters of the parent class apply.

    Args:
        num_step (`int`, *optional*, defaults to 32):
            Number of unmasking steps. Every step commits a slice of the still-masked codebook entries.
        guidance_scale (`float`, *optional*, defaults to 2.0):
            Classifier-free guidance scale. `0` disables guidance and scores the conditional branch alone.
        t_shift (`float`, *optional*, defaults to 0.1):
            Shift applied to the linear timestep schedule. Values below `1` commit fewer entries in the early,
            low-confidence steps.
        layer_penalty_factor (`float`, *optional*, defaults to 5.0):
            Confidence penalty applied per codebook index, so that coarser codebooks are committed first.
        position_temperature (`float`, *optional*, defaults to 5.0):
            Temperature of the Gumbel noise added to the confidence scores that decide which positions to commit.
            `0` commits the most confident positions deterministically.
        class_temperature (`float`, *optional*, defaults to 0.0):
            Temperature of the Gumbel noise added to the top-10% filtered token log-probabilities. `0` selects the
            highest-probability code.
    """

    def __init__(self, **kwargs):
        # `GenerationConfig.__init__` is deliberately not called: none of the autoregressive parameters it defines
        # are read by `OmniVoiceGenerationMixin.generate`.
        self.num_step: int = kwargs.pop("num_step", 32)
        self.guidance_scale: float = kwargs.pop("guidance_scale", 2.0)
        self.t_shift: float = kwargs.pop("t_shift", 0.1)
        self.layer_penalty_factor: float = kwargs.pop("layer_penalty_factor", 5.0)
        self.position_temperature: float = kwargs.pop("position_temperature", 5.0)
        self.class_temperature: float = kwargs.pop("class_temperature", 0.0)

        self._commit_hash: str | None = kwargs.pop("_commit_hash", None)
        self._from_model_config: bool | None = kwargs.pop("_from_model_config", None)
        self.transformers_version: str | None = kwargs.pop("transformers_version", None)

        if len(kwargs) > 0:
            raise ValueError(f"Unexpected kwargs: {kwargs.keys()}")

        self.validate()

    def validate(self, **unused_kwargs):
        if not isinstance(self.num_step, int) or self.num_step <= 0:
            raise ValueError(f"`num_step` must be a positive integer, but got {self.num_step}")
        if self.guidance_scale < 0:
            raise ValueError(f"`guidance_scale` must be >= 0.0, but got {self.guidance_scale}")
        if self.t_shift <= 0:
            raise ValueError(f"`t_shift` must be > 0.0, but got {self.t_shift}")
        if self.position_temperature < 0:
            raise ValueError(f"`position_temperature` must be >= 0.0, but got {self.position_temperature}")
        if self.class_temperature < 0:
            raise ValueError(f"`class_temperature` must be >= 0.0, but got {self.class_temperature}")

    def get_generation_mode(self, *args, **kwargs):
        raise NotImplementedError("`OmniVoiceGenerationConfig` does not support `get_generation_mode`.")

    def from_model_config(self, *args, **kwargs):
        raise NotImplementedError("`OmniVoiceGenerationConfig` does not support `from_model_config`.")


def _get_time_steps(num_step: int, t_shift: float) -> list[float]:
    timesteps = torch.linspace(0.0, 1.0, num_step + 1)
    return (t_shift * timesteps / (1 + (t_shift - 1) * timesteps)).tolist()


def _gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_logits = logits / temperature
    uniform = torch.rand_like(scaled_logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise


def _filter_top_k(logits: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    k = math.ceil(ratio * logits.shape[-1])
    values, indices = logits.topk(k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, indices, values)
    return filtered


class OmniVoiceGenerationMixin:
    """
    Mixin holding the iterative unmasking loop of [`OmniVoiceForConditionalGeneration`].
    """

    def adjust_generation_fn(
        self,
        generation_config,
        from_auto_class,
        from_pipeline,
        pretrained_model_name_or_path,
        cache_dir,
        force_download,
        proxies,
        local_files_only,
        token,
        revision,
        subfolder,
        trust_remote_code,
        **kwargs,
    ):
        """
        Sets the model level generation config from the checkpoint being loaded.

        Args:
            generation_config ([`OmniVoiceGenerationConfig`], *optional*):
                Configuration passed to `from_pretrained`, which takes precedence over the checkpoint's own.
            from_auto_class (`bool`):
                Whether the model is being loaded through an `Auto` class.
            from_pipeline (`str`, *optional*):
                Name of the pipeline the model is being loaded for.
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Repository id or local path the generation config is read from.
            cache_dir, force_download, proxies, local_files_only, token, revision, subfolder:
                Forwarded to [`~OmniVoiceGenerationConfig.from_pretrained`].
            trust_remote_code (`bool`, *optional*):
                Unused.
            kwargs:
                Forwarded to [`~OmniVoiceGenerationConfig.from_pretrained`].
        """
        del trust_remote_code

        if generation_config is not None:
            self.generation_config = self.generation_config_class.from_dict(generation_config.to_dict())
        elif pretrained_model_name_or_path is not None:
            try:
                self.generation_config = self.generation_config_class.from_pretrained(
                    pretrained_model_name_or_path,
                    _from_auto=from_auto_class,
                    _from_pipeline=from_pipeline,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    **kwargs,
                )
            except OSError:
                logger.info("No generation config file was found; the defaults are used.")

    def _unmasking_schedule(self, target_length: int, generation_config: OmniVoiceGenerationConfig) -> list[int]:
        """
        Builds the per-step budget of codebook entries to commit for one canvas.

        Args:
            target_length (`int`):
                Number of audio frames in the canvas.
            generation_config ([`OmniVoiceGenerationConfig`]):
                Configuration supplying `num_step` and `t_shift`.

        Returns:
            `list[int]`: One entry per step, summing to `target_length * config.num_audio_codebook`.
        """
        timesteps = _get_time_steps(generation_config.num_step, generation_config.t_shift)
        total = target_length * self.config.num_audio_codebook
        remaining = total
        schedule = []
        for step in range(generation_config.num_step):
            if step == generation_config.num_step - 1:
                num_to_commit = remaining
            else:
                num_to_commit = min(math.ceil(total * (timesteps[step + 1] - timesteps[step])), remaining)
            schedule.append(int(num_to_commit))
            remaining -= int(num_to_commit)
        return schedule

    def _predict_tokens_with_scoring(
        self,
        conditional_logits: torch.Tensor,
        unconditional_logits: torch.Tensor,
        generation_config: OmniVoiceGenerationConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies classifier-free guidance and picks a candidate code for every position of one canvas.

        Args:
            conditional_logits (`torch.Tensor` of shape `(1, num_codebooks, target_length, audio_vocab_size)`):
                Logits of the forward pass that saw the style, text and reference-audio prefix.
            unconditional_logits (`torch.Tensor` of shape `(1, num_codebooks, target_length, audio_vocab_size)`):
                Logits of the forward pass that saw the canvas alone.
            generation_config ([`OmniVoiceGenerationConfig`]):
                Configuration supplying `guidance_scale` and `class_temperature`.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: The candidate codes and their confidence scores, both of shape
            `(1, num_codebooks, target_length)`.
        """
        if generation_config.guidance_scale != 0:
            conditional_log_probs = F.log_softmax(conditional_logits, dim=-1)
            unconditional_log_probs = F.log_softmax(unconditional_logits, dim=-1)
            log_probs = torch.log_softmax(
                conditional_log_probs
                + generation_config.guidance_scale * (conditional_log_probs - unconditional_log_probs),
                dim=-1,
            )
        else:
            log_probs = F.log_softmax(conditional_logits, dim=-1)

        log_probs[..., self.config.audio_mask_id] = -float("inf")

        if generation_config.class_temperature > 0.0:
            filtered_log_probs = _filter_top_k(log_probs, ratio=0.1)
            predicted_tokens = _gumbel_sample(filtered_log_probs, generation_config.class_temperature).argmax(dim=-1)
        else:
            predicted_tokens = log_probs.argmax(dim=-1)

        return predicted_tokens, log_probs.max(dim=-1)[0]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.BoolTensor,
        target_lengths: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        generation_config: OmniVoiceGenerationConfig | None = None,
        **kwargs,
    ) -> torch.LongTensor:
        r"""
        Fills the masked audio canvas at the end of every sequence in `input_ids` by iterative unmasking.

        Each step scores every still-masked codebook entry with classifier-free guidance, commits the highest
        scoring ones according to the timestep schedule, and feeds the partially filled canvas back in. The
        unconditional branch of the guidance is the canvas on its own, batched alongside the conditional sequences
        so that a single forward pass covers both.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, num_codebooks, sequence_length)`):
                Prompt built by [`OmniVoiceProcessor`], ending with `target_lengths` fully masked audio frames and
                right-padded to `sequence_length`.
            audio_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
                Marks the positions of `input_ids` that hold audio frames rather than text tokens.
            target_lengths (`torch.LongTensor` of shape `(batch_size,)`):
                Number of audio frames to generate for each item.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid attending to right-padding. Assumed all ones when not provided.
            generation_config ([`OmniVoiceGenerationConfig`], *optional*):
                Decoding parameters. Defaults to the model's own `generation_config`.
            kwargs:
                Field overrides for `generation_config`, e.g. `num_step=16`.

        Returns:
            `torch.LongTensor` of shape `(batch_size, num_codebooks, max(target_lengths))`: The generated audio
            codes. Frames past an item's own `target_lengths` stay at `config.audio_mask_id`, which
            [`~OmniVoiceProcessor.batch_decode`] trims.
        """
        if generation_config is None:
            generation_config = getattr(self, "generation_config", None)
        if not isinstance(generation_config, OmniVoiceGenerationConfig):
            generation_config = OmniVoiceGenerationConfig()
        if kwargs:
            generation_config = copy.deepcopy(generation_config)
            for key, value in kwargs.items():
                if not hasattr(generation_config, key):
                    raise ValueError(f"`{key}` is not a field of `OmniVoiceGenerationConfig`.")
                setattr(generation_config, key, value)
            generation_config.validate()

        device = self.device
        num_codebooks = self.config.num_audio_codebook
        mask_id = self.config.audio_mask_id

        input_ids = input_ids.to(device)
        audio_mask = audio_mask.to(device)
        batch_size, _, sequence_length = input_ids.shape

        if attention_mask is None:
            prompt_lengths = [sequence_length] * batch_size
        else:
            prompt_lengths = attention_mask.to(device).long().sum(-1).tolist()
        target_lengths = [int(length) for length in target_lengths]

        # Rows `[0, batch_size)` hold the conditional sequences and rows `[batch_size, 2 * batch_size)` the
        # canvas-only unconditional counterparts, so one forward pass covers both branches of the guidance.
        batch_input_ids = torch.full(
            (2 * batch_size, num_codebooks, sequence_length), mask_id, dtype=torch.long, device=device
        )
        batch_audio_mask = torch.zeros((2 * batch_size, sequence_length), dtype=torch.bool, device=device)
        batch_attention_mask = torch.zeros((2 * batch_size, sequence_length), dtype=torch.bool, device=device)

        for i in range(batch_size):
            prompt_length, target_length = prompt_lengths[i], target_lengths[i]
            canvas = slice(prompt_length - target_length, prompt_length)

            batch_input_ids[i, :, :prompt_length] = input_ids[i, :, :prompt_length]
            batch_audio_mask[i, :prompt_length] = audio_mask[i, :prompt_length]
            batch_attention_mask[i, :prompt_length] = True

            batch_input_ids[batch_size + i, :, :target_length] = input_ids[i, :, canvas]
            batch_audio_mask[batch_size + i, :target_length] = audio_mask[i, canvas]
            batch_attention_mask[batch_size + i, :target_length] = True

        max_target_length = max(target_lengths)
        audio_codes = torch.full(
            (batch_size, num_codebooks, max_target_length), mask_id, dtype=torch.long, device=device
        )
        schedules = [self._unmasking_schedule(length, generation_config) for length in target_lengths]
        codebook_ids = torch.arange(num_codebooks, device=device).view(1, -1, 1)

        for step in range(generation_config.num_step):
            batch_logits = self(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            ).logits.to(torch.float32)

            for i in range(batch_size):
                num_to_commit = schedules[i][step]
                if num_to_commit <= 0:
                    continue

                prompt_length, target_length = prompt_lengths[i], target_lengths[i]
                canvas = slice(prompt_length - target_length, prompt_length)
                conditional_logits = batch_logits[i : i + 1, :, canvas, :]
                unconditional_logits = batch_logits[batch_size + i : batch_size + i + 1, :, :target_length, :]

                predicted_tokens, scores = self._predict_tokens_with_scoring(
                    conditional_logits, unconditional_logits, generation_config
                )
                scores = scores - codebook_ids * generation_config.layer_penalty_factor
                if generation_config.position_temperature > 0.0:
                    scores = _gumbel_sample(scores, generation_config.position_temperature)

                committed = audio_codes[i : i + 1, :, :target_length]
                scores.masked_fill_(committed != mask_id, -float("inf"))

                _, topk_indices = torch.topk(scores.flatten(), num_to_commit)
                flat_committed = committed.flatten()
                flat_committed[topk_indices] = predicted_tokens.flatten()[topk_indices]
                committed.copy_(flat_committed.view_as(committed))

                batch_input_ids[i : i + 1, :, canvas] = committed
                batch_input_ids[batch_size + i : batch_size + i + 1, :, :target_length] = committed

        return audio_codes


__all__ = ["OmniVoiceGenerationConfig", "OmniVoiceGenerationMixin"]
