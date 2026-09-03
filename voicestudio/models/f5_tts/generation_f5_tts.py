"""Flow matching sampling for F5-TTS."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers.utils import ModelOutput


EPSS_TIMESTEPS = {
    5: [0, 2, 4, 8, 16, 32],
    6: [0, 2, 4, 6, 8, 16, 32],
    7: [0, 2, 4, 6, 8, 16, 24, 32],
    10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
    12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
}


def get_epss_timesteps(num_steps: int, device, dtype) -> torch.Tensor:
    r"""
    Builds the empirically pruned step sampling schedule for a low number of function evaluations.

    Args:
        num_steps (`int`):
            Number of solver steps. Values without a tabulated schedule fall back to a uniform one.
        device (`torch.device`):
            Device the schedule is built on.
        dtype (`torch.dtype`):
            Dtype of the schedule.

    Returns:
        `torch.Tensor`: Schedule of shape `(num_steps + 1,)` rising from 0 to 1.
    """
    timesteps = EPSS_TIMESTEPS.get(num_steps, [])
    if not timesteps:
        return torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)
    return (1 / 32) * torch.tensor(timesteps, device=device, dtype=dtype)


class F5TTSFixedStepODESolver:
    r"""
    Fixed step ordinary differential equation solver over an explicit time grid.

    Args:
        function (`Callable`):
            Vector field, called as `function(time, value)`.
        initial_value (`torch.Tensor`):
            State at the first time point of the grid.
        method (`str`, *optional*, defaults to `"euler"`):
            Integration rule, one of `"euler"` or `"midpoint"`.

    Raises:
        ValueError: If `method` is neither `"euler"` nor `"midpoint"`.
    """

    def __init__(self, function: Callable, initial_value: torch.Tensor, method: str = "euler"):
        if method not in ("euler", "midpoint"):
            raise ValueError(f"`method` must be one of 'euler' or 'midpoint', got {method}.")
        self.function = function
        self.initial_value = initial_value
        self.method = method

    def _compute_step(self, time_start, time_step, value_start):
        if self.method == "euler":
            return time_step * self.function(time_start, value_start)
        half_step = 0.5 * time_step
        value_mid = value_start + self.function(time_start, value_start) * half_step
        return time_step * self.function(time_start + half_step, value_mid)

    def _linear_interpolation(self, time_start, time_end, value_start, value_end, time_point):
        if time_point == time_start:
            return value_start
        if time_point == time_end:
            return value_end
        weight = (time_point - time_start) / (time_end - time_start)
        return value_start + weight * (value_end - value_start)

    def integrate(self, time_points: torch.Tensor) -> torch.Tensor:
        r"""
        Integrates the vector field over the given time grid.

        Args:
            time_points (`torch.Tensor`):
                Strictly increasing grid of shape `(num_points,)`, whose first entry the initial value belongs to.

        Returns:
            `torch.Tensor`: Trajectory of shape `(num_points, *initial_value.shape)`.
        """
        solution = torch.empty(
            len(time_points),
            *self.initial_value.shape,
            dtype=self.initial_value.dtype,
            device=self.initial_value.device,
        )
        solution[0] = self.initial_value

        current_index = 1
        current_value = self.initial_value
        for time_start, time_end in zip(time_points[:-1], time_points[1:]):
            time_step = time_end - time_start
            next_value = current_value + self._compute_step(time_start, time_step, current_value)

            while current_index < len(time_points) and time_end >= time_points[current_index]:
                solution[current_index] = self._linear_interpolation(
                    time_start, time_end, current_value, next_value, time_points[current_index]
                )
                current_index += 1

            current_value = next_value

        return solution


@dataclass
class F5TTSGenerationOutput(ModelOutput):
    r"""
    Output of [`~F5TTSGenerationMixin.generate`].

    Args:
        mel_spectrogram (`torch.FloatTensor` of shape `(batch_size, sequence_length, mel_dim)`):
            Generated log mel spectrogram, carrying the reference speech on the conditioned frames.
        trajectory (`torch.FloatTensor` of shape `(num_steps + 1, batch_size, sequence_length, mel_dim)`,
        *optional*):
            State of the flow at every point of the solver's time grid.
    """

    mel_spectrogram: torch.FloatTensor | None = None
    trajectory: torch.FloatTensor | None = None


class F5TTSGenerationMixin:
    r"""
    Sampling loop of a conditional flow matching text to speech model. Integrates the vector field predicted by the
    backbone from Gaussian noise up to the data distribution, with classifier free guidance, sway sampling and
    optional empirically pruned step sampling, and restores the reference speech on the conditioned frames.
    """

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        conditioning_features: torch.Tensor,
        duration,
        attention_mask: torch.Tensor | None = None,
        num_steps: int = 32,
        guidance_scale: float = 2.0,
        sway_sampling_coef: float | None = -1.0,
        ode_method: str = "euler",
        use_epss: bool = True,
        max_duration: int = 65536,
        no_ref_audio: bool = False,
        edit_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        return_trajectory: bool = False,
    ) -> F5TTSGenerationOutput:
        r"""
        Args:
            input_ids (`torch.Tensor` of shape `(batch_size, text_length)`):
                Character ids of the reference transcription followed by the text to speak, padded with the filler
                id `0`.
            conditioning_features (`torch.FloatTensor` of shape `(batch_size, ref_length, mel_dim)`):
                Log mel spectrogram of the reference speech.
            duration (`int` or `torch.Tensor` of shape `(batch_size,)`):
                Total number of mel frames to produce, reference frames included.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, ref_length)`, *optional*):
                Mask over the reference frames, `True` on valid frames.
            num_steps (`int`, *optional*, defaults to 32):
                Number of solver steps, that is the number of function evaluations of the backbone.
            guidance_scale (`float`, *optional*, defaults to 2.0):
                Classifier free guidance scale. Below `1e-5` the unconditional branch is skipped entirely.
            sway_sampling_coef (`float`, *optional*, defaults to -1.0):
                Coefficient of the sway reshaping of the time grid. `None` leaves the grid as is.
            ode_method (`str`, *optional*, defaults to `"euler"`):
                Integration rule, one of `"euler"` or `"midpoint"`.
            use_epss (`bool`, *optional*, defaults to `True`):
                Whether to use the empirically pruned step sampling grid instead of a uniform one.
            max_duration (`int`, *optional*, defaults to 65536):
                Upper bound on the number of mel frames produced.
            no_ref_audio (`bool`, *optional*, defaults to `False`):
                Whether to zero the reference speech, leaving the text as the only conditioning.
            edit_mask (`torch.BoolTensor` of shape `(batch_size, ref_length)`, *optional*):
                Extra mask ANDed onto the reference mask, to keep only part of the reference speech.
            generator (`torch.Generator`, *optional*):
                Generator the initial noise is drawn with. It is rewound to the same state for every entry of the
                batch, so one entry's spectrogram does not depend on how many others it was generated with.
            return_trajectory (`bool`, *optional*, defaults to `False`):
                Whether to return the state of the flow at every point of the time grid.

        Returns:
            [`F5TTSGenerationOutput`]
        """
        self.eval()
        backbone = self.get_backbone()

        conditioning_features = conditioning_features.to(next(self.parameters()).dtype)
        batch_size, cond_seq_len = conditioning_features.shape[:2]
        device, dtype = conditioning_features.device, conditioning_features.dtype

        if attention_mask is None:
            lengths = torch.full((batch_size,), cond_seq_len, device=device, dtype=torch.long)
        else:
            lengths = attention_mask.sum(dim=1).to(torch.long)

        positions = torch.arange(int(lengths.amax()), device=device)
        cond_mask = positions[None, :] < lengths[:, None]
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch_size,), duration, device=device, dtype=torch.long)
        duration = torch.maximum(torch.maximum((input_ids != 0).sum(dim=-1), lengths) + 1, duration)
        duration = duration.clamp(max=max_duration)
        total_duration = int(duration.amax())

        conditioning_features = F.pad(
            conditioning_features, (0, 0, 0, total_duration - cond_seq_len), value=0.0
        )
        if no_ref_audio:
            conditioning_features = torch.zeros_like(conditioning_features)

        cond_mask = F.pad(cond_mask, (0, total_duration - cond_mask.shape[-1]), value=False).unsqueeze(-1)
        step_cond = torch.where(cond_mask, conditioning_features, torch.zeros_like(conditioning_features))

        if batch_size > 1:
            frame_positions = torch.arange(total_duration, device=device)
            padding_mask = frame_positions[None, :] < duration[:, None]
        else:
            padding_mask = None

        def ode_function(time_step, hidden_states):
            if guidance_scale < 1e-5:
                return self(
                    input_ids=input_ids,
                    input_features=hidden_states,
                    conditioning_features=step_cond,
                    timestep=time_step,
                    attention_mask=padding_mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                ).vector_field

            vector_field = self(
                input_ids=input_ids,
                input_features=hidden_states,
                conditioning_features=step_cond,
                timestep=time_step,
                attention_mask=padding_mask,
                cfg_infer=True,
                cache=True,
            ).vector_field
            prediction, null_prediction = torch.chunk(vector_field, 2, dim=0)
            return prediction + (prediction - null_prediction) * guidance_scale

        # Rewinding the generator per entry keeps a batched run bit identical to the same entries run alone.
        generator_state = generator.get_state() if generator is not None else None
        samples = []
        for length in duration:
            if generator_state is not None:
                generator.set_state(generator_state)
            samples.append(torch.randn(int(length), self.mel_dim, device=device, dtype=dtype, generator=generator))
        noise = pad_sequence(samples, padding_value=0, batch_first=True)

        if use_epss:
            time_points = get_epss_timesteps(num_steps, device=device, dtype=dtype)
        else:
            time_points = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)
        if sway_sampling_coef is not None:
            time_points = time_points + sway_sampling_coef * (
                torch.cos(torch.pi / 2 * time_points) - 1 + time_points
            )

        solver = F5TTSFixedStepODESolver(function=ode_function, initial_value=noise, method=ode_method)
        trajectory = solver.integrate(time_points)
        backbone.clear_cache()

        mel_spectrogram = torch.where(cond_mask, conditioning_features, trajectory[-1])

        return F5TTSGenerationOutput(
            mel_spectrogram=mel_spectrogram,
            trajectory=trajectory if return_trajectory else None,
        )


__all__ = ["F5TTSFixedStepODESolver", "F5TTSGenerationMixin", "F5TTSGenerationOutput", "get_epss_timesteps"]
